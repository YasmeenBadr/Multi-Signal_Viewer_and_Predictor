# This file defines a Flask blueprint for an ECG analysis application.
# It handles loading a deep learning model (HuBERTECG), streaming ECG data
# (either from WFDB records or simulation), and performing real-time prediction
# on the streaming segments.

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Blueprint, request, jsonify, render_template

# NOTE: The HuBERTECG and HuBERTECGConfig imports assume these are available
# in the environment (e.g., in a local hubert_ecg package).
# If the environment is purely standard Python, these imports may fail.
try:
    from hubert_ecg.hubert_ecg import HuBERTECG, HuBERTECGConfig
except ImportError:
    # Placeholder classes for environment without external library
    class HuBERTECGConfig:
        def _init_(self, **kwargs):
            # Default hidden size as used in the standard model
            self.hidden_size = 768
    
    class HuBERTECG(nn.Module):
        """Mock HuBERTECG class for environments without the library."""
        def _init_(self, config):
            super()._init_()
            # Simulate the feature extraction output
            self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        
        def forward(self, x):
            # Simulate an output structure with last_hidden_state
            batch_size, seq_len = x.shape
            # Create a dummy feature vector: sequence length is reduced (e.g., / 10)
            dummy_output = torch.rand(batch_size, seq_len // 10, self.linear.out_features)
            return {"last_hidden_state": dummy_output}


# Optional libraries needed for WFDB data loading and training (pandas is required for PTB-XL)
try:
    import wfdb
except Exception:
    wfdb = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from scipy.signal import resample as scipy_resample
except Exception:
    scipy_resample = None

# -------------------------
# Blueprint (export bp for existing app.py)
# -------------------------
ECG_BP = Blueprint("ecg", __name__, url_prefix="/ecg", template_folder="../templates")
bp = ECG_BP      # keep ecg.bp compatible with your app.py

# -------------------------
# Configuration
# -------------------------
CKPT = "hubert_ecg/hubert_ecg_small_resaved.pt"    # adjust path if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease classes we'll use (can be extended)
DISEASE_CLASSES = ["Normal", "MI", "Conduction", "Hypertrophy", "STTC", "Other"]

# Streaming state - holds the current position and signal data
_stream = {
    "loaded": False,        # True if a real WFDB record was loaded
    "signals": None,        # The full numpy signal array
    "channels": None,       # List of channel/lead names
    "fs": 500,              # Original sampling frequency
    "pos": 0,               # Current reading position in the signal array
    "global_idx": 0,        # Index for global time calculation
    "record_path": None
}

DISPLAY_FS = 200        # Downsampled fs for frontend visualization (optimizes transfer size)
STREAMING_CHUNK_DURATION = 1.0 # The duration of the data chunk (in seconds) to send per update

# -------------------------
# Model loading
# -------------------------
# Safely add HuBERTECGConfig to safe globals if torch version supports it
try:
    # This prevents pickle/unpickle errors when loading models saved with custom classes
    torch.serialization.add_safe_globals([HuBERTECGConfig])
except AttributeError:
    pass # Ignore if torch version is older

def _safe_load_checkpoint(path):
    """Loads a PyTorch checkpoint, handling potential missing files."""
    if not os.path.exists(path):
        # Allow running without checkpoint for demonstration purposes
        print(f"Checkpoint not found at {path}. Model will be initialized randomly.")
        return None
    try:
        # Try loading with weights_only=False to allow loading custom classes/configs
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint with weights_only=False: {e}. Trying simple load.")
        ckpt = torch.load(path, map_location="cpu")
    return ckpt

def _build_model_from_ckpt(path):
    """Initializes model and loads state dictionary from a checkpoint."""
    ckpt = _safe_load_checkpoint(path)
    
    # Try to get config
    if isinstance(ckpt, dict) and "model_config" in ckpt:
        mc = ckpt["model_config"]
        config_obj = HuBERTECGConfig(**mc) if isinstance(mc, dict) else mc
    else:
        config_obj = HuBERTECGConfig()

    model = HuBERTECG(config_obj)
    
    # Load state dict
    state = None
    if isinstance(ckpt, dict):
        # Look for common state dict keys
        for k in ("model_state_dict", "model", "state_dict", "model_state"):
            if k in ckpt:
                state = ckpt[k]
                break
    if state is None and ckpt is not None:
        # If ckpt is just the state dict itself
        state = ckpt
        
    if state:
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print("Warning: model.load_state_dict strict load failed:", e)
            # Try again with strict=False in case of unexpected layer mismatches
            model.load_state_dict(state, strict=False)

    return model, config_obj

# Load model
try:
    print("Loading HuBERTECG model from:", CKPT)
    MODEL, MODEL_CONFIG = _build_model_from_ckpt(CKPT)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    print("HuBERTECG loaded successfully.")
except Exception as e:
    print("Could not load HuBERTECG checkpoint or model setup failed:", e)
    print("Using untrained HuBERTECG model with mock weights.")
    MODEL_CONFIG = HuBERTECGConfig()
    MODEL = HuBERTECG(MODEL_CONFIG).to(DEVICE)
    MODEL.eval()

# Classifier head: Fine-tuning layer for the downstream task (disease classification)
_in_features = getattr(MODEL_CONFIG, "hidden_size",
                        getattr(MODEL, "config", {}).hidden_size if hasattr(MODEL, "config") else 768)
classifier_head = nn.Sequential(
    nn.Linear(_in_features, 128),
    nn.ReLU(),
    nn.Linear(128, len(DISEASE_CLASSES))
).to(DEVICE)
classifier_head.eval()

# -------------------------
# WFDB record loader & Data Simulation
# -------------------------
def find_first_wfdb_record(data_root="data"):
    """Searches for the first WFDB record (.dat file) in the data directory."""
    if wfdb is None:
        return None
    for root, dirs, files in os.walk(data_root):
        for fname in files:
            if fname.lower().endswith(".dat"):
                # Return the base path (excluding .dat extension)
                return os.path.join(root, fname[:-4])
    return None

def load_wfdb_record(record_base):
    """Loads signal data and metadata from a WFDB record base path."""
    rec = wfdb.rdrecord(record_base)
    # Convert signals to float32 for PyTorch compatibility
    signals = rec.p_signal.astype(np.float32)
    sig_names = rec.sig_name
    fs = int(rec.fs) if hasattr(rec, "fs") else 500
    return signals, sig_names, fs

# Attempt to load a real record first
_record_base = find_first_wfdb_record("data")
if _record_base and wfdb is not None:
    try:
        s, names, sr = load_wfdb_record(_record_base)
        _stream.update({
            "signals": s,
            "channels": names,
            "fs": sr,
            "pos": 0,
            "global_idx": 0,
            "loaded": True,
            "record_path": _record_base
        })
        print(f"Loaded WFDB record {_record_base} with fs={sr}, channels={len(names)}")
    except Exception as e:
        print("Failed to load WFDB record:", e)
        _stream["loaded"] = False
else:
    print("No WFDB record found or WFDB library not available — using simulated ECG.")
    _stream["loaded"] = False

# If loading failed, generate a dummy signal
if not _stream["loaded"]:
    fs = 360 # Common simulation FS
    _stream["fs"] = fs
    # Generate 300 seconds of data
    t = np.linspace(0, 300, 300*fs, endpoint=False)
    
    # Simulate 12-lead ECG using sine waves and spikes
    sim = np.zeros((len(t), 12), dtype=np.float32)
    for ch in range(12):
        # Basic sine wave and Gaussian noise
        sim[:, ch] = (0.6 * np.sin(2 * np.pi * 1.0 * t + 0.1 * ch) + 
                      0.05 * np.random.randn(len(t)))
        
        # Simulate R-peaks (spikes) every 1.0 seconds
        spike_times = np.arange(0, 300, 1.0) + 0.02 * ch
        for st in spike_times:
            idx = int(st*fs)
            if 0 <= idx < len(t):
                # Add a sharp, bipolar complex to simulate QRS
                sim[idx:idx+3, ch] += np.array([0.8, 1.2, 0.6]) # Simple R-peak sim
    
    _stream["signals"] = sim
    _stream["channels"] = [f"Lead{ch+1}" for ch in range(12)]
    _stream["loaded"] = True # Mark as loaded even if simulated

# -------------------------
# Prediction Logic
# -------------------------
def _prepare_signal_for_model(sig, min_len=1000, target_len=None):
    """Pads or truncates signal to a suitable length for the model input."""
    sig = np.asarray(sig, dtype=np.float32).flatten()
    target_len = target_len or max(min_len, len(sig))
    
    # Pad if shorter than target length
    if len(sig) < target_len:
        sig = np.pad(sig, (0, target_len-len(sig)))
    
    # Truncate if longer than target length
    return sig[:target_len]

def predict_cycle(sig):
    """Runs a single ECG segment through the HuBERT model and classifier head."""
    # Ensure signal is at least 1000 samples long
    arr = _prepare_signal_for_model(sig, min_len=1000)
    # Convert to PyTorch tensor (batch dimension: [1, seq_len])
    x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = MODEL(x)
        
        # Extract features from the model output (HuBERT typically uses last_hidden_state)
        if isinstance(out, dict) and "last_hidden_state" in out:
            # Use mean of sequence features (common approach for classification)
            feats = out["last_hidden_state"].mean(dim=1)
        elif isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor):
            feats = out[0].mean(dim=1)
        elif isinstance(out, torch.Tensor) and out.dim() >= 2:
            feats = out.mean(dim=1)
        else:
            raise RuntimeError("Unsupported model output type.")
        
        logits = classifier_head(feats)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get the highest probability class
        idx = int(np.argmax(probs))
        
        return f"{DISEASE_CLASSES[idx]} ({probs[idx]:.2f})"

# -------------------------
# Routes
# -------------------------
@ECG_BP.route("/")
def index():
    """Renders the HTML template for the frontend visualization."""
    return render_template("ecg.html")

@ECG_BP.route("/config")
def config_route():
    """Returns critical configuration info for the frontend."""
    return jsonify({
        "num_channels": _stream["signals"].shape[1],
        "fs": int(_stream["fs"]),
        "display_fs": int(DISPLAY_FS), # Correctly exposing the display frequency
        "channels": _stream["channels"]
    })

@ECG_BP.route("/update", methods=["POST"])
def update():
    """Streams the next chunk of ECG data and returns a prediction."""
    req = request.json or {}
    # Channels to send to the frontend for plotting
    channels = req.get("channels", [])
    
    chunk_duration = STREAMING_CHUNK_DURATION
    fs = int(_stream["fs"])
    # Calculate the number of high-resolution samples to read
    N_chunk_highres = int(chunk_duration * fs) 

    total_len = _stream["signals"].shape[0]
    start, end = _stream["pos"], _stream["pos"] + N_chunk_highres
    
    # Handle wrap-around (simulating continuous stream from a finite file)
    if end <= total_len:
        seg_block = _stream["signals"][start:end, :]
        _stream["pos"] = end
    else:
        # Wrap around: part1 is from the end, part2 is from the beginning
        part1 = _stream["signals"][start:]
        part2 = _stream["signals"][:end-total_len]
        seg_block = np.concatenate([part1, part2], axis=0)
        _stream["pos"] = end % total_len

    # Update global index for frontend time axis calculation
    global_start_idx = _stream["global_idx"]
    global_end_idx = global_start_idx + seg_block.shape[0]
    _stream["global_idx"] = global_end_idx

    # Generate high-resolution time array
    times_highres = np.arange(global_start_idx, global_end_idx) / float(fs)
    
    # Calculate downsampling step (e.g., if FS=360 and DISPLAY_FS=200, step is 2)
    step = max(1, int(round(fs / DISPLAY_FS)))
    
    signals_dict, time_ds = {}, None
    
    # Downsample the data before sending to frontend
    for ch_idx in channels:
        # Check if the requested channel index is valid
        s = seg_block[:, ch_idx] if 0 <= ch_idx < seg_block.shape[1] else seg_block.mean(axis=1)
        
        # Apply downsampling
        s_ds = s[::step]
        t_ds = times_highres[::step]
        
        signals_dict[str(ch_idx)] = s_ds.tolist()
        if time_ds is None: time_ds = t_ds.tolist() # Only need one time array

    prediction = "No channel selected"
    if channels:
        # Use the first selected channel's full chunk for prediction
        arr = seg_block[:, channels[0]]
        try: 
            prediction = predict_cycle(arr)
        except Exception as e: 
            prediction = f"Prediction Error: {e}"

    return jsonify({"time": time_ds or [], "signals": signals_dict, "prediction": prediction})

# -------------------------
# Training Route
# -------------------------
@ECG_BP.route("/train", methods=["POST"])
def train():
    """
    Fine-tunes the classifier head using a small subset of the PTB-XL database.
    Requires pandas and wfdb to be installed and PTB-XL data available locally.
    """
    if pd is None or wfdb is None:
        return jsonify({"status": "error", "message": "Training requires pandas and wfdb libraries to be installed."}), 400

    data = request.json or {}
    ptbxl_base = data.get("ptbxl_base", "data/ptbxl")
    max_records = int(data.get("max_records", 200))
    epochs = int(data.get("epochs", 3))
    batch_size = int(data.get("batch_size", 8))
    seq_len = int(data.get("seq_len", 5000)) # Input sequence length for the model

    # FIX: detect ptbxl_database.csv correctly
    csv_candidates = []
    for root, dirs, files in os.walk(ptbxl_base):
        for f in files:
            if f.lower().endswith(".csv") and ("ptbxl" in f.lower() or f.lower() == "ptbxl_database.csv"):
                csv_candidates.append(os.path.join(root, f))
    if not csv_candidates:
        return jsonify({"status": "error", "message": f"No PTB-XL CSV found under '{ptbxl_base}'. Please place ptbxl_database.csv there."}), 400

    csv_path = csv_candidates[0]
    df = pd.read_csv(csv_path, index_col=0)

    # Detect the filename column
    fn_col = next((c for c in ["filename_lr","filename","fname","record_name"] if c in df.columns), None)
    if fn_col is None:
        return jsonify({"status": "error", "message": "Filename column not found in CSV."}), 400

    signals_list, labels_list = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= max_records: break
        
        # Construct the full record path
        fname = row[fn_col]
        rec_base = os.path.join(ptbxl_base, fname) if not os.path.isabs(fname) else fname
        
        try:
            # Load signal
            rec = wfdb.rdrecord(rec_base)
            sig = rec.p_signal.astype(np.float32)
            # Use Lead II (index 1) or default to the first lead
            lead_idx = rec.sig_name.index("II") if "II" in rec.sig_name else 0
            sig1 = sig[:, lead_idx]
            signals_list.append(sig1)
            
            # Extract Label (simplistic mapping for demonstration)
            lab = "Other"
            if "diagnostic_superclass" in df.columns and not pd.isna(row.get("diagnostic_superclass")):
                lab = str(row["diagnostic_superclass"])
            elif "scp_codes" in row:
                try:
                    scp = row["scp_codes"]
                    scp_d = eval(scp) if isinstance(scp, str) else scp
                    if isinstance(scp_d, dict) and scp_d: lab = list(scp_d.keys())[0]
                except: lab = "Other"
                
            if isinstance(lab,str):
                if "NORM" in lab.upper(): lab="Normal"
                elif "MI" in lab.upper(): lab="MI"
                elif "HYP" in lab.upper(): lab="Hypertrophy"
                elif "ST" in lab.upper(): lab="STTC"
                elif "CD" in lab.upper(): lab="Conduction"
                else: lab="Other"
            labels_list.append(lab)
        except Exception as e: 
            print(f"Skipping record {rec_base}: {e}")
            continue

    if len(signals_list) < 10:
        return jsonify({"status": "error", "message": f"Not enough records ({len(signals_list)} found) to train."}), 400

    # Prepare Tensors
    X, Y = [], []
    for sig, lab in zip(signals_list, labels_list):
        # Pad/truncate signal to sequence length
        arr = sig[:seq_len] if len(sig) >= seq_len else np.pad(sig, (0, seq_len-len(sig)))
        X.append(arr)
        
        # Convert label to index
        label_idx = DISEASE_CLASSES.index(lab) if lab in DISEASE_CLASSES else DISEASE_CLASSES.index("Other")
        Y.append(label_idx)

    X = np.stack(X)
    Y = np.array(Y, dtype=np.int64)
    X_t, Y_t = torch.tensor(X, dtype=torch.float32).to(DEVICE), torch.tensor(Y, dtype=torch.long).to(DEVICE)

    # Training setup
    # Freeze the main HuBERT model layers
    for p in MODEL.parameters(): p.requires_grad=False
    classifier_head.train()
    optimizer = optim.Adam(classifier_head.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    n = len(X_t)
    for ep in range(epochs):
        perm = torch.randperm(n) # Shuffle data indices
        running_loss = 0
        for i in range(0,n,batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_t[idx], Y_t[idx]
            
            # Forward pass through HuBERT (frozen)
            with torch.no_grad():
                out = MODEL(xb)
                # Extract features
                feats = out.last_hidden_state.mean(dim=1) if hasattr(out,"last_hidden_state") else out[0].mean(dim=1)
                
            # Forward pass through the classifier head (trainable)
            logits = classifier_head(feats)
            loss = loss_fn(logits, yb)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += float(loss.item())*xb.size(0)
        
        print(f"[train] epoch {ep+1}/{epochs} avg_loss={running_loss/n:.4f}")

    # Save model head after training
    classifier_head.eval()
    save_path = os.environ.get("HUBERT_FINETUNED","hubert_ecg/hubert_ecg_finetuned_head.pth")
    # Save the fine-tuned classifier head state dict
    torch.save({"classifier_state_dict": classifier_head.state_dict(),"classes":DISEASE_CLASSES}, save_path)
    
    return jsonify({"status":"trained","model_saved_to":save_path, "total_records": n})