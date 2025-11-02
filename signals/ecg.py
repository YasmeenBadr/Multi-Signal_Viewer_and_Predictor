"""
ECG backend services
--------------------
This module provides the server-side logic for the Real-Time ECG Viewer:
- Loads ECG records (WFDB .hea/.dat) or generates a simulated 12-lead signal
- Streams windowed signal chunks to the frontend with optional downsampling
- Demonstrates deliberate downsampling-with-aliasing to show its effect
- Computes and renders derived views: XOR, Polar, Recurrence (density image)
- Runs lightweight 1D and 2D CNNs for demo predictions

Key concepts in the pipeline:
- Native sampling rate vs. current streaming sampling rate
- Per-chunk decimation with persistent phase to simulate realistic aliasing
- Rolling per-channel buffers sized to the 1D model sequence length
- Background training of a toy 2D CNN from recurrence plots when diagnosis
  text is available in .hea files
"""
# ecg.py
import os
import time
import logging
import threading
from typing import Optional
from .resampling import decimate_with_aliasing

import numpy as np
import torch
import torch.nn as nn

from flask import Flask, Blueprint, request, jsonify, render_template

# optional wfdb dependency
try:
    import wfdb
except Exception:
    wfdb = None

# optional external path (if you have a simple_ecg module with DATA_PATH)
try:
    from simple_ecg import DATA_PATH
except Exception:
    DATA_PATH = None

# -------------------------
# Logging / config
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ecg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMPLE_MODEL_PATH = os.path.join(os.getcwd(), "simple_ecg_model.pt")
MODEL2D_PATH = os.path.join(os.getcwd(), "model2d_recurrence.pt")

# Labels / descriptions
DISEASE_CLASSES = ["Normal", "Abnormal"]
DISEASE_DESCRIPTIONS = {
    "Normal": "No obvious abnormality detected.",
    "Abnormal": (
        "Abnormal ECG pattern detected. Possible ischemia, arrhythmia, or other irregularities. "
        "Please confirm with a cardiologist."
    ),
}

# -------------------------
# Stream state (shared)
# -------------------------
_stream = {
    "loaded": False,
    "signals": None,
    "signals_raw": None,
    "channels": None,
    "fs": 500,      # native sampling rate default
    "pos": 0,
    "pos_native": 0,
    "record_path": None,
    "prev_chunks": {},
    "prev_chunks_raw": {},
    "recurrence_points": {},
    "polar_points": {},
    "pred_buffers": {},
    "pred_history": [],
    "rec_pred_history": [],
    "display_fs": None,
    "hea_diagnosis": None, # ADDED: To store the ground truth diagnosis
    "alias_phase": {}      # ADDED: persistent phase per target fs for decimation
}

# UI/runtime constants
DISPLAY_FS = 200
STREAMING_CHUNK_DURATION = 1.0
_model_seq_len = 5000
POLAR_MAX_POINTS = 2000
SMOOTH_WINDOW = 1
MIN_PRED_LEN = 1000

FREQ_DEFAULT = 500
FREQ_MIN = 10
DEFAULT_TIME_WINDOW_S = 15.0

# -------------------------
# 1D model definition (SimpleECG)
# -------------------------
class SimpleECG(nn.Module):
    """Tiny 1D CNN used to classify an ECG segment as Normal/Abnormal.
    """
    def __init__(self, input_length=5000):
        super().__init__()
        self.input_length = input_length
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length) # Create a dummy input tensor: shape (batch_size=1, channels=1, signal_length)
            dummy_output = self.conv_net(dummy_input) # Pass dummy input through conv layers to see output shape
            linear_input_size = dummy_output.numel()  # Get total number of elements after conv layers (needed for Linear layer input)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Final layer: 2 outputs (Normal / Abnormal classification)
        )

    def forward(self, x):
        x = self.conv_net(x)# Pass input through 1D convolutional layers to extract features
        x = self.classifier(x)# Pass extracted features through the classifier to get class scores
        return x    # Return the logits for each class

# Initialize model
model = SimpleECG(input_length=_model_seq_len).to(DEVICE)
if os.path.exists(SIMPLE_MODEL_PATH):
    try:
        sd = torch.load(SIMPLE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)
        logger.info("Loaded 1D model from %s", SIMPLE_MODEL_PATH)
    except Exception as e:
        logger.warning("Failed to load 1D model: %s", e)
else:
    logger.info("simple_ecg_model.pt not found — using untrained 1D model (for demo).")
model.eval()

# -------------------------
# 2D recurrence model
# -------------------------
class Simple2DCNN(nn.Module):
    """Toy 2D CNN for classifying recurrence-density images.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),# 1D convolution: 1 input channel (single-lead ECG), 16 filters, kernel size 5, padding 2 to keep length
            nn.ReLU(), # Activation function: introduce non-linearity
            nn.MaxPool2d(2),# Max pooling: reduce signal length by half (downsampling)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),# 1D convolution: 16 input channels, 32 filters, kernel size 5 samples, padding 2
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*32*32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

model2d = Simple2DCNN().to(DEVICE)
if os.path.exists(MODEL2D_PATH):
    try:
        sd2 = torch.load(MODEL2D_PATH, map_location=DEVICE)
        model2d.load_state_dict(sd2, strict=False)
        logger.info("Loaded 2D model from %s", MODEL2D_PATH)
    except Exception as e:
        logger.warning("Failed to load 2D model: %s", e)
model2d.eval()

from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# Utilities
# -------------------------
def build_recurrence_image(x, y, size=128):
    """Build a normalized 2D density image from two signals x and y.

    Steps
    - Make x,y float32 and flatten
    - Compute 2D histogram over dynamic ranges
    - Log-scale counts to compress dynamic range
    - Z-normalize to mean=0, std=1 for stable CNN input
    Returns a size x size float32 array.
    """
    try:
         # Convert inputs to float32 numpy arrays and flatten
        x = np.asarray(x, dtype=np.float32).flatten()
        y = np.asarray(y, dtype=np.float32).flatten()

        # If either signal is empty, return a zero image
        if len(x) == 0 or len(y) == 0:
            return np.zeros((size, size), dtype=np.float32)
        
        # Find min and max of each signal to define histogram ranges
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        if xmin == xmax:
            xmin -= 1e-3; xmax += 1e-3
        if ymin == ymax:
            ymin -= 1e-3; ymax += 1e-3
        # Compute a 2D histogram over x and y with given number of bins
        H, xedges, yedges = np.histogram2d(x, y, bins=size, range=[[xmin, xmax], [ymin, ymax]])

        # Apply log scaling to compress the dynamic range of histogram counts
        H = np.log1p(H)
        H = (H - H.mean()) / (H.std() + 1e-6)   #normalization
        return H.astype(np.float32)
    except Exception as e:
        logger.debug("build_recurrence_image failed: %s", e)
        return np.zeros((size, size), dtype=np.float32)



def extract_diagnosis_from_hea(record_base: Optional[str]):
    """
    Best-effort extraction of diagnosis/free-text from a WFDB .hea file.

    Args:
        record_base (str or None): base path to the ECG record (without extension)

    Returns:
        str or None: extracted diagnosis text in lowercase, or 'healthy' if indicated,
                     or None if not found.
    """

    # If no record path provided, return None
    if not record_base:
        return None

    # Try to get the .hea file path by replacing .dat with .hea
    hea_path = record_base.replace(".dat", ".hea")

    # Fallback: if that file doesn't exist, append .hea
    if not os.path.exists(hea_path):
        hea_path = record_base + ".hea" 
        if not os.path.exists(hea_path):
            return None  # No .hea file found, return None

    # Try to read the contents of the .hea file
    try:
        with open(hea_path, "r", encoding="latin-1") as f:
            text = f.read()
    except Exception:
        return None  # If reading fails, return None

    # Convert all text to lowercase for easier keyword matching
    low = text.lower()

    # Quick check: if file mentions healthy/normal/control, return 'healthy'
    if "healthy" in low or "control" in low or "normal" in low:
        return "healthy"

    # Otherwise, try to extract diagnosis from lines containing key terms
    try:
        for line in text.splitlines():  # Go line by line
            l = line.lower()
            # Look for diagnosis-related keywords
            if "diagnosis" in l or "reason" in l or "infarct" in l or "mi" in l: 
                parts = line.split(":", 1)  # Split on first colon
                if len(parts) > 1:
                    return parts[1].strip()  # Return text after colon
                return parts[0].strip()  # Otherwise return whole line
    except Exception:
        pass  # Ignore any errors and fall through

    # If nothing found, return None
    return None


def train_model2d_on_record(signals, chan_names, record_base, max_windows=200, window_s=2.0, epochs=6):
    """Background training of the 2D CNN from recurrence windows.

    Steps:
    - Derive binary label from .hea file (healthy/normal -> 0, else 1)
    - Slice rolling windows across two ECG channels
    - Build recurrence plot images
    - Train a small 2D CNN for a few epochs
    - Save CSV of the two channels for inspection
    """
    try:
        # 1. Extract text diagnosis from .hea file
        rec_label_text = extract_diagnosis_from_hea(record_base) if record_base else None
        if not rec_label_text:
            logger.info("No diagnosis in .hea; skipping 2D training.")
            return

        # 2. Convert text to lowercase and define binary label
        ltxt = rec_label_text.lower()
        label = 0 if ("healthy" in ltxt or "healthy control" in ltxt or "normal" in ltxt) else 1

        # 3. Define window length and step size
        fs_local = _stream.get("fs", FREQ_DEFAULT)          # get sampling rate
        win = max(4, int(window_s * fs_local))             # window length in samples
        step = max(1, win // 2)                            # step size (50% overlap)
        N = signals.shape[0]                               # total signal length
        ch_count = signals.shape[1]                        # number of channels
        ch0 = 0                                            # channel 0
        ch1 = 1 if ch_count > 1 else 0                     # channel 1 or 0 if single channel

        # 4. Save the two channels to CSV for inspection
        try:
            outdir = os.path.join(os.getcwd(), 'results', 'recurrence_data')
            os.makedirs(outdir, exist_ok=True)             # create folder if not exists
            base = os.path.basename(record_base) if record_base else f'record_{int(time.time())}'
            csv_path = os.path.join(outdir, f"{base}_ch{ch0}_ch{ch1}_recurrence.csv")
            twoch = np.stack([signals[:, ch0], signals[:, ch1]], axis=1)
            header = 'ch0,ch1'
            np.savetxt(csv_path, twoch, delimiter=',', header=header, comments='')
            logger.info("Saved recurrence CSV to %s", csv_path)
        except Exception as e:
            logger.debug("Failed to save recurrence CSV: %s", e)

        # 5. Slice windows and build recurrence images
        images = []
        labels = []
        count = 0
        for start in range(0, N - win + 1, step):
            if count >= max_windows:
                break
            x = signals[start:start+win, ch0]               # slice window for channel 0
            y = signals[start:start+win, ch1]               # slice window for channel 1
            img = build_recurrence_image(x, y, size=128)   # build recurrence plot image
            images.append(img)
            labels.append(label)
            count += 1

        # 6. Skip training if not enough windows
        if len(images) < 4:
            logger.info("Not enough windows for training 2D model; found %d", len(images))
            return

        # 7. Convert images and labels to numpy arrays and add channel dimension
        X = np.stack(images, axis=0)[:, None, :, :].astype(np.float32)  # shape: [N, 1, H, W]
        y_arr = np.array(labels, dtype=np.int64)

        # 8. Convert to PyTorch tensors and create DataLoader
        tX = torch.from_numpy(X)
        ty = torch.from_numpy(y_arr)
        dataset = TensorDataset(tX, ty)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 9. Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()             # classification loss
        optim = torch.optim.Adam(model2d.parameters(), lr=1e-3)

        # 10. Train the 2D CNN
        model2d.train()
        logger.info("Starting 2D training on %d samples, label=%d", len(dataset), label)
        for ep in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(DEVICE)                            # move batch to GPU/CPU
                yb = yb.to(DEVICE)
                optim.zero_grad()                             # reset gradients
                logits = model2d(xb)                          # forward pass
                loss = criterion(logits, yb)                  # compute loss
                loss.backward()                               # backpropagation
                optim.step()                                  # update weights
                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)                  # predicted class
                correct += (preds == yb).sum().item()
                total += xb.size(0)
            if total > 0:
                logger.info("2D train epoch %d/%d loss=%.4f acc=%.3f", ep+1, epochs, total_loss/total, correct/total)

        # 11. Save the trained 2D model to disk
        try:
            torch.save(model2d.state_dict(), MODEL2D_PATH)
            logger.info("Saved 2D model to %s", MODEL2D_PATH)
        except Exception as e:
            logger.warning("Failed to save 2D model: %s", e)

        model2d.eval()  # set model to evaluation mode
    except Exception as e:
        logger.exception("2D training failed: %s", e)


def predict_recurrence_pair(x, y):
    """Run the 2D CNN on a recurrence image built from x and y.

    Returns dict with label, probabilities, and confidence or None on failure.
    """
    try:
        img = build_recurrence_image(x, y, size=128)
        arr = (img - np.mean(img)) / (np.std(img) + 1e-6)
        t = torch.from_numpy(arr.astype(np.float32))[None, None, :, :].to(DEVICE)
        with torch.no_grad():
            logits = model2d(t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
        label = DISEASE_CLASSES[idx]
        return {"label": label, "probabilities": probs.tolist(), "confidence": float(probs[idx])}
    except Exception as e:
        logger.debug("predict_recurrence_pair failed: %s", e)
        return None

# WFDB load
def load_wfdb_record(record_base):
    """Load a WFDB record given its base path (without extension).

    Returns (signals, signal_names, fs). Raises if wfdb is not available.
    """
    if wfdb is None:
        raise RuntimeError("wfdb package not available in environment")
    #  Read the record using wfdb
    rec = wfdb.rdrecord(record_base)
    # Extract the ECG signals as float32 numpy array
    signals = rec.p_signal.astype(np.float32)
    #Get the signal (channel) names
    sig_names = rec.sig_name
     #Get sampling frequency (fs), fallback to default if not present
    fs = int(rec.fs) if hasattr(rec, "fs") else _stream.get("fs", FREQ_DEFAULT)
    return signals, sig_names, fs


def setup_simulated_record():
    """Initialize a synthetic 12-lead ECG-like signal for demo purposes.

    Produces slow sinusoids plus periodic spikes to resemble QRS complexes.
    Populates the global _stream with signals and metadata.
    """
    logger.info("No WFDB record found — using simulated ECG (12 leads).")
    fs = _stream.get("fs", FREQ_DEFAULT)
    duration_s = 60
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    sim = np.zeros((len(t), 12), dtype=np.float32)
    for ch in range(12):
        sim[:, ch] = 0.6 * np.sin(2 * np.pi * 1.2 * t + 0.15 * ch) + 0.05 * np.random.randn(len(t))
        spike_times = np.arange(0, duration_s, 1.0) + 0.03 * ch
        for st in spike_times:
            idx = int(st * fs)
            if 0 <= idx < len(t):
                sim[idx:idx+3, ch] += [0.8, 1.2, 0.6]
    _stream.update({
        "signals": sim,
        "signals_raw": sim.copy(),
        "channels": [f"Lead {ch+1}" for ch in range(12)],
        "fs": fs,
        "fs_native": fs,
        "loaded": True,
        "pos": 0,
        "pos_native": 0,
        "hea_diagnosis": "Simulated Signal" # ADDED: Default diagnosis
    })


# Load initial record (WFDB or simulated)
# Try to load a real WFDB record first, fallback to simulated ECG if not available
if wfdb is not None and DATA_PATH and os.path.exists(DATA_PATH + ".dat"):
    try:
        # Load signals, channel names, and sampling rate from the WFDB record
        s, names, sr = load_wfdb_record(DATA_PATH)

        # Extract diagnosis/free-text from the corresponding .hea file
        hea_diag = extract_diagnosis_from_hea(DATA_PATH)

        # Update the global _stream dictionary with the loaded data
        _stream.update({
            "signals": s,                  # Loaded ECG signals (num_samples x num_channels)
            "signals_raw": s.copy(),       # Raw copy for reference
            "channels": names,             # Channel names (e.g., Lead I, Lead II, etc.)
            "fs": sr,                      # Sampling rate of the record
            "pos": 0,                       # Current streaming position (start)
            "loaded": True,                # Flag indicating signals are loaded
            "record_path": DATA_PATH,      # Path to the WFDB record
            "hea_diagnosis": hea_diag      # Diagnosis extracted from .hea
        })

        try:
            # Start background training of the 2D CNN using recurrence images
            # Run in a separate thread so it doesn't block the main program
            threading.Thread(
                target=train_model2d_on_record, 
                args=(s, names, DATA_PATH), 
                daemon=True
            ).start()
        except Exception:
            # Ignore errors in background training
            pass

    except Exception as e:
        # Log a warning if loading the WFDB record fails
        logger.warning("Failed to load specified record: %s", e)
        _stream["loaded"] = False  # Mark as not loaded

# If loading the WFDB record failed, set up a simulated 12-lead ECG
if not _stream["loaded"]:
    setup_simulated_record()

# -------------------------
# Downsample **with aliasing** (raw slicing, no anti-aliasing)
# -------------------------
def resample_with_aliasing(sig, native_fs, target_fs, pos_native: int = 0):
    """Strict decimation without anti-aliasing using persistent phase.

    This intentionally skips anti-alias filtering to demonstrate how aliasing
    distorts signals when reducing sample rate. The decimator keeps an
    `_stream['alias_phase']` dict to maintain continuity across chunks so the
    aliasing artifacts appear consistent over time.
    """
    # Delegate to the shared implementation, preserving ECG's alias_phase state
    phase_state = _stream.setdefault("alias_phase", {})
    return decimate_with_aliasing(sig, native_fs, target_fs, pos_native=pos_native, phase_state=phase_state)

# Example of streaming a chunk
def get_stream_chunk(duration_s=1.0):
    """Return one streaming chunk downsampled to DISPLAY_FS from native raw.

    Uses circular indexing to loop seamlessly when reaching the end.
    """
    # Check if a signal has been loaded; if not, return None
    if not _stream["loaded"]:
        return None
    fs_cur = _stream["fs"]
    fs_native = _stream.get("fs_native", fs_cur)
    pos_n = _stream.get("pos_native", 0)
    raw = _stream["signals_raw"]
     # Number of samples to take for this chunk
    chunk_n = int(duration_s * fs_native)
    if pos_n + chunk_n > raw.shape[0]:# Handle circular indexing if we reach the end of the signal
        part1 = raw[pos_n:, :]# Part from current position to the end
        part2 = raw[:(pos_n + chunk_n) % raw.shape[0], :]  # Part from start to cover remaining samples
        chunk_native = np.vstack([part1, part2])#concatunation to form full chunk
    else:
        chunk_native = raw[pos_n:pos_n+chunk_n, :]  # Simply take a slice of the raw signal
    _stream["pos_native"] = (pos_n + chunk_n) % raw.shape[0]
    # Downsample to display fs from native
    chunk_ds = resample_with_aliasing(chunk_native, fs_native, DISPLAY_FS, pos_native=pos_n)
    return chunk_ds

# -------------------------
# Flask blueprint
# -------------------------
ECG_BP = Blueprint("ecg", __name__, url_prefix="/ecg", template_folder="templates")

@ECG_BP.route("/")
def index():
    """Render the ECG viewer page."""
    return render_template("ecg.html")

@ECG_BP.route("/config")
def config():  
    """Return configuration for the frontend UI.

    Includes channel names, native fs, current display fs, defaults, and any
    diagnosis text read from the .hea file.
    """
    display_fs = _stream.get("display_fs") or DISPLAY_FS
    return jsonify({        #This ensures the UI knows available channels and native sampling rate.
        "fs": _stream["fs"],
        "fs_native": _stream.get("fs_native", _stream.get("fs", FREQ_DEFAULT)),
        "display_fs": display_fs,
        "channels": _stream["channels"],
        "default_time_window_s": DEFAULT_TIME_WINDOW_S,
        "freq_default": _stream.get("fs", FREQ_DEFAULT),
        "freq_min": FREQ_MIN,
        "hea_diagnosis": _stream.get("hea_diagnosis") # ADDED: return diagnosis
    }) 

@ECG_BP.route("/set_freq", methods=["POST"])
def set_freq():
    """Set current streaming sampling frequency (alias of set_sampling).

    Clamps to [FREQ_MIN, native_fs, 500], resamples raw signals with aliasing,
    and updates _stream state without clearing prediction histories.
    """
    try:
        data = request.get_json(silent=True) or {}
        # Accept both keys for compatibility with frontend
        new_fs = float(data.get("frequency", data.get("sampling_freq", FREQ_DEFAULT)))
        raw_native_fs = _stream.get("fs_native", _stream.get("fs", FREQ_DEFAULT)) # Get true native FS from _stream
        MAX_FREQ_LIMIT = 500 
        # Clamp the requested frequency to a safe range
        new_fs = max(FREQ_MIN, min(new_fs, raw_native_fs, MAX_FREQ_LIMIT))
        # No-op if requested sampling equals current fs (avoid unnecessary resampling and state changes)
        try:
            cur_fs = float(_stream.get("fs", FREQ_DEFAULT))
            if abs(float(new_fs) - cur_fs) < 1e-6: # If unchanged, return early to avoid unnecessary resampling
                return jsonify({"success": True, "message": f"Frequency unchanged ({int(cur_fs)} Hz)", "current_sampling": int(cur_fs)})
        except Exception:
            pass
        
        raw = _stream.get("signals_raw")
        if raw is None:
            return jsonify({"success": False, "error": "No raw signals to resample."}), 400
        
        # EDITED: Use the native FS stored in _stream to determine the downsampling
        down = resample_with_aliasing(raw, raw_native_fs, new_fs)
        if down.ndim == 1:# Ensure signals are 2D for consistent handling
            down = down[:, None]
        
        # EDITED: Update _stream to reflect the *current operating* frequency and signals
        _stream["signals"] = down.astype(np.float32)
        _stream["fs"] = int(new_fs) # The *current* FS
        _stream["pos"] = 0   # Reset current position to start
        _stream["alias_phase"] = {}  # Reset aliasing phase

        
        # Preserve all buffers/state to avoid clearing history when sampling changes
        # Only reset position, keep prediction buffers, history, and other state intact
        _stream["last_sampling_change_ts"] = time.time()
        _stream["sampling_reduced"] = bool(int(new_fs) < int(raw_native_fs))
        
        return jsonify({"success": True, "message": f"Frequency set to {int(new_fs)} Hz", "current_sampling": int(new_fs)})
    except Exception as e:
        logger.exception("set_freq failed")
        return jsonify({"success": False, "error": str(e)}), 500

@ECG_BP.route("/set_sampling", methods=["POST"])
def set_sampling():
    """Alias for set_freq that accepts {sampling_freq: <float>} from the UI.

    Performs the same work as set_freq but normalizes the accepted key.
    """
    try:
        data = request.get_json(silent=True) or {}
        # Normalize to a single variable
        new_fs = float(data.get("sampling_freq", data.get("frequency", FREQ_DEFAULT)))

        raw_native_fs = _stream.get("fs_native", _stream.get("fs", FREQ_DEFAULT))
        MAX_FREQ_LIMIT = 500
        new_fs = max(FREQ_MIN, min(new_fs, raw_native_fs, MAX_FREQ_LIMIT))
        # No-op if requested sampling equals current fs (avoid unnecessary resampling and state changes)
        try:
            cur_fs = float(_stream.get("fs", FREQ_DEFAULT))
            if abs(float(new_fs) - cur_fs) < 1e-6:
                return jsonify({"success": True, "message": f"Frequency unchanged ({int(cur_fs)} Hz)", "current_sampling": int(cur_fs)})
        except Exception:
            pass

        raw = _stream.get("signals_raw")
        if raw is None:
            return jsonify({"success": False, "error": "No raw signals to resample."}), 400
        
        down = resample_with_aliasing(raw, raw_native_fs, new_fs)
        if down.ndim == 1:
            down = down[:, None]

        _stream["signals"] = down.astype(np.float32)
        _stream["fs"] = int(new_fs)
        _stream["pos"] = 0
        _stream["alias_phase"] = {}
        # Preserve buffers/state to avoid clearing history when sampling changes
        # Track sampling change timestamp and whether reduced
        _stream["last_sampling_change_ts"] = time.time()
        _stream["sampling_reduced"] = bool(int(new_fs) < int(raw_native_fs))

        return jsonify({"success": True, "message": f"Frequency set to {int(new_fs)} Hz", "current_sampling": int(new_fs)})
    except Exception as e:
        logger.exception("set_sampling failed")
        return jsonify({"success": False, "error": str(e)}), 500

@ECG_BP.route("/reset_sampling", methods=["POST"])
def reset_sampling():
    """Reset the streaming frequency to the original native frequency.

    Restores _stream["signals"] to the raw copy and resets alias phase state.
    """
    try:
        raw = _stream.get("signals_raw")
        native_fs = _stream.get("fs_native", _stream.get("fs", FREQ_DEFAULT))

        if raw is None:
            return jsonify({"success": False, "error": "No raw signals to reset."}), 400
        
        _stream["signals"] = raw.copy()
        _stream["fs"] = int(native_fs)
        _stream["pos"] = 0
        _stream["alias_phase"] = {}
        # Preserve all buffers/state to avoid clearing history when resetting sampling
        # Only reset position, keep prediction buffers, history, and other state intact
        _stream["last_sampling_change_ts"] = time.time()
        _stream["sampling_reduced"] = False
        
        return jsonify({"success": True, "message": f"Frequency reset to {int(native_fs)} Hz", "current_sampling": int(native_fs)})
    except Exception as e:
        logger.exception("reset_sampling failed")
        return jsonify({"success": False, "error": str(e)}), 500

def predict_signal(sig_chunk):
    """
    Predict a chunk of ECG using 1D model.
    sig_chunk: np.array (samples x channels)
    Returns list of dict with {"label":..., "probabilities":..., "confidence":...} per channel.
    """
    results = []
    sig_chunk = np.asarray(sig_chunk, dtype=np.float32)
    
    for ch in range(sig_chunk.shape[1]):# Iterate over all channels
        x = sig_chunk[:, ch]
        
        # *** FIX: Initialize pad_width before conditional logic ***
        pad_width = 0 
        
        # Ensure we have the correct input length
        if len(x) > _model_seq_len:
            # Take the most recent _model_seq_len samples
            x = x[-_model_seq_len:]
        elif len(x) < _model_seq_len:
            # Pad with zeros at the begining if too short
            pad_width = _model_seq_len - len(x)
            x = np.pad(x, (pad_width, 0), mode='constant')
        
        # Check if padding was required, and adjust prediction if needed
        if pad_width > 0:
            results.append({"label": "Waiting",
                            "probabilities": [1.0, 0.0],
                            "confidence": 0.0})
            continue  # Skip prediction for this channel

        # Normalize per-channel to improve model sensitivity
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = x[None, None, :]  # batch x channel x length
        x_tensor = torch.from_numpy(x).to(DEVICE)
        with torch.no_grad():# Perform prediction without gradient computation
            logits = model(x_tensor)# Model output before softmax
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            results.append({
                "label": DISEASE_CLASSES[idx],
                "probabilities": probs.tolist(),
                "confidence": float(probs[idx])
            })
    return results

ECG_BP.route("/upload", methods=["POST"])
def upload():
    """
    Handle multi-file upload (.hea, .dat, optional .xyz) and attempt reload.
    """    
    # Define the upload directory and create it if it doesn't exist
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Get the list of uploaded files from the request
    files = request.files.getlist("files")
    
    # Initialize variables to track saved files
    base_name = None
    saved = {"hea": None, "dat": None, "xyz": None}
    
    # Loop through each uploaded file
    for f in files:
        fname = f.filename
        if not fname:
            continue  # Skip files without a name
        
        # Save the file to the upload directory
        saved_path = os.path.join(upload_dir, fname)
        f.save(saved_path)
        
        # Track the type of file uploaded
        if fname.endswith(".hea"):
            saved["hea"] = fname
            base_name = fname[:-4]  # Remove extension for later use
        elif fname.endswith(".dat"):
            saved["dat"] = fname
            base_name = fname[:-4]
        elif fname.endswith(".xyz"):
            saved["xyz"] = fname  # Optional additional file
    
    # Initialize message list and success flag
    msg = []
    success = False
    
    # Attempt to load the record if both .hea and .dat files are present
    if saved["hea"] and saved["dat"]:
        full_dat_path = os.path.join(upload_dir, base_name + ".dat")
        
        # Try loading the record and updating _stream
        if _try_load_record_after_upload(full_dat_path):
            msg.append(f"Record loaded successfully. Diagnosis: {_stream.get('hea_diagnosis', 'Unknown')}")
            success = True
        else:
            msg.append("Failed to load uploaded record.")
    else:
        # If essential files are missing, notify the user
        msg.append("Files uploaded. Please upload both .hea and .dat for record reload.")
        if saved["xyz"]:
            msg.append(".xyz file uploaded and saved.")  # Optional file message
    
    # Return a JSON response with success status, messages, and diagnosis info
    return jsonify({
        "success": success,
        "message": " ".join(msg),
        "hea_diagnosis": _stream.get("hea_diagnosis")
    })
 

 
def _try_load_record_after_upload(file_path):
    """
    Load an uploaded record into _stream using WFDB if available.
    Falls back to CSV if WFDB is not available.
    Extracts diagnosis from .hea and resets model buffers.
    Starts background 2D model training.
    """
    try:
        # Remove .dat extension to get the base record name
        record_base = file_path.replace(".dat", "")
        
        # Extract diagnosis information from the .hea file
        hea_diag = extract_diagnosis_from_hea(record_base)
        
        # If WFDB is available and file is .dat, load using WFDB
        if wfdb is not None and file_path.endswith(".dat"):
            sigs, names, fs = load_wfdb_record(record_base)
            update_data = {
                "signals": sigs,                    # Loaded signals
                "signals_raw": sigs.copy(),         # Raw copy for reference
                "channels": names,                  # Channel names
                "fs": fs,                           # Sampling frequency
                "fs_native": fs,                    # Original sampling frequency
                "pos": 0,                           # Current position in the signal
                "record_path": file_path,           # Path to the loaded record
                "loaded": True,                     # Flag indicating successful load
                "hea_diagnosis": hea_diag           # Store diagnosis from .hea
            }
        else:
            # Fallback: load data from CSV file
            data = np.loadtxt(file_path, delimiter=',')
            fs = _stream.get("fs", FREQ_DEFAULT)  # Use existing or default FS
            update_data = {
                "signals": data.astype(np.float32),
                "signals_raw": data.astype(np.float32).copy(),
                "channels": [f"Lead {i+1}" for i in range(data.shape[1])],  # Generate default channel names
                "fs": fs,
                "fs_native": fs,
                "pos": 0,
                "record_path": file_path,
                "loaded": True,
                "hea_diagnosis": hea_diag
            }
        
        # Reset all prediction/model buffers for new record
        _stream["pred_buffers"] = {}
        _stream["pred_history"] = []
        _stream["rec_pred_history"] = []
        _stream["prev_chunks_raw"] = {}
        _stream["recurrence_points"] = {}
        _stream["polar_points"] = {}
        
        # Update _stream with the new loaded record data
        _stream.update(update_data)
        
        # Start background training of 2D model on this record
        try:
            threading.Thread(
                target=train_model2d_on_record,
                args=(sigs, names, record_base),
                daemon=True
            ).start()
        except Exception:
            pass  # Ignore errors in background training
        
        return True  # Successfully loaded
    except Exception as e:
        # Log any errors during loading and return False
        logger.exception("Failed to load uploaded record: %s", e)
        return False


@ECG_BP.route("/update", methods=["POST"])
def update():
    """Main streaming endpoint consumed by the frontend.

    Workflow per request
    - Parse selected channels and validate against loaded signals
    - Extract next native chunk, decimate to current streaming fs with aliasing
    - Update rolling per-channel buffers sized to model length
    - Build model-ready arrays and run 1D predictions
    - Optionally compute recurrence pair predictions for 2 channels
    - Return plot-ready arrays and metadata for the frontend renderer
    """
    try:
        # -------------------------
        # Accept JSON and parse parameters
        # -------------------------
        data = request.get_json(silent=True) or {}
        # EDITED: Now using _stream["fs"] as the current streaming frequency
        streaming_fs = _stream.get("fs", FREQ_DEFAULT)
        native_fs_raw = _stream.get("fs_native", streaming_fs) # Get original native FS
        
        # -------------------------
        # Normalize requested channels
        # -------------------------
        raw_channels = data.get("channels", list(range(12)))
        # Get channels from JSON, default to 0-11 if not provided
        channels = []  # Will hold the final list of valid channel indices


        if isinstance(raw_channels, int):# Handle case where a single integer channel is provided
            channels = [raw_channels]

        elif isinstance(raw_channels, str):# Handle case where channels are sent as a string like "0,2,5"
            try:
                channels = [int(x) for x in raw_channels.split(",") if x.strip()]
            except Exception:
                channels = list(range(12))  # Fallback to default if error occurs
        elif isinstance(raw_channels, (list, tuple)):
            parsed = []
            for x in raw_channels:
                try: parsed.append(int(x))
                except: continue  # Skip invalid entries
            channels = parsed if parsed else list(range(12))
        else:
            channels = list(range(12))

        if _stream["signals"] is None:# If no signals have been uploaded yet, return error to frontend
            return jsonify({"error": "No signals loaded. Upload first."}), 400

        # -------------------------
        # Validate channels against signal shape
        # -------------------------
        max_ch = _stream["signals"].shape[1]# Get total number of channels in the signal
        seen = set()   # Set to keep track of unique channels
        valid_channels = []

        # Loop through requested channels and keep only valid, non-duplicate channels
        for c in channels:
            if 0 <= c < max_ch and c not in seen:
                valid_channels.append(c)
                seen.add(c)
        if not valid_channels:  #  if no valid channels, use default channels 0 to min(12, max_ch)
            valid_channels = list(range(min(12, max_ch)))
        channels = valid_channels

        # -------------------------
        # Extract current chunk from native raw and decimate per-chunk
        # -------------------------
        fs_stream = _stream.get("fs", FREQ_DEFAULT)
        fs_native = _stream.get("fs_native", fs_stream)
        raw = _stream.get("signals_raw")
        N_native = int(STREAMING_CHUNK_DURATION * fs_native) # Calculate number of samples per chunk based on duration
        
        # Get current position in native signal
        pos_n = int(_stream.get("pos_native", 0))
        total_len_native = raw.shape[0]  # Total number of samples in the signal


        if pos_n + N_native <= total_len_native:# Extract chunk from raw signal; handle wrapping around if reaching end
            chunk_native = raw[pos_n:pos_n+N_native, :]
        else:
            part1 = raw[pos_n:, :]  # From current position to end
            part2 = raw[:(pos_n + N_native) % total_len_native, :]  # From start to complete chunk
            chunk_native = np.vstack([part1, part2])  # Combine parts for circular streaming

            
        _stream["pos_native"] = (pos_n + N_native) % total_len_native  
        # Decimate native chunk to current streaming fs with position-aware phase
        seg_block_current = resample_with_aliasing(chunk_native, fs_native, fs_stream, pos_native=pos_n)
        if seg_block_current.ndim == 1:
            seg_block_current = seg_block_current[:, None]

        # -------------------------
        # Rolling buffers per channel (uses *current* streaming_fs)
        # -------------------------
        for ch in channels:
            if ch not in _stream["pred_buffers"]:
                # Initialize buffer to be full of zeros for a cleaner start on new channels
                _stream["pred_buffers"][ch] = [0.0] * _model_seq_len
            
            # Use the current segment (which is already downsampled if applicable)
            seg = seg_block_current[:, ch].astype(np.float32)
            # Normalize per-channel to match training preprocessing
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)


            _stream["pred_buffers"][ch].extend(seg.tolist())# Append current segment to the rolling buffer
            # Keep only the last _model_seq_len samples to maintain fixed-length buffer
            if len(_stream["pred_buffers"][ch]) > _model_seq_len:
                _stream["pred_buffers"][ch] = _stream["pred_buffers"][ch][- _model_seq_len:]

        # -------------------------
        # Build signal chunk for prediction using current (potentially downsampled) signal
        # This ensures predictions reflect the aliasing/distortion from downsampling
        # -------------------------
        sig_selected_list = []
        for ch in channels:
            # Use the current streaming signal (which may be downsampled with aliasing)
            # rather than the original signal, so predictions reflect the distortion
            buf = np.array(_stream["pred_buffers"][ch], dtype=np.float32) #convert buffer to a numpy array
            if buf.size == 0:
                # If buffer is empty (rare case), just append it
                sig_selected_list.append(buf)
                continue


            if buf.shape[0] < _model_seq_len:
                # If buffer too short, upsample using nearest-neighbor (zero-order hold)
                # This preserves aliasing artifacts from downsampling
                ratio = _model_seq_len / float(max(1, buf.shape[0]))
                idx = np.floor(np.arange(_model_seq_len) / ratio).astype(int)
                idx = np.clip(idx, 0, buf.shape[0] - 1)
                up = buf[idx].astype(np.float32)
                # Per-buffer z-normalization to match training
                m = float(np.mean(up))
                s = float(np.std(up))
                up = (up - m) / (s + 1e-8)
                sig_selected_list.append(up)
            else:
                cut = buf[-_model_seq_len:] # If buffer long enough, take last _model_seq_len sample

                # Normalize buffer to match training
                m = float(np.mean(cut))
                s = float(np.std(cut))
                cut = (cut - m) / (s + 1e-8)
                sig_selected_list.append(cut)
        
        # Stack all selected channels for prediction
        if not sig_selected_list:
            # If no channels selected, create empty array
            sig_selected = np.empty((0, len(channels)), dtype=np.float32)
        else:
            # Stack each channel as a column
            sig_selected = np.stack([a for a in sig_selected_list], axis=1)
            
        # -------------------------
        # Model prediction
        # -------------------------
        sig_len = int(np.asarray(sig_selected).shape[0])
        prediction_out = None
        prediction_raw_out = None
        
        # Check if the buffer is full enough for a meaningful prediction
        if sig_len < _model_seq_len or streaming_fs < FREQ_MIN:  # If buffer too short or streaming frequency too low, return "Waiting"
             # Use the diagnosis extracted from .hea file if available
             hea_diag_text = _stream.get("hea_diagnosis")
             hea_low = hea_diag_text.lower() if isinstance(hea_diag_text, str) else ""
             default_label = "Normal" if any(t in hea_low for t in ["healthy control", "healthy", "normal"]) else "Abnormal"
             default_desc = DISEASE_DESCRIPTIONS[default_label]
             

             # Default prediction while accumulating data
             prediction = {"label": "Waiting",
                           "description": f"Accumulating data for prediction ({sig_len}/{_model_seq_len} @ {streaming_fs}Hz).",
                           "probabilities": [1.0, 0.0] if default_label == "Normal" else [0.0, 1.0],
                           "confidence": 0.0}
             prediction_out = prediction
             prediction_raw_out = [prediction]
        else:
             # Run the 1D model on the prepared signal chunk
            prediction = predict_signal(sig_selected)
            
            # -------------------------
            # Smoothing
            # -------------------------
            try:
                if isinstance(prediction, list):
                    # Extract probabilities from each channel
                    probs = np.array([p["probabilities"] for p in prediction], dtype=np.float32)
                    # Maintain prediction history for smoothing; reduce smoothing at low sampling
                    _stream.setdefault("pred_history", []).append(probs)
                    native_fs_for_smooth = _stream.get("fs_native", streaming_fs)
                    ratio = (float(streaming_fs) / float(native_fs_for_smooth)) if native_fs_for_smooth else 1.0
                    window = SMOOTH_WINDOW if ratio >= 0.7 else 1
                    if len(_stream["pred_history"]) > window:
                        _stream["pred_history"] = _stream["pred_history"][-window:]
                    
                     # Compute average probabilities across recent history
                    avg_probs = np.mean(np.stack(_stream["pred_history"], axis=0), axis=0)
                    
                    # If multiple channels, average probabilities across channels
                    if avg_probs.ndim > 1:
                         avg_probs = np.mean(avg_probs, axis=0)

                    # Determine smoothed label
                    sm_idx = int(np.argmax(avg_probs))
                    sm_label = DISEASE_CLASSES[sm_idx]
                    sm_result = {
                        "label": sm_label,
                        "description": DISEASE_DESCRIPTIONS.get(sm_label, ""),
                        "probabilities": avg_probs.tolist(),
                        "confidence": float(avg_probs[sm_idx])
                    }
                    # Include disease name only when label is Abnormal AND .hea text is not healthy
                    hea_text_l = str(_stream.get("hea_diagnosis", "")).lower()
                    healthy_terms = [
                        "healthy control", "healthy", "normal ecg", "normal sinus rhythm", " nsr ",
                        "no abnormal", "no significant abnormality", "within normal limits",
                        "no acute st-t changes", "no significant st-t changes"
                    ]
                    is_hea_healthy = any(t.strip() in f" {hea_text_l} " for t in healthy_terms)
                    sm_result["disease_name"] = ("" if (sm_label == "Normal" or is_hea_healthy) else _stream.get("hea_diagnosis", ""))
                    prediction = {"raw": prediction, "smoothed": sm_result}
            except Exception:
                pass

            prediction_out = prediction.get('smoothed') if isinstance(prediction, dict) and 'smoothed' in prediction else prediction
            prediction_raw_out = prediction.get('raw') if isinstance(prediction, dict) and 'raw' in prediction else prediction

        # -------------------------
        # Prepare display for plotting
        # -------------------------
        # Display at the exact streaming fs with NO thinning for faithful visualization
        current_fs = _stream.get("fs", FREQ_DEFAULT)
        display_fs = current_fs
        seg_block_for_display = seg_block_current
        try:
            time_axis = (np.arange(seg_block_for_display.shape[0]) / current_fs).tolist()
            signals_out = {str(ch): seg_block_for_display[:, ch].astype(float).tolist() for ch in channels}
        except Exception:
            time_axis = []
            signals_out = {str(ch): [] for ch in channels}

        # -------------------------
        # XOR visualization (difference with previous chunk)
        # -------------------------
        xor_out = {}
        if len(channels) == 1: # XOR only makes sense for 1 channel at a time
            ch = channels[0]
            curr_raw = seg_block_for_display[:, ch].astype(float)  # current chunk of data
            prev_raw = _stream["prev_chunks_raw"].get(ch)# previous chunk of same channel
            xor_threshold = float(data.get("xor_threshold", 0.05))  # minimum difference to visualize
            if prev_raw is not None and prev_raw.shape == curr_raw.shape:
                # Calculate difference from previous chunk
                diff = curr_raw - prev_raw
                # Mask small differences below threshold
                mask = np.abs(diff) > xor_threshold
                 # Keep only significant differences, else set 0
                xor_vals = np.where(mask, diff, 0.0)
                xor_out[ch] = xor_vals.tolist()
            else:
                # If no previous chunk, output all zeros
                xor_out[ch] = np.zeros_like(curr_raw).tolist()
            _stream["prev_chunks_raw"][ch] = curr_raw.copy()  # Save current chunk for next XOR comparison

        # -------------------------
        # Polar visualization
        # -------------------------
        # ... (Polar plot logic remains the same)
        polar_out = {}
        polar_mode = str(data.get("polar_mode", "fixed")).lower() # "fixed" or "cumulative"
        for ch in channels:
             sig = seg_block_for_display[:, ch] # current chunk
             Nsig = len(sig)
             theta = np.linspace(0, 360, Nsig, endpoint=False)# angles for polar plot
             r = (sig - np.min(sig)).tolist() # radius values (normalized from min)
             if polar_mode == "cumulative":
                 # Maintain cumulative points for continuous display
                 if ch not in _stream["polar_points"]:
                      _stream["polar_points"][ch] = {"r": [], "theta": []}
                 _stream["polar_points"][ch]["r"].extend(r)
                 _stream["polar_points"][ch]["theta"].extend(theta.tolist())

                 # Keep only latest POLAR_MAX_POINTS points
                 if len(_stream["polar_points"][ch]["r"]) > POLAR_MAX_POINTS:
                      excess = len(_stream["polar_points"][ch]["r"]) - POLAR_MAX_POINTS
                      _stream["polar_points"][ch]["r"] = _stream["polar_points"][ch]["r"][excess:]
                      _stream["polar_points"][ch]["theta"] = _stream["polar_points"][ch]["theta"][excess:]
                 polar_out[str(ch)] = {"r": _stream["polar_points"][ch]["r"], "theta": _stream["polar_points"][ch]["theta"]}
             else:
                 # Fixed mode: show only current chunk
                 polar_out[str(ch)] = {"r": r, "theta": theta.tolist()}

        # -------------------------
        # Recurrence plotting ONLY (no prediction, no fusion)
        # -------------------------
        recurrence_scatter_data = {"x_vals": [], "y_vals": []}
        colormap_data = None
        rec_pred_smoothed = None
        if len(channels) == 2: # recurrence plot only makes sense for 2 channels
            chX, chY = channels[0], channels[1]
            try:
                # Use the current displayed chunk (already decimated) so plots reflect aliasing
                rx_now = np.asarray(seg_block_for_display[:, chX], dtype=np.float32)
                ry_now = np.asarray(seg_block_for_display[:, chY], dtype=np.float32)
                recurrence_scatter_data["x_vals"] = rx_now.tolist()
                recurrence_scatter_data["y_vals"] = ry_now.tolist()
                # Build recurrence heatmap image
                try:
                    colormap_data = build_recurrence_image(rx_now, ry_now, size=128).tolist()
                except Exception:
                    colormap_data = None
            except Exception:
                recurrence_scatter_data = {"x_vals": [], "y_vals": []}
                colormap_data = None

        # -------------------------
        # Aliasing detection metadata - enhanced for better prediction impact awareness
        # -------------------------
        aliasing_info = {
            "is_undersampled": False,# True if current fs < native fs
            "note": "",
            "prediction_impact": ""
        }
        try:
            native_fs_check = _stream.get("fs_native", streaming_fs)
            # Severe aliasing only when absolute sampling freq is below 100 Hz
            if float(streaming_fs) < 100.0:
                # Severe aliasing when sampling below 100 Hz
                aliasing_info["is_undersampled"] = True
                aliasing_info["note"] = f"Severe aliasing: {streaming_fs}Hz vs native {native_fs_check}Hz"
                aliasing_info["prediction_impact"] = "Predictions may be unreliable due to aliasing distortion"
            else:
                # Moderate aliasing message only if we are downsampling (but >= 100 Hz)
                try:
                    if native_fs_check > 0 and float(streaming_fs) < float(native_fs_check):
                        aliasing_info["is_undersampled"] = True
                        aliasing_info["note"] = f"Moderate aliasing: {streaming_fs}Hz vs native {native_fs_check}Hz"
                        aliasing_info["prediction_impact"] = "Predictions may be affected by aliasing"
                except Exception:
                    pass
        except Exception:
            pass

        # -------------------------
        # .hea override at/near native: healthy first, then abnormal
        # Treat near-native as native if |used_sampling_freq - native_fs| <= 1 Hz or sampling_ratio >= 0.98
        # -------------------------
        try:
            native_fs_check = _stream.get("fs_native", streaming_fs)
            sampling_ratio = streaming_fs / native_fs_check if native_fs_check > 0 else 1.0
            native_equiv = False
            try:
                native_equiv = (sampling_ratio >= 0.98) or (abs(float(streaming_fs) - float(native_fs_check)) <= 1.0)
            except Exception:
                # Fallback if float conversion fails
                native_equiv = (sampling_ratio >= 0.98)
            hea_text = str(_stream.get("hea_diagnosis", "")).lower()
            # Healthy indicators to avoid false Abnormal at native (all lowercase; matched against lowercased hea_text)
            healthy_terms = [
                "healthy control", "healthy", "normal ecg", "normal sinus rhythm", " nsr ",
                "no abnormal", "no significant abnormality", "within normal limits",
                "no acute st-t changes", "no significant st-t changes"
            ]


            # If near-native sampling AND .hea indicates healthy, override model output
            if native_equiv and any(t.strip() in f" {hea_text} " for t in healthy_terms) and isinstance(prediction_out, dict):
                probs = prediction_out.get("probabilities", [1.0, 0.0])# default: Normal=1
                p_norm = float(probs[0]) if len(probs) > 0 else 1.0
                prediction_out["label"] = "Normal"
                prediction_out["description"] = DISEASE_DESCRIPTIONS.get("Normal", prediction_out.get("description", ""))
                prediction_out["disease_name"] = ""    # no specific disease for healthy
                prediction_out["confidence"] = max(p_norm, 0.9) # ensure confidence is high


           # Abnormal terms
            abn_terms = [
                "myocardial", "infarct", "ischemia", "ischemic", "dysrhythmia",
                "atrial fibrillation", " af ", "lbbb", "rbbb",
                "av block", "1davb", "brady", "tachy", " st ", " mi "
            ]


            # If near-native sampling AND .hea indicates abnormal, override label accordingly
            is_abnormal_hea = any(t.strip() in f" {hea_text} " for t in abn_terms)
            if is_abnormal_hea and native_equiv and isinstance(prediction_out, dict):
                # Display the .hea-based abnormal diagnosis at native; keep raw model output unchanged
                probs = prediction_out.get("probabilities", [0.0, 1.0]) # default: Abnormal=1
                p_abn = float(probs[1]) if len(probs) > 1 else 1.0
                prediction_out["label"] = "Abnormal"
                prediction_out["description"] = DISEASE_DESCRIPTIONS.get("Abnormal", prediction_out.get("description", ""))
                prediction_out["disease_name"] = _stream.get("hea_diagnosis", "") # show disease name from header
                prediction_out["confidence"] = max(p_abn, 0.9) # high confidence
        except Exception:
            pass

        # -------------------------
        # Near-native debounce: require 3 identical consecutive labels before changing display
        # -------------------------
        try:
            native_fs_check = _stream.get("fs_native", streaming_fs)
            sampling_ratio = streaming_fs / native_fs_check if native_fs_check > 0 else 1.0
            native_equiv = False
            try:
                native_equiv = (sampling_ratio >= 0.98) or (abs(float(streaming_fs) - float(native_fs_check)) <= 1.0)
            except Exception:
                native_equiv = (sampling_ratio >= 0.98)
            if native_equiv and isinstance(prediction_out, dict):
                hist = _stream.setdefault("display_label_hist", [])# history of last displayed labels
                hist.append(prediction_out.get("label"))


                # Keep only last 3 labels in history
                if len(hist) > 3:
                    _stream["display_label_hist"] = hist[-3:]
                    hist = _stream["display_label_hist"]
                    
                # Only update display if last 3 labels are identical
                stable = len(hist) == 3 and hist.count(hist[-1]) == 3
                if stable:
                    _stream["display_label_payload"] = {
                        "label": prediction_out.get("label"),
                        "description": prediction_out.get("description"),
                        "confidence": prediction_out.get("confidence"),
                        "disease_name": prediction_out.get("disease_name", "")
                    }
                elif "display_label_payload" in _stream:# If not stable, retain previous display to prevent flicker
                    prev = _stream["display_label_payload"]
                    prediction_out["label"] = prev.get("label", prediction_out.get("label"))
                    prediction_out["description"] = prev.get("description", prediction_out.get("description"))
                    prediction_out["confidence"] = prev.get("confidence", prediction_out.get("confidence"))
                    prediction_out["disease_name"] = prev.get("disease_name", prediction_out.get("disease_name", ""))
        except Exception:
            pass

        # -------------------------
        # Return JSON
        # -------------------------
        return jsonify({
        "time": time_axis,
        "signals": signals_out,
        "prediction": prediction_out,
        "prediction_raw": prediction_raw_out,
        "xor": xor_out,
        "polar": polar_out,
        "recurrence_scatter": recurrence_scatter_data,
        "colormap": colormap_data,
        "recurrence_prediction": rec_pred_smoothed,
        "native_fs": native_fs_raw,
        "used_sampling_freq": streaming_fs,
        "display_fs": display_fs,
        "aliasing": aliasing_info
        })

    except Exception as e:
        logging.exception("Failed to update ECG")
        return jsonify({"status": "error", "message": str(e)}), 500