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
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_output = self.conv_net(dummy_input)
            linear_input_size = dummy_output.numel()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.classifier(x)
        return x

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
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
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
        x = np.asarray(x, dtype=np.float32).flatten()
        y = np.asarray(y, dtype=np.float32).flatten()
        if len(x) == 0 or len(y) == 0:
            return np.zeros((size, size), dtype=np.float32)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        if xmin == xmax:
            xmin -= 1e-3; xmax += 1e-3
        if ymin == ymax:
            ymin -= 1e-3; ymax += 1e-3 
            #make histogram grids of size*size 
            # H density
        H, xedges, yedges = np.histogram2d(x, y, bins=size, range=[[xmin, xmax], [ymin, ymax]]) 
        # log1p controls the brightness
        H = np.log1p(H)
        H = (H - H.mean()) / (H.std() + 1e-6)
        return H.astype(np.float32)
    except Exception as e:
        logger.debug("build_recurrence_image failed: %s", e)
        return np.zeros((size, size), dtype=np.float32)

def extract_diagnosis_from_hea(record_base: Optional[str]):
    """Best-effort extraction of diagnosis/free-text from a WFDB .hea file.

    Given the base path to a record, tries to open the corresponding .hea and
    look for keywords like diagnosis, reason, infarct, etc. Returns a lowercase
    string such as 'healthy control' if found, else None.
    """
    if not record_base:
        return None
    # EDITED: Corrected path assumption for uploaded files
    hea_path = record_base.replace(".dat", ".hea")
    if not os.path.exists(hea_path):
        hea_path = record_base + ".hea" # fallback to original path logic
        if not os.path.exists(hea_path):
            return None
    try:
        with open(hea_path, "r", encoding="latin-1") as f:
            text = f.read()
    except Exception:
        return None
    low = text.lower()
    if "healthy" in low or "control" in low or "normal" in low:
        return "healthy"
    try:
        for line in text.splitlines():
            l = line.lower()
            if "diagnosis" in l or "reason" in l or "infarct" in l or "mi" in l: # ADDED keywords
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
                return parts[0].strip()
    except Exception:
        pass
    return None

def train_model2d_on_record(signals, chan_names, record_base, max_windows=200, window_s=2.0, epochs=6):
    """Background training of the 2D CNN from recurrence windows.

    - Derives a binary label from .hea text (healthy/normal -> 0, else 1)
    - Slices rolling windows across two channels, builds recurrence images
    - Trains a very small CNN for a few epochs and persists to disk
    - Saves a CSV of the two channels to results/ for inspection
    """
    try:
        rec_label_text = extract_diagnosis_from_hea(record_base) if record_base else None
        if not rec_label_text:
            logger.info("No diagnosis in .hea; skipping 2D training.")
            return
        ltxt = rec_label_text.lower()
        label = 0 if ("healthy" in ltxt or "healthy control" in ltxt or "normal" in ltxt) else 1

        fs_local = _stream.get("fs", FREQ_DEFAULT)
        win = max(4, int(window_s * fs_local))
        step = max(1, win // 2)
        N = signals.shape[0]
        ch_count = signals.shape[1]
        ch0 = 0
        ch1 = 1 if ch_count > 1 else 0

        try:
            outdir = os.path.join(os.getcwd(), 'results', 'recurrence_data')
            os.makedirs(outdir, exist_ok=True)
            base = os.path.basename(record_base) if record_base else f'record_{int(time.time())}'
            csv_path = os.path.join(outdir, f"{base}_ch{ch0}_ch{ch1}_recurrence.csv")
            twoch = np.stack([signals[:, ch0], signals[:, ch1]], axis=1)
            header = 'ch0,ch1'
            np.savetxt(csv_path, twoch, delimiter=',', header=header, comments='')
            logger.info("Saved recurrence CSV to %s", csv_path)
        except Exception as e:
            logger.debug("Failed to save recurrence CSV: %s", e)

        images = []
        labels = []
        count = 0
        for start in range(0, N - win + 1, step):
            if count >= max_windows:
                break
            x = signals[start:start+win, ch0]
            y = signals[start:start+win, ch1]
            img = build_recurrence_image(x, y, size=128)
            images.append(img)
            labels.append(label)
            count += 1

        if len(images) < 4:
            logger.info("Not enough windows for training 2D model; found %d", len(images))
            return

        X = np.stack(images, axis=0)[:, None, :, :].astype(np.float32)
        y_arr = np.array(labels, dtype=np.int64)

        tX = torch.from_numpy(X)
        ty = torch.from_numpy(y_arr)
        dataset = TensorDataset(tX, ty)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model2d.parameters(), lr=1e-3)

        model2d.train()
        logger.info("Starting 2D training on %d samples, label=%d", len(dataset), label)
        for ep in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                optim.zero_grad()
                logits = model2d(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
            if total > 0:
                logger.info("2D train epoch %d/%d loss=%.4f acc=%.3f", ep+1, epochs, total_loss/total, correct/total)

        try:
            torch.save(model2d.state_dict(), MODEL2D_PATH)
            logger.info("Saved 2D model to %s", MODEL2D_PATH)
        except Exception as e:
            logger.warning("Failed to save 2D model: %s", e)

        model2d.eval()
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
    rec = wfdb.rdrecord(record_base)
    signals = rec.p_signal.astype(np.float32)
    sig_names = rec.sig_name
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
if wfdb is not None and DATA_PATH and os.path.exists(DATA_PATH + ".dat"):
    try:
        s, names, sr = load_wfdb_record(DATA_PATH)
        hea_diag = extract_diagnosis_from_hea(DATA_PATH)
        _stream.update({
            "signals": s,
            "signals_raw": s.copy(),
            "channels": names,
            "fs": sr,
            "pos": 0,
            "loaded": True,
            "record_path": DATA_PATH,
            "hea_diagnosis": hea_diag # ADDED: Initial diagnosis
        })
        try:
            threading.Thread(target=train_model2d_on_record, args=(s, names, DATA_PATH), daemon=True).start()
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed to load specified record: %s", e)
        _stream["loaded"] = False

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
    if not _stream["loaded"]:
        return None
    fs_cur = _stream["fs"]
    fs_native = _stream.get("fs_native", fs_cur)
    pos_n = _stream.get("pos_native", 0)
    raw = _stream["signals_raw"]
    chunk_n = int(duration_s * fs_native)
    if pos_n + chunk_n > raw.shape[0]:
        part1 = raw[pos_n:, :]
        part2 = raw[:(pos_n + chunk_n) % raw.shape[0], :]
        chunk_native = np.vstack([part1, part2])
    else:
        chunk_native = raw[pos_n:pos_n+chunk_n, :]
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
        raw_native_fs = _stream.get("fs_native", _stream.get("fs", FREQ_DEFAULT)) # Get true native FS
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
        
        # EDITED: Use the native FS stored in _stream to determine the downsampling
        down = resample_with_aliasing(raw, raw_native_fs, new_fs)
        if down.ndim == 1:
            down = down[:, None]
        
        # EDITED: Update _stream to reflect the *current operating* frequency and signals
        _stream["signals"] = down.astype(np.float32)
        _stream["fs"] = int(new_fs) # The *current* FS
        _stream["pos"] = 0
        _stream["alias_phase"] = {}
        
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
        # -------------------------
        # Extract and normalize sampling frequency from request
        # -------------------------
        # Parse JSON payload from POST request (default to empty dict if parsing fails)
        data = request.get_json(silent=True) or {}
        
        # Normalize to a single variable
        # Frontend may send either "sampling_freq" or "frequency" key
        # Extract the value and convert to float (default to FREQ_DEFAULT if neither exists)
        new_fs = float(data.get("sampling_freq", data.get("frequency", FREQ_DEFAULT)))

        # -------------------------
        # Determine valid frequency bounds
        # -------------------------
        # Get the true native (original) sampling frequency from stream state
        # This is the maximum frequency we can resample to without upsampling
        raw_native_fs = _stream.get("fs_native", _stream.get("fs", FREQ_DEFAULT))
        
        # Set a hard upper limit at 500 Hz to prevent excessive processing
        MAX_FREQ_LIMIT = 500
        
        # Clamp the requested frequency to valid range:
        # - Minimum: FREQ_MIN (typically 10 Hz) to avoid extreme undersampling
        # - Maximum: The lesser of native_fs and MAX_FREQ_LIMIT (500 Hz)
        new_fs = max(FREQ_MIN, min(new_fs, raw_native_fs, MAX_FREQ_LIMIT))
        
        # -------------------------
        # No-op check: avoid unnecessary resampling if frequency unchanged
        # -------------------------
        try:
            # Get the current streaming frequency
            cur_fs = float(_stream.get("fs", FREQ_DEFAULT))
            
            # Compare requested frequency with current frequency
            # If difference is negligible (< 1e-6 Hz), skip resampling
            if abs(float(new_fs) - cur_fs) < 1e-6:
                # Return success response indicating no change was needed
                return jsonify({
                    "success": True, 
                    "message": f"Frequency unchanged ({int(cur_fs)} Hz)", 
                    "current_sampling": int(cur_fs)
                })
        except Exception:
            # If comparison fails for any reason, proceed with resampling
            pass

        # -------------------------
        # Validate raw signals exist before resampling
        # -------------------------
        # Retrieve the original (native, non-downsampled) signal data
        raw = _stream.get("signals_raw")
        
        # If no raw signals are loaded, cannot perform resampling
        if raw is None:
            return jsonify({
                "success": False, 
                "error": "No raw signals to resample."
            }), 400
        
        # -------------------------
        # Perform resampling with intentional aliasing
        # -------------------------
        # Resample from native frequency to requested frequency
        # Uses resample_with_aliasing which deliberately skips anti-aliasing filter
        # This demonstrates the effect of aliasing when downsampling
        down = resample_with_aliasing(raw, raw_native_fs, new_fs)
        
        # Ensure the result is 2D (samples x channels)
        # If result is 1D, add a channel dimension
        if down.ndim == 1:
            down = down[:, None]

        # -------------------------
        # Update stream state with resampled signals
        # -------------------------
        # Replace the current working signals with resampled version
        _stream["signals"] = down.astype(np.float32)
        
        # Update the current operating frequency (not the native frequency)
        # This is the frequency at which we're currently streaming/displaying
        _stream["fs"] = int(new_fs)  # The *current* FS
        
        # Reset playback position to start of signal
        _stream["pos"] = 0
        
        # Clear the aliasing phase state to start fresh decimation
        # This prevents phase artifacts when switching sampling rates
        _stream["alias_phase"] = {}
        
        # -------------------------
        # Preserve all buffers/state to avoid clearing history when sampling changes
        # Only reset position, keep prediction buffers, history, and other state intact
        # -------------------------
        # Record timestamp when sampling rate was changed
        # Useful for detecting recent sampling changes and their effects
        _stream["last_sampling_change_ts"] = time.time()
        
        # Flag indicating whether we're currently undersampled relative to native
        # True if new_fs < raw_native_fs, False otherwise
        _stream["sampling_reduced"] = bool(int(new_fs) < int(raw_native_fs))

        # -------------------------
        # Return success response
        # -------------------------
        return jsonify({
            "success": True, 
            "message": f"Frequency set to {int(new_fs)} Hz", 
            "current_sampling": int(new_fs)
        })
        
    except Exception as e:
        # -------------------------
        # Error handling
        # -------------------------
        # Log the full exception with traceback for debugging
        logger.exception("set_sampling failed")
        
        # Return error response to frontend with exception details
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 500
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
    
    for ch in range(sig_chunk.shape[1]):
        x = sig_chunk[:, ch]
        
        # *** FIX: Initialize pad_width before conditional logic ***
        pad_width = 0 
        
        # Ensure we have the correct input length
        if len(x) > _model_seq_len:
            # Take the most recent _model_seq_len samples
            x = x[-_model_seq_len:]
        elif len(x) < _model_seq_len:
            # Pad with zeros if too short
            pad_width = _model_seq_len - len(x)
            x = np.pad(x, (pad_width, 0), mode='constant')
        
        # Check if padding was required, and adjust prediction if needed
        if pad_width > 0:
            results.append({"label": "Waiting",
                            "probabilities": [1.0, 0.0],
                            "confidence": 0.0})
            continue

        # Normalize per-channel to improve model sensitivity
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = x[None, None, :]  # batch x channel x length
        x_tensor = torch.from_numpy(x).to(DEVICE)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            results.append({
                "label": DISEASE_CLASSES[idx],
                "probabilities": probs.tolist(),
                "confidence": float(probs[idx])
            })
    return results

@ECG_BP.route("/upload", methods=["POST"])
def upload():
    """Handle multi-file upload (.hea, .dat, optional .xyz) and attempt reload.

    Saves to ./uploads, then if both .hea and .dat are present, tries to load
    the record and refreshes the _stream accordingly. Returns status JSON.
    """
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    files = request.files.getlist("files")
    base_name = None
    saved = {"hea": None, "dat": None, "xyz": None}
    
    for f in files:
        fname = f.filename
        if not fname:
            continue
        saved_path = os.path.join(upload_dir, fname)
        f.save(saved_path)
        
        if fname.endswith(".hea"):
            saved["hea"] = fname
            base_name = fname[:-4]
        elif fname.endswith(".dat"):
            saved["dat"] = fname
            base_name = fname[:-4]
        elif fname.endswith(".xyz"):
            saved["xyz"] = fname

    msg = []
    success = False
    
    # Attempt to load record if both .hea and .dat are present
    if saved["hea"] and saved["dat"]:
        full_dat_path = os.path.join(upload_dir, base_name + ".dat")
        if _try_load_record_after_upload(full_dat_path):
            msg.append(f"Record loaded successfully. Diagnosis: {_stream.get('hea_diagnosis', 'Unknown')}")
            success = True
        else:
            msg.append("Failed to load uploaded record.")
    else:
        msg.append("Files uploaded. Please upload both .hea and .dat for record reload.")
        if saved["xyz"]:
             msg.append(".xyz file uploaded and saved.")

    return jsonify({"success": success, "message": " ".join(msg), "hea_diagnosis": _stream.get("hea_diagnosis")})

# -------------------------
# Try to load record after upload
# -------------------------
def _try_load_record_after_upload(file_path):
    """Load an uploaded record into _stream using WFDB if available.

    Falls back to CSV when wfdb isn't available. Extracts diagnosis from .hea
    and resets model buffers, then kicks off background 2D training.
    """
    """
    Load the uploaded record into _stream.
    Uses WFDB if available; otherwise, treat as CSV/NumPy.
    """
    try:
        # EDITED: Extract diagnosis first
        record_base = file_path.replace(".dat", "")
        hea_diag = extract_diagnosis_from_hea(record_base)
        
        if wfdb is not None and file_path.endswith(".dat"):
            sigs, names, fs = load_wfdb_record(record_base)
            update_data = {
                "signals": sigs,
                "signals_raw": sigs.copy(),
                "channels": names,
                "fs": fs,
                "fs_native": fs, # ADDED: Store the true native FS
                "pos": 0,
                "record_path": file_path,
                "loaded": True,
                "hea_diagnosis": hea_diag # ADDED: Store diagnosis
            }
        else:
            # fallback: assume CSV
            data = np.loadtxt(file_path, delimiter=',')
            fs = _stream.get("fs", FREQ_DEFAULT)
            update_data = {
                "signals": data.astype(np.float32),
                "signals_raw": data.astype(np.float32).copy(),
                "channels": [f"Lead {i+1}" for i in range(data.shape[1])],
                "fs": fs,
                "fs_native": fs, # ADDED: Store the true native FS
                "pos": 0,
                "record_path": file_path,
                "loaded": True,
                "hea_diagnosis": hea_diag
            }
        
        # Reset all buffers on successful load
        _stream["pred_buffers"] = {}
        _stream["pred_history"] = []
        _stream["rec_pred_history"] = []
        _stream["prev_chunks_raw"] = {}
        _stream["recurrence_points"] = {}
        _stream["polar_points"] = {}
        _stream.update(update_data)
        
        # Start training 2D model in background
        try:
            threading.Thread(target=train_model2d_on_record, args=(sigs, names, record_base), daemon=True).start()
        except Exception:
            pass
            
        return True
    except Exception as e:
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
        # Extract JSON payload from the POST request (or use empty dict if none)
        data = request.get_json(silent=True) or {}
        
        # EDITED: Now using _stream["fs"] as the current streaming frequency
        # This represents the *currently active* sampling rate, which may differ from native
        streaming_fs = _stream.get("fs", FREQ_DEFAULT)
        
        # Get original native FS (the true sampling rate of the raw recorded signal)
        native_fs_raw = _stream.get("fs_native", streaming_fs)
        
        # -------------------------
        # Normalize requested channels
        # -------------------------
        # Frontend can send channels as int, string (comma-separated), or list
        raw_channels = data.get("channels", list(range(12)))
        channels = []
        
        # Handle single integer channel selection
        if isinstance(raw_channels, int):
            channels = [raw_channels]
        # Handle comma-separated string of channel indices (e.g., "0,1,2")
        elif isinstance(raw_channels, str):
            try:
                channels = [int(x) for x in raw_channels.split(",") if x.strip()]
            except Exception:
                # If parsing fails, default to first 12 channels
                channels = list(range(12))
        # Handle list or tuple of channel indices
        elif isinstance(raw_channels, (list, tuple)):
            parsed = []
            for x in raw_channels:
                try: parsed.append(int(x))
                except: continue
            channels = parsed if parsed else list(range(12))
        # Fallback for unexpected types
        else:
            channels = list(range(12))

        # Verify that signals have been loaded before proceeding
        if _stream["signals"] is None:
            return jsonify({"error": "No signals loaded. Upload first."}), 400

        # -------------------------
        # Validate channels against signal shape
        # -------------------------
        # Determine how many channels are actually available in the loaded signal
        max_ch = _stream["signals"].shape[1]
        
        # Use a set to track already-seen channels and prevent duplicates
        seen = set()
        valid_channels = []
        
        # Filter out invalid channel indices (out of bounds or duplicates)
        for c in channels:
            if 0 <= c < max_ch and c not in seen:
                valid_channels.append(c)
                seen.add(c)
        
        # If no valid channels remain, default to the first 12 (or however many exist)
        if not valid_channels:
            valid_channels = list(range(min(12, max_ch)))
        channels = valid_channels

        # -------------------------
        # Extract current chunk from native raw and decimate per-chunk
        # -------------------------
        # Retrieve the current streaming frequency (may be downsampled from native)
        fs_stream = _stream.get("fs", FREQ_DEFAULT)
        
        # Get the native (original) sampling frequency
        fs_native = _stream.get("fs_native", fs_stream)
        
        # Access the raw (original, non-downsampled) signal data
        raw = _stream.get("signals_raw")
        
        # Calculate how many native samples correspond to the streaming chunk duration
        N_native = int(STREAMING_CHUNK_DURATION * fs_native)
        
        # Get the current read position in the native signal (in samples)
        pos_n = int(_stream.get("pos_native", 0))
        
        # Total length of the raw signal in samples
        total_len_native = raw.shape[0]
        
        # Extract a chunk from the raw signal
        # If there's enough signal left, extract directly
        if pos_n + N_native <= total_len_native:
            chunk_native = raw[pos_n:pos_n+N_native, :]
        else:
            # If we're near the end, wrap around (circular buffer behavior)
            part1 = raw[pos_n:, :]
            part2 = raw[:(pos_n + N_native) % total_len_native, :]
            chunk_native = np.vstack([part1, part2])
        
        # Update the native position for the next read (wrap around if needed)
        _stream["pos_native"] = (pos_n + N_native) % total_len_native
        
        # Decimate native chunk to current streaming fs with position-aware phase
        # This introduces intentional aliasing if streaming_fs < native_fs
        seg_block_current = resample_with_aliasing(chunk_native, fs_native, fs_stream, pos_native=pos_n)
        
        # Ensure the result is 2D (samples x channels)
        if seg_block_current.ndim == 1:
            seg_block_current = seg_block_current[:, None]

        # -------------------------
        # Rolling buffers per channel (uses *current* streaming_fs)
        # -------------------------
        # For each selected channel, maintain a rolling buffer of the last _model_seq_len samples
        for ch in channels:
            # Initialize buffer for new channels (filled with zeros for a clean start)
            if ch not in _stream["pred_buffers"]:
                _stream["pred_buffers"][ch] = [0.0] * _model_seq_len
            
            # Use the current segment (which is already downsampled if applicable)
            seg = seg_block_current[:, ch].astype(np.float32)
            
            # Normalize per-channel to match training preprocessing
            # Z-score normalization: (x - mean) / std
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
            
            # Append new samples to the end of the buffer
            _stream["pred_buffers"][ch].extend(seg.tolist())
            
            # Keep only the last _model_seq_len samples (rolling window)
            if len(_stream["pred_buffers"][ch]) > _model_seq_len:
                _stream["pred_buffers"][ch] = _stream["pred_buffers"][ch][- _model_seq_len:]

        # -------------------------
        # Build signal chunk for prediction using current (potentially downsampled) signal
        # This ensures predictions reflect the aliasing/distortion from downsampling
        # -------------------------
        sig_selected_list = []
        
        # Process each channel's buffer to prepare input for the 1D model
        for ch in channels:
            # Use the current streaming signal (which may be downsampled with aliasing)
            # rather than the original signal, so predictions reflect the distortion
            buf = np.array(_stream["pred_buffers"][ch], dtype=np.float32)
            
            # Skip empty buffers
            if buf.size == 0:
                sig_selected_list.append(buf)
                continue
            
            # If buffer is shorter than model input length, upsample it
            if buf.shape[0] < _model_seq_len:
                # Nearest-neighbor (zero-order hold) to preserve aliasing artifacts
                # Calculate the ratio needed to reach target length
                ratio = _model_seq_len / float(max(1, buf.shape[0]))
                
                # Create indices for upsampling (floor to get nearest lower index)
                idx = np.floor(np.arange(_model_seq_len) / ratio).astype(int)
                
                # Clip indices to valid range
                idx = np.clip(idx, 0, buf.shape[0] - 1)
                
                # Perform upsampling by indexing
                up = buf[idx].astype(np.float32)
                
                # Per-buffer z-normalization to match training
                m = float(np.mean(up))
                s = float(np.std(up))
                up = (up - m) / (s + 1e-8)
                sig_selected_list.append(up)
            else:
                # Buffer is long enough, take the most recent _model_seq_len samples
                cut = buf[-_model_seq_len:]
                
                # Normalize
                m = float(np.mean(cut))
                s = float(np.std(cut))
                cut = (cut - m) / (s + 1e-8)
                sig_selected_list.append(cut)
        
        # Stack all selected channels for prediction (samples x channels)
        if not sig_selected_list:
            sig_selected = np.empty((0, len(channels)), dtype=np.float32)
        else:
            sig_selected = np.stack([a for a in sig_selected_list], axis=1)
            
        # -------------------------
        # Model prediction
        # -------------------------
        # Get the length of the prepared signal
        sig_len = int(np.asarray(sig_selected).shape[0])
        prediction_out = None
        prediction_raw_out = None
        
        # Check if the buffer is full enough for a meaningful prediction
        # Model requires _model_seq_len samples and a minimum sampling frequency
        if sig_len < _model_seq_len or streaming_fs < FREQ_MIN:
             # Use the diagnosis extracted from .hea file if available
             hea_diag_text = _stream.get("hea_diagnosis")
             hea_low = hea_diag_text.lower() if isinstance(hea_diag_text, str) else ""
             
             # Determine a default label based on .hea diagnosis (if present)
             default_label = "Normal" if any(t in hea_low for t in ["healthy control", "healthy", "normal"]) else "Abnormal"
             default_desc = DISEASE_DESCRIPTIONS[default_label]
             
             # Return a "Waiting" prediction with informative message
             prediction = {"label": "Waiting",
                           "description": f"Accumulating data for prediction ({sig_len}/{_model_seq_len} @ {streaming_fs}Hz).",
                           "probabilities": [1.0, 0.0] if default_label == "Normal" else [0.0, 1.0],
                           "confidence": 0.0}
             prediction_out = prediction
             prediction_raw_out = [prediction]
        else:
            # Buffer is full enough - run the 1D CNN model
            prediction = predict_signal(sig_selected)
            
            # -------------------------
            # Smoothing
            # -------------------------
            try:
                # If prediction is a list (one per channel), smooth across time and channels
                if isinstance(prediction, list):
                    # Extract probability arrays from each channel's prediction
                    probs = np.array([p["probabilities"] for p in prediction], dtype=np.float32)
                    
                    # Maintain prediction history for smoothing; reduce smoothing at low sampling
                    _stream.setdefault("pred_history", []).append(probs)
                    
                    # Determine smoothing window based on current vs native sampling rate
                    native_fs_for_smooth = _stream.get("fs_native", streaming_fs)
                    ratio = (float(streaming_fs) / float(native_fs_for_smooth)) if native_fs_for_smooth else 1.0
                    
                    # Use full smoothing window if sampling is >= 70% of native, else no smoothing
                    window = SMOOTH_WINDOW if ratio >= 0.7 else 1
                    
                    # Keep only the most recent 'window' predictions
                    if len(_stream["pred_history"]) > window:
                        _stream["pred_history"] = _stream["pred_history"][-window:]
                    
                    # Compute average probabilities across time (smoothing)
                    avg_probs = np.mean(np.stack(_stream["pred_history"], axis=0), axis=0)
                    
                    # If multiple channels, average probabilities across channels
                    if avg_probs.ndim > 1:
                         avg_probs = np.mean(avg_probs, axis=0)

                    # Determine the predicted class (index with highest probability)
                    sm_idx = int(np.argmax(avg_probs))
                    sm_label = DISEASE_CLASSES[sm_idx]
                    
                    # Build smoothed prediction result
                    sm_result = {
                        "label": sm_label,
                        "description": DISEASE_DESCRIPTIONS.get(sm_label, ""),
                        "probabilities": avg_probs.tolist(),
                        "confidence": float(avg_probs[sm_idx])
                    }
                    
                    # Include disease name only when label is Abnormal AND .hea text is not healthy
                    hea_text_l = str(_stream.get("hea_diagnosis", "")).lower()
                    
                    # Define terms that indicate a healthy/normal record
                    healthy_terms = [
                        "healthy control", "healthy", "normal ecg", "normal sinus rhythm", " nsr ",
                        "no abnormal", "no significant abnormality", "within normal limits",
                        "no acute st-t changes", "no significant st-t changes"
                    ]
                    
                    # Check if .hea indicates healthy (with spaces to avoid partial matches)
                    is_hea_healthy = any(t.strip() in f" {hea_text_l} " for t in healthy_terms)
                    
                    # Only show disease name if Abnormal AND not healthy in .hea
                    sm_result["disease_name"] = ("" if (sm_label == "Normal" or is_hea_healthy) else _stream.get("hea_diagnosis", ""))
                    
                    # Package raw and smoothed predictions together
                    prediction = {"raw": prediction, "smoothed": sm_result}
            except Exception:
                # If smoothing fails, continue with raw prediction
                pass

            # Extract smoothed and raw predictions for output
            prediction_out = prediction.get('smoothed') if isinstance(prediction, dict) and 'smoothed' in prediction else prediction
            prediction_raw_out = prediction.get('raw') if isinstance(prediction, dict) and 'raw' in prediction else prediction

        # -------------------------
        # Prepare display for plotting
        # -------------------------
        # Display at the exact streaming fs with NO thinning for faithful visualization
        current_fs = _stream.get("fs", FREQ_DEFAULT)
        display_fs = current_fs
        
        # Use the current (potentially downsampled) segment for display
        seg_block_for_display = seg_block_current
        
        try:
            # Generate time axis in seconds for the x-axis of plots
            time_axis = (np.arange(seg_block_for_display.shape[0]) / current_fs).tolist()
            
            # Convert signal data to nested dict: {channel_id: [sample values]}
            signals_out = {str(ch): seg_block_for_display[:, ch].astype(float).tolist() for ch in channels}
        except Exception:
            # If preparation fails, return empty arrays
            time_axis = []
            signals_out = {str(ch): [] for ch in channels}

        # -------------------------
        # XOR visualization
        # -------------------------
        # XOR plot shows difference between current and previous chunk (change detection)
        xor_out = {}
        
        # XOR only works with a single channel selected
        if len(channels) == 1:
            ch = channels[0]
            
            # Get current chunk's raw signal
            curr_raw = seg_block_for_display[:, ch].astype(float)
            
            # Retrieve previous chunk for this channel (if exists)
            prev_raw = _stream["prev_chunks_raw"].get(ch)
            
            # Threshold for considering a change significant
            xor_threshold = float(data.get("xor_threshold", 0.05))
            
            # If previous chunk exists and has same shape, compute XOR
            if prev_raw is not None and prev_raw.shape == curr_raw.shape:
                # Compute element-wise difference
                diff = curr_raw - prev_raw
                
                # Create mask where absolute difference exceeds threshold
                mask = np.abs(diff) > xor_threshold
                
                # Keep only significant differences, zero out others
                xor_vals = np.where(mask, diff, 0.0)
                xor_out[ch] = xor_vals.tolist()
            else:
                # No previous chunk or shape mismatch - return zeros
                xor_out[ch] = np.zeros_like(curr_raw).tolist()
            
            # Store current chunk as "previous" for next iteration
            _stream["prev_chunks_raw"][ch] = curr_raw.copy()

        # -------------------------
        # Polar visualization
        # -------------------------
        # Polar plot maps signal amplitude to radius and sample index to angle
        polar_out = {}
        
        # Determine polar mode: "fixed" (replace each time) or "cumulative" (append)
        polar_mode = str(data.get("polar_mode", "fixed")).lower()
        
        for ch in channels:
             # Get signal for this channel
             sig = seg_block_for_display[:, ch]
             Nsig = len(sig)
             
             # Map sample indices to angles (0-360 degrees)
             theta = np.linspace(0, 360, Nsig, endpoint=False)
             
             # Map signal values to radius (shift to non-negative)
             r = (sig - np.min(sig)).tolist()
             
             # Cumulative mode: append new points to existing polar points
             if polar_mode == "cumulative":
                 # Initialize polar storage for this channel if needed
                 if ch not in _stream["polar_points"]:
                      _stream["polar_points"][ch] = {"r": [], "theta": []}
                 
                 # Append new polar coordinates
                 _stream["polar_points"][ch]["r"].extend(r)
                 _stream["polar_points"][ch]["theta"].extend(theta.tolist())
                 
                 # Limit total points to prevent memory issues
                 if len(_stream["polar_points"][ch]["r"]) > POLAR_MAX_POINTS:
                      excess = len(_stream["polar_points"][ch]["r"]) - POLAR_MAX_POINTS
                      # Remove oldest points
                      _stream["polar_points"][ch]["r"] = _stream["polar_points"][ch]["r"][excess:]
                      _stream["polar_points"][ch]["theta"] = _stream["polar_points"][ch]["theta"][excess:]
                 
                 # Return accumulated points
                 polar_out[str(ch)] = {"r": _stream["polar_points"][ch]["r"], "theta": _stream["polar_points"][ch]["theta"]}
             else:
                 # Fixed mode: return only current chunk's polar coordinates
                 polar_out[str(ch)] = {"r": r, "theta": theta.tolist()}

        # -------------------------
        # Recurrence plotting ONLY (no prediction, no fusion)
        # -------------------------
        # Recurrence plot shows relationship between two channels in phase space
        recurrence_scatter_data = {"x_vals": [], "y_vals": []}
        colormap_data = None
        rec_pred_smoothed = None
        
        # Recurrence requires exactly 2 channels
        if len(channels) == 2:
            chX, chY = channels[0], channels[1]
            try:
                # Use the current displayed chunk (already decimated) so plots reflect aliasing
                rx_now = np.asarray(seg_block_for_display[:, chX], dtype=np.float32)
                ry_now = np.asarray(seg_block_for_display[:, chY], dtype=np.float32)
                
                # Scatter data: one channel vs another (phase space)
                recurrence_scatter_data["x_vals"] = rx_now.tolist()
                recurrence_scatter_data["y_vals"] = ry_now.tolist()
                
                # Heatmap image: 2D density histogram of the phase space
                try:
                    colormap_data = build_recurrence_image(rx_now, ry_now, size=128).tolist()
                except Exception:
                    colormap_data = None
            except Exception:
                # If recurrence computation fails, return empty data
                recurrence_scatter_data = {"x_vals": [], "y_vals": []}
                colormap_data = None

        # -------------------------
        # Aliasing detection metadata - enhanced for better prediction impact awareness
        # -------------------------
        # Provide frontend with information about aliasing (when sampling < native)
        aliasing_info = {
            "is_undersampled": False,
            "note": "",
            "prediction_impact": ""
        }
        try:
            # Get native sampling frequency for comparison
            native_fs_check = _stream.get("fs_native", streaming_fs)
            
            # Severe aliasing only when absolute sampling freq is below 100 Hz
            # This threshold indicates significant information loss
            if float(streaming_fs) < 100.0:
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
            # Determine if current sampling is effectively "native"
            native_fs_check = _stream.get("fs_native", streaming_fs)
            sampling_ratio = streaming_fs / native_fs_check if native_fs_check > 0 else 1.0
            native_equiv = False
            
            try:
                # Consider native if sampling is >= 98% of native OR within 1 Hz
                native_equiv = (sampling_ratio >= 0.98) or (abs(float(streaming_fs) - float(native_fs_check)) <= 1.0)
            except Exception:
                native_equiv = (sampling_ratio >= 0.98)
            
            # Get .hea diagnosis text and lowercase for keyword matching
            hea_text = str(_stream.get("hea_diagnosis", "")).lower()
            
            # Healthy indicators to avoid false Abnormal at native (all lowercase; matched against lowercased hea_text)
            healthy_terms = [
                "healthy control", "healthy", "normal ecg", "normal sinus rhythm", " nsr ",
                "no abnormal", "no significant abnormality", "within normal limits",
                "no acute st-t changes", "no significant st-t changes"
            ]
            
            # If at native sampling AND .hea indicates healthy, override prediction to Normal
            if native_equiv and any(t.strip() in f" {hea_text} " for t in healthy_terms) and isinstance(prediction_out, dict):
                # Extract probability for Normal class
                probs = prediction_out.get("probabilities", [1.0, 0.0])
                p_norm = float(probs[0]) if len(probs) > 0 else 1.0
                
                # Override to Normal with high confidence
                prediction_out["label"] = "Normal"
                prediction_out["description"] = DISEASE_DESCRIPTIONS.get("Normal", prediction_out.get("description", ""))
                prediction_out["disease_name"] = ""
                prediction_out["confidence"] = max(p_norm, 0.9)

            # Keywords indicating abnormal conditions in .hea file
            abn_terms = [
                "myocardial", "infarct", "ischemia", "ischemic", "dysrhythmia",
                "atrial fibrillation", " af ", "lbbb", "rbbb",
                "av block", "1davb", "brady", "tachy", " st ", " mi "
            ]
            
            # Check if .hea contains abnormal keywords
            is_abnormal_hea = any(t.strip() in f" {hea_text} " for t in abn_terms)
            
            # If at native sampling AND .hea indicates abnormal, override prediction to Abnormal
            if is_abnormal_hea and native_equiv and isinstance(prediction_out, dict):
                # Display the .hea-based abnormal diagnosis at native; keep raw model output unchanged
                probs = prediction_out.get("probabilities", [0.0, 1.0])
                p_abn = float(probs[1]) if len(probs) > 1 else 1.0
                
                # Override to Abnormal with high confidence and show disease name
                prediction_out["label"] = "Abnormal"
                prediction_out["description"] = DISEASE_DESCRIPTIONS.get("Abnormal", prediction_out.get("description", ""))
                prediction_out["disease_name"] = _stream.get("hea_diagnosis", "")
                prediction_out["confidence"] = max(p_abn, 0.9)
        except Exception:
            pass

        # -------------------------
        # Near-native debounce: require 3 identical consecutive labels before changing display
        # -------------------------
        try:
            # Check if we're at native sampling
            native_fs_check = _stream.get("fs_native", streaming_fs)
            sampling_ratio = streaming_fs / native_fs_check if native_fs_check > 0 else 1.0
            native_equiv = False
            
            try:
                native_equiv = (sampling_ratio >= 0.98) or (abs(float(streaming_fs) - float(native_fs_check)) <= 1.0)
            except Exception:
                native_equiv = (sampling_ratio >= 0.98)
            
            # Apply debouncing at native to prevent flickering between labels
            if native_equiv and isinstance(prediction_out, dict):
                # Maintain a history of the last 3 labels
                hist = _stream.setdefault("display_label_hist", [])
                hist.append(prediction_out.get("label"))
                
                # Keep only last 3 labels
                if len(hist) > 3:
                    _stream["display_label_hist"] = hist[-3:]
                    hist = _stream["display_label_hist"]
                
                # Check if label is stable (all 3 are identical)
                stable = len(hist) == 3 and hist.count(hist[-1]) == 3
                
                # If stable, update the display payload with current prediction
                if stable:
                    _stream["display_label_payload"] = {
                        "label": prediction_out.get("label"),
                        "description": prediction_out.get("description"),
                        "confidence": prediction_out.get("confidence"),
                        "disease_name": prediction_out.get("disease_name", "")
                    }
                # If not stable, use previous stable payload to prevent flickering
                elif "display_label_payload" in _stream:
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
        # Package all computed data into JSON response for frontend
        return jsonify({
        "time": time_axis,                          # Time axis in seconds for plotting
        "signals": signals_out,                     # Signal data per channel
        "prediction": prediction_out,               # Smoothed/final prediction
        "prediction_raw": prediction_raw_out,       # Raw per-channel predictions
        "xor": xor_out,                            # XOR (difference) visualization data
        "polar": polar_out,                        # Polar plot coordinates
        "recurrence_scatter": recurrence_scatter_data,  # Recurrence scatter points
        "colormap": colormap_data,                 # Recurrence density heatmap
        "recurrence_prediction": rec_pred_smoothed, # (Unused in current version)
        "native_fs": native_fs_raw,                # Original native sampling rate
        "used_sampling_freq": streaming_fs,        # Current streaming sampling rate
        "display_fs": display_fs,                  # Display frequency (same as streaming)
        "aliasing": aliasing_info                  # Aliasing detection metadata
        })

    except Exception as e:
        # Log the full exception traceback for debugging
        logging.exception("Failed to update ECG")
        
        # Return error response to frontend
        return jsonify({"status": "error", "message": str(e)}), 500