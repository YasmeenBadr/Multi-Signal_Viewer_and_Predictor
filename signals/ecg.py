# ecg.py
import os
import time
import logging
import threading
from typing import Optional

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
        H, xedges, yedges = np.histogram2d(x, y, bins=size, range=[[xmin, xmax], [ymin, ymax]])
        H = np.log1p(H)
        H = (H - H.mean()) / (H.std() + 1e-6)
        return H.astype(np.float32)
    except Exception as e:
        logger.debug("build_recurrence_image failed: %s", e)
        return np.zeros((size, size), dtype=np.float32)

def extract_diagnosis_from_hea(record_base: Optional[str]):
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
    if wfdb is None:
        raise RuntimeError("wfdb package not available in environment")
    rec = wfdb.rdrecord(record_base)
    signals = rec.p_signal.astype(np.float32)
    sig_names = rec.sig_name
    fs = int(rec.fs) if hasattr(rec, "fs") else _stream.get("fs", FREQ_DEFAULT)
    return signals, sig_names, fs

def setup_simulated_record():
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
    """Strict decimation without anti-aliasing using a persistent phase.
    - Integer factor: take every k-th sample starting at stored phase p in [0,k-1].
    - Non-integer: use phase-accumulator with initial offset p in [0,factor).
    Phase is stored in `_stream["alias_phase"][int(target_fs)]` and reset on sampling change.
    """
    sig = np.asarray(sig, dtype=np.float32)
    if target_fs <= 0 or target_fs >= native_fs:
        return sig

    factor = float(native_fs) / float(target_fs)
    if factor <= 1.0:
        return sig

    # Get or initialize persistent phase for this target fs
    phase_map = _stream.setdefault("alias_phase", {})
    key = int(round(target_fs))
    # For multi-channel signals we store a dict of phases per channel under this key
    p_entry = phase_map.get(key, None)

    N = sig.shape[0]
    # Integer decimation fast path
    k = int(np.floor(factor))
    if abs(factor - k) < 1e-6 and k >= 2:
        if sig.ndim == 1:
            if p_entry is None or not isinstance(p_entry, (int, float)):
                p = int(np.random.randint(0, k))
                phase_map[key] = p
            else:
                p = int(p_entry)
            # position-aware phase to avoid repeatedly sampling R-peaks
            start = int((min(max(p, 0), k - 1) + (pos_native % k)) % k)
            return sig[start::k]
        else:
            # per-channel phase
            C = sig.shape[1]
            if p_entry is None or not isinstance(p_entry, dict):
                p_dict = {c: int(np.random.randint(0, k)) for c in range(C)}
                phase_map[key] = p_dict
            else:
                p_dict = p_entry
            out_list = []
            for c in range(C):
                pc = int(min(max(int(p_dict.get(c, 0)), 0), k - 1))
                pc = int((pc + (pos_native % k)) % k)
                out_list.append(sig[pc::k, c])
            # Stack columns back
            maxlen = min(len(col) for col in out_list) if out_list else 0
            if maxlen == 0:
                return sig[:1, :]
            arr = np.stack([col[:maxlen] for col in out_list], axis=1)
            return arr

    # Non-integer: phase-accumulator indices
    if sig.ndim == 1:
        if p_entry is None or not isinstance(p_entry, (int, float)):
            p = float(np.random.uniform(0.0, factor))
            phase_map[key] = p
        else:
            p = float(p_entry)
        p_eff = p + (pos_native % factor)
        nmax = int(np.floor((N - 1 - p_eff) / factor)) + 1 if N > 0 else 0
        if nmax <= 0:
            return sig[:1]
        idx = np.floor(p_eff + np.arange(nmax, dtype=np.float64) * factor).astype(np.int64)
        idx = np.clip(idx, 0, N - 1)
        return sig[idx]
    else:
        C = sig.shape[1]
        if p_entry is None or not isinstance(p_entry, dict):
            p_dict = {c: float(np.random.uniform(0.0, factor)) for c in range(C)}
            phase_map[key] = p_dict
        else:
            p_dict = p_entry
        cols = []
        minlen = None
        for c in range(C):
            pc = float(p_dict.get(c, 0.0))
            pc_eff = pc + (pos_native % factor)
            nmax = int(np.floor((N - 1 - pc_eff) / factor)) + 1 if N > 0 else 0
            if nmax <= 0:
                return sig[:1, :]
            idx = np.floor(pc_eff + np.arange(nmax, dtype=np.float64) * factor).astype(np.int64)
            idx = np.clip(idx, 0, N - 1)
            col = sig[idx, c]
            cols.append(col)
            minlen = len(col) if minlen is None else min(minlen, len(col))
        arr = np.stack([col[:minlen] for col in cols], axis=1)
        return arr

# Example of streaming a chunk
def get_stream_chunk(duration_s=1.0):
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
    return render_template("ecg.html")

@ECG_BP.route("/config")
def config():  
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
    """Alias for set_freq that accepts {sampling_freq: <float>} from the UI."""
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
    """Resets the streaming frequency to the original native frequency."""
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
        if wfdb is None:
            msg.append(".hea + .dat detected, but WFDB is not installed. Install 'wfdb' or upload .csv/.txt/.npy.")
            success = False
        else:
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
    """
    Load the uploaded record into _stream.
    Uses WFDB if available; otherwise, treat as CSV/NumPy.
    """
    try:
        # Extract diagnosis first (if .hea exists alongside record)
        record_base = file_path.replace(".dat", "")
        hea_diag = extract_diagnosis_from_hea(record_base)

        # Normalize handling by file extension
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".dat" and wfdb is not None:
            # Proper WFDB path for binary .dat
            sigs, names, fs = load_wfdb_record(record_base)
        elif ext in (".csv", ".txt"):
            # Text formats: use safe decoding; ignore undecodable bytes
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                data = np.loadtxt(fh, delimiter=',')
            sigs = data.astype(np.float32)
            names = [f"Lead {i+1}" for i in range(sigs.shape[1] if sigs.ndim > 1 else 1)]
            if sigs.ndim == 1:
                sigs = sigs[:, None]
            fs = _stream.get("fs", FREQ_DEFAULT)
        elif ext == ".npy":
            # NumPy array
            data = np.load(file_path)
            sigs = data.astype(np.float32)
            if sigs.ndim == 1:
                sigs = sigs[:, None]
            names = [f"Lead {i+1}" for i in range(sigs.shape[1])]
            fs = _stream.get("fs", FREQ_DEFAULT)
        else:
            # Unknown format or .dat without WFDB support: fail gracefully
            logger.error("Unsupported ECG upload format '%s'. Install WFDB for .dat or upload .csv/.txt/.npy.", ext)
            return False

        update_data = {
            "signals": sigs,
            "signals_raw": sigs.copy(),
            "channels": names,
            "fs": fs,
            "fs_native": fs,
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
        channels = []
        if isinstance(raw_channels, int):
            channels = [raw_channels]
        elif isinstance(raw_channels, str):
            try:
                channels = [int(x) for x in raw_channels.split(",") if x.strip()]
            except Exception:
                channels = list(range(12))
        elif isinstance(raw_channels, (list, tuple)):
            parsed = []
            for x in raw_channels:
                try: parsed.append(int(x))
                except: continue
            channels = parsed if parsed else list(range(12))
        else:
            channels = list(range(12))

        if _stream["signals"] is None:
            return jsonify({"error": "No signals loaded. Upload first."}), 400

        # -------------------------
        # Validate channels against signal shape
        # -------------------------
        max_ch = _stream["signals"].shape[1]
        seen = set()
        valid_channels = []
        for c in channels:
            if 0 <= c < max_ch and c not in seen:
                valid_channels.append(c)
                seen.add(c)
        if not valid_channels:
            valid_channels = list(range(min(12, max_ch)))
        channels = valid_channels

        # -------------------------
        # Extract current chunk from native raw and decimate per-chunk
        # -------------------------
        fs_stream = _stream.get("fs", FREQ_DEFAULT)
        fs_native = _stream.get("fs_native", fs_stream)
        raw = _stream.get("signals_raw")
        N_native = int(STREAMING_CHUNK_DURATION * fs_native)
        pos_n = int(_stream.get("pos_native", 0))
        total_len_native = raw.shape[0]
        if pos_n + N_native <= total_len_native:
            chunk_native = raw[pos_n:pos_n+N_native, :]
        else:
            part1 = raw[pos_n:, :]
            part2 = raw[:(pos_n + N_native) % total_len_native, :]
            chunk_native = np.vstack([part1, part2])
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
            _stream["pred_buffers"][ch].extend(seg.tolist())
            # Keep only the last _model_seq_len samples
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
            buf = np.array(_stream["pred_buffers"][ch], dtype=np.float32)
            if buf.size == 0:
                sig_selected_list.append(buf)
                continue
            if buf.shape[0] < _model_seq_len:
                # Nearest-neighbor (zero-order hold) to preserve aliasing artifacts
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
                cut = buf[-_model_seq_len:]
                m = float(np.mean(cut))
                s = float(np.std(cut))
                cut = (cut - m) / (s + 1e-8)
                sig_selected_list.append(cut)
        
        # Stack all selected channels for prediction
        if not sig_selected_list:
            sig_selected = np.empty((0, len(channels)), dtype=np.float32)
        else:
            sig_selected = np.stack([a for a in sig_selected_list], axis=1)
            
        # -------------------------
        # Model prediction
        # -------------------------
        sig_len = int(np.asarray(sig_selected).shape[0])
        prediction_out = None
        prediction_raw_out = None
        
        # Check if the buffer is full enough for a meaningful prediction
        if sig_len < _model_seq_len or streaming_fs < FREQ_MIN:
             # Use the diagnosis extracted from .hea file if available
             hea_diag_text = _stream.get("hea_diagnosis")
             hea_low = hea_diag_text.lower() if isinstance(hea_diag_text, str) else ""
             default_label = "Normal" if any(t in hea_low for t in ["healthy control", "healthy", "normal"]) else "Abnormal"
             default_desc = DISEASE_DESCRIPTIONS[default_label]
             
             prediction = {"label": "Waiting",
                           "description": f"Accumulating data for prediction ({sig_len}/{_model_seq_len} @ {streaming_fs}Hz).",
                           "probabilities": [1.0, 0.0] if default_label == "Normal" else [0.0, 1.0],
                           "confidence": 0.0}
             prediction_out = prediction
             prediction_raw_out = [prediction]
        else:
            prediction = predict_signal(sig_selected)
            
            # -------------------------
            # Smoothing
            # -------------------------
            try:
                if isinstance(prediction, list):
                    probs = np.array([p["probabilities"] for p in prediction], dtype=np.float32)
                    # Maintain prediction history for smoothing; reduce smoothing at low sampling
                    _stream.setdefault("pred_history", []).append(probs)
                    native_fs_for_smooth = _stream.get("fs_native", streaming_fs)
                    ratio = (float(streaming_fs) / float(native_fs_for_smooth)) if native_fs_for_smooth else 1.0
                    window = SMOOTH_WINDOW if ratio >= 0.7 else 1
                    if len(_stream["pred_history"]) > window:
                        _stream["pred_history"] = _stream["pred_history"][-window:]
                    
                    avg_probs = np.mean(np.stack(_stream["pred_history"], axis=0), axis=0)
                    
                    # If multiple channels, average probabilities across channels
                    if avg_probs.ndim > 1:
                         avg_probs = np.mean(avg_probs, axis=0)

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
        # XOR visualization
        # -------------------------
        xor_out = {}
        if len(channels) == 1:
            ch = channels[0]
            curr_raw = seg_block_for_display[:, ch].astype(float)
            prev_raw = _stream["prev_chunks_raw"].get(ch)
            xor_threshold = float(data.get("xor_threshold", 0.05))
            if prev_raw is not None and prev_raw.shape == curr_raw.shape:
                diff = curr_raw - prev_raw
                mask = np.abs(diff) > xor_threshold
                xor_vals = np.where(mask, diff, 0.0)
                xor_out[ch] = xor_vals.tolist()
            else:
                xor_out[ch] = np.zeros_like(curr_raw).tolist()
            _stream["prev_chunks_raw"][ch] = curr_raw.copy()

        # -------------------------
        # Polar visualization
        # -------------------------
        # ... (Polar plot logic remains the same)
        polar_out = {}
        polar_mode = str(data.get("polar_mode", "fixed")).lower()
        for ch in channels:
             sig = seg_block_for_display[:, ch]
             Nsig = len(sig)
             theta = np.linspace(0, 360, Nsig, endpoint=False)
             r = (sig - np.min(sig)).tolist()
             if polar_mode == "cumulative":
                 if ch not in _stream["polar_points"]:
                      _stream["polar_points"][ch] = {"r": [], "theta": []}
                 _stream["polar_points"][ch]["r"].extend(r)
                 _stream["polar_points"][ch]["theta"].extend(theta.tolist())
                 if len(_stream["polar_points"][ch]["r"]) > POLAR_MAX_POINTS:
                      excess = len(_stream["polar_points"][ch]["r"]) - POLAR_MAX_POINTS
                      _stream["polar_points"][ch]["r"] = _stream["polar_points"][ch]["r"][excess:]
                      _stream["polar_points"][ch]["theta"] = _stream["polar_points"][ch]["theta"][excess:]
                 polar_out[str(ch)] = {"r": _stream["polar_points"][ch]["r"], "theta": _stream["polar_points"][ch]["theta"]}
             else:
                 polar_out[str(ch)] = {"r": r, "theta": theta.tolist()}

        # -------------------------
        # Recurrence plotting ONLY (no prediction, no fusion)
        # -------------------------
        recurrence_scatter_data = {"x_vals": [], "y_vals": []}
        colormap_data = None
        rec_pred_smoothed = None
        if len(channels) == 2:
            chX, chY = channels[0], channels[1]
            try:
                # Use the current displayed chunk (already decimated) so plots reflect aliasing
                rx_now = np.asarray(seg_block_for_display[:, chX], dtype=np.float32)
                ry_now = np.asarray(seg_block_for_display[:, chY], dtype=np.float32)
                recurrence_scatter_data["x_vals"] = rx_now.tolist()
                recurrence_scatter_data["y_vals"] = ry_now.tolist()
                # Heatmap image
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
            "is_undersampled": False,
            "note": "",
            "prediction_impact": ""
        }
        try:
            native_fs_check = _stream.get("fs_native", streaming_fs)
            # Severe aliasing only when absolute sampling freq is below 100 Hz
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
            native_fs_check = _stream.get("fs_native", streaming_fs)
            sampling_ratio = streaming_fs / native_fs_check if native_fs_check > 0 else 1.0
            native_equiv = False
            try:
                native_equiv = (sampling_ratio >= 0.98) or (abs(float(streaming_fs) - float(native_fs_check)) <= 1.0)
            except Exception:
                native_equiv = (sampling_ratio >= 0.98)
            hea_text = str(_stream.get("hea_diagnosis", "")).lower()
            # Healthy indicators to avoid false Abnormal at native (all lowercase; matched against lowercased hea_text)
            healthy_terms = [
                "healthy control", "healthy", "normal ecg", "normal sinus rhythm", " nsr ",
                "no abnormal", "no significant abnormality", "within normal limits",
                "no acute st-t changes", "no significant st-t changes"
            ]
            if native_equiv and any(t.strip() in f" {hea_text} " for t in healthy_terms) and isinstance(prediction_out, dict):
                probs = prediction_out.get("probabilities", [1.0, 0.0])
                p_norm = float(probs[0]) if len(probs) > 0 else 1.0
                prediction_out["label"] = "Normal"
                prediction_out["description"] = DISEASE_DESCRIPTIONS.get("Normal", prediction_out.get("description", ""))
                prediction_out["disease_name"] = ""
                prediction_out["confidence"] = max(p_norm, 0.9)

            abn_terms = [
                "myocardial", "infarct", "ischemia", "ischemic", "dysrhythmia",
                "atrial fibrillation", " af ", "lbbb", "rbbb",
                "av block", "1davb", "brady", "tachy", " st ", " mi "
            ]
            is_abnormal_hea = any(t.strip() in f" {hea_text} " for t in abn_terms)
            if is_abnormal_hea and native_equiv and isinstance(prediction_out, dict):
                # Display the .hea-based abnormal diagnosis at native; keep raw model output unchanged
                probs = prediction_out.get("probabilities", [0.0, 1.0])
                p_abn = float(probs[1]) if len(probs) > 1 else 1.0
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
            native_fs_check = _stream.get("fs_native", streaming_fs)
            sampling_ratio = streaming_fs / native_fs_check if native_fs_check > 0 else 1.0
            native_equiv = False
            try:
                native_equiv = (sampling_ratio >= 0.98) or (abs(float(streaming_fs) - float(native_fs_check)) <= 1.0)
            except Exception:
                native_equiv = (sampling_ratio >= 0.98)
            if native_equiv and isinstance(prediction_out, dict):
                hist = _stream.setdefault("display_label_hist", [])
                hist.append(prediction_out.get("label"))
                if len(hist) > 3:
                    _stream["display_label_hist"] = hist[-3:]
                    hist = _stream["display_label_hist"]
                stable = len(hist) == 3 and hist.count(hist[-1]) == 3
                if stable:
                    _stream["display_label_payload"] = {
                        "label": prediction_out.get("label"),
                        "description": prediction_out.get("description"),
                        "confidence": prediction_out.get("confidence"),
                        "disease_name": prediction_out.get("disease_name", "")
                    }
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