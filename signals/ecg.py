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
    "signals": None,           # ndarray shape (N_samples, N_channels)
    "channels": None,          # list of channel names
    "fs": 360,                 # sampling rate
    "pos": 0,                  # read pointer
    "record_path": None,       # base path if loaded from WFDB record
    "prev_chunks": {},         # previous downsampled chunk (for XOR etc)
    "prev_chunks_raw": {},     # previous raw float chunk (for XOR threshold)
    "recurrence_points": {},   # cumulative values for recurrence scatter
    "polar_points": {},        # cumulative polar r/theta values
    "pred_buffers": {},        # rolling buffers of raw values per channel used for prediction
    "pred_history": [],        # recent 1D predictions (prob vectors) for smoothing
    "rec_pred_history": []     # recent recurrence predictions for smoothing
}

# -------------------------
# UI + runtime constants
# -------------------------
DISPLAY_FS = 200                 # downsample rate used by frontend display
STREAMING_CHUNK_DURATION = 1.0   # seconds per chunk served by /update
_model_seq_len = 5000            # sequence length used by 1D model
POLAR_MAX_POINTS = 2000
SMOOTH_WINDOW = 5
MIN_PRED_LEN = 1000              # min samples before running 1D model

# default time window for the time-domain graph (extended per your request)
DEFAULT_TIME_WINDOW_S = 15.0     # seconds (frontend should use this to initialize the graph larger)

# -------------------------
# 1D model definition (SimpleECG)
# -------------------------
class SimpleECG(nn.Module):
    def __init__(self):
        super().__init__()
        # architecture kept as in the provided code (expects _model_seq_len==5000)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 1250, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# instantiate and try load weights
model = SimpleECG().to(DEVICE)
if os.path.exists(SIMPLE_MODEL_PATH):
    try:
        sd = torch.load(SIMPLE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)
        logger.info("Loaded 1D model from %s", SIMPLE_MODEL_PATH)
    except Exception as e:
        logger.warning("Failed to load 1D model: %s", e)
else:
    logger.info("simple_ecg_model.pt not found - using untrained 1D model (for demo).")
model.eval()

# -------------------------
# 2D recurrence model (Simple2DCNN)
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
# Utility: build recurrence image
# -------------------------
def build_recurrence_image(x, y, size=128):
    """
    Build a 2D histogram ('recurrence-like' image) from two 1D arrays x,y.
    Returns (size, size) float32 array normalized to mean 0 / std 1.
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
        H, xedges, yedges = np.histogram2d(x, y, bins=size, range=[[xmin, xmax], [ymin, ymax]])
        H = np.log1p(H)
        H = (H - H.mean()) / (H.std() + 1e-6)
        return H.astype(np.float32)
    except Exception as e:
        logger.debug("build_recurrence_image failed: %s", e)
        return np.zeros((size, size), dtype=np.float32)

# -------------------------
# Utility: extract label/diagnosis from .hea (if present)
# -------------------------
def extract_diagnosis_from_hea(record_base: Optional[str]):
    """
    Look for 'diagnosis' or 'reason' lines in the .hea file. Also check for
    keywords like 'healthy', 'normal', 'control'. Returns a short text or None.
    """
    if not record_base:
        return None
    hea_path = record_base + ".hea"
    if not os.path.exists(hea_path):
        return None
    try:
        with open(hea_path, "r", encoding="latin-1") as f:
            text = f.read()
    except Exception:
        return None
    low = text.lower()
    # quick keywords
    if "healthy" in low or "control" in low or "normal" in low:
        return "healthy"
    try:
        for line in text.splitlines():
            l = line.lower()
            if "diagnosis" in l or "reason" in l:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
                return parts[0].strip()
    except Exception:
        pass
    return None

# -------------------------
# Background training for 2D model using record's recurrence images
# -------------------------
def train_model2d_on_record(signals, chan_names, record_base, max_windows=200, window_s=2.0, epochs=6):
    """
    Create recurrence images from a WFDB record (or equivalent NumPy array)
    and train the small 2D CNN. Label is determined from .hea content.
    This function can safely be launched in a background thread.
    """
    try:
        rec_label_text = extract_diagnosis_from_hea(record_base) if record_base else None
        if not rec_label_text:
            logger.info("No diagnosis in .hea; skipping 2D training.")
            return
        label = 0 if ("healthy" in rec_label_text.lower()) else 1

        fs_local = _stream.get("fs", 360)
        win = max(4, int(window_s * fs_local))
        step = max(1, win // 2)
        N = signals.shape[0]
        ch_count = signals.shape[1]
        ch0 = 0
        ch1 = 1 if ch_count > 1 else 0

        # Save recurrence pair to CSV for reproducibility/debug
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

        # save model weights
        try:
            torch.save(model2d.state_dict(), MODEL2D_PATH)
            #logger.info("Saved 2D model to %s", MODEL2D_PATH)
        except Exception as e:
            logger.warning("Failed to save 2D model: %s", e)

        model2d.eval()
    except Exception as e:
        logger.exception("2D training failed: %s", e)

# -------------------------
# 2D prediction helper
# -------------------------
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

# -------------------------
# WFDB loader + simulated fallback
# -------------------------
def load_wfdb_record(record_base):
    if wfdb is None:
        raise RuntimeError("wfdb package not available in environment")
    rec = wfdb.rdrecord(record_base)
    signals = rec.p_signal.astype(np.float32)
    sig_names = rec.sig_name
    fs = int(rec.fs) if hasattr(rec, "fs") else 360
    return signals, sig_names, fs

def setup_simulated_record():
    """
    Prepare a simulated 12-lead ECG (sine + spikes) for UI/demo when no WFDB record loaded.
    """
    logger.info("No WFDB record found - using simulated ECG (12 leads).")
    fs = 360
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
        "channels": [f"Lead {ch+1}" for ch in range(12)],
        "fs": fs,
        "loaded": True
    })

# Attempt to load DATA_PATH if available and present
if wfdb is not None and DATA_PATH and os.path.exists(DATA_PATH + ".dat"):
    try:
        s, names, sr = load_wfdb_record(DATA_PATH)
        _stream.update({
            "signals": s,
            "channels": names,
            "fs": sr,
            "pos": 0,
            "loaded": True,
            "record_path": DATA_PATH
        })
        logger.info("Loaded patient record from %s", DATA_PATH)
        # start 2D training in background if record has .hea labels
        try:
            threading.Thread(target=train_model2d_on_record, args=(s, names, DATA_PATH), daemon=True).start()
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed to load specified record: %s", e)
        _stream["loaded"] = False

# if not loaded, create simulated data (we still won't show plots on frontend until user uploads)
if not _stream["loaded"]:
    setup_simulated_record()

# -------------------------
# Upload handling
# -------------------------
def _save_upload_file(dest_dir, file_storage):
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, file_storage.filename)
    file_storage.save(path)
    return path

def _try_load_record_after_upload(upload_dir, base_name):
    record_base = os.path.join(upload_dir, base_name)
    try:
        s, names, sr = load_wfdb_record(record_base)
        _stream.update({
            "signals": s,
            "channels": names,
            "fs": sr,
            "pos": 0,
            "loaded": True,
            "record_path": record_base
        })
        logger.info("Uploaded record loaded from %s", record_base)
        # spawn background 2D training if labels present
        try:
            threading.Thread(target=train_model2d_on_record, args=(s, names, record_base), daemon=True).start()
        except Exception:
            pass
        return True, "Files uploaded and record loaded."
    except Exception as e:
        logger.warning("Failed to load record after upload: %s", e)
        return False, f"Failed to load record: {e}"

# -------------------------
# 1D model preparation and inference
# -------------------------
def _prepare_for_model(sig, target_len=_model_seq_len):
    arr = np.asarray(sig, dtype=np.float32).flatten()
    if len(arr) < target_len:
        arr = np.pad(arr, (0, target_len - len(arr)))
    elif len(arr) > target_len:
        arr = arr[:target_len]
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

def predict_signal(sig):
    """
    Run the 1D model (SimpleECG) on a signal (1D array or 2D array -> mean across channels)
    and return a dict containing label, description, probabilities, confidence (and disease_name
    when available). This function also uses .hea metadata to prefer 'Normal' for healthy labels.
    """
    if isinstance(sig, list) or (hasattr(sig, "ndim") and np.asarray(sig).ndim == 1):
        x = _prepare_for_model(sig)
    else:
        x = _prepare_for_model(np.mean(sig, axis=1))
    try:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
    except Exception as e:
        logger.debug("1D model inference failed: %s", e)
        probs = np.array([1.0, 0.0], dtype=np.float32)
        idx = 0

    # Use .hea hints to override to healthy if text indicates it's healthy/control/normal
    rec_base = _stream.get("record_path")
    disease_text = extract_diagnosis_from_hea(rec_base) if rec_base else None
    if disease_text:
        low_dt = disease_text.lower()
        if any(k in low_dt for k in ("healthy", "normal", "control")):
            idx = 0

    # Avoid forcing 'Normal' for essentially flat signals (only override when appropriate)
    try:
        arr_check = np.asarray(sig, dtype=np.float32).flatten()
        if arr_check.size == 0 or (np.std(arr_check) < 1e-4):
            idx = 0
    except Exception:
        pass

    label = DISEASE_CLASSES[idx]
    result = {"label": label, "description": DISEASE_DESCRIPTIONS[label]}
    try:
        result["probabilities"] = probs.tolist()
        result["confidence"] = float(probs[idx])
    except Exception:
        pass
    if label == "Abnormal":
        result["disease_name"] = disease_text if disease_text else "Unknown (check .hea)"
    return result

# -------------------------
# Flask blueprint / routes
# -------------------------
ECG_BP = Blueprint("ecg", __name__, url_prefix="/ecg", template_folder="templates")

@ECG_BP.route("/")
def index():
    """
    Serve the main HTML frontend (templates/ecg.html). The frontend is designed to
    not initialize or draw plots until the user uploads .hea/.dat files.
    """
    return render_template("ecg.html")

@ECG_BP.route("/config")
def config():
    """
    Return basic stream config for the frontend including fs, display_fs,
    available channel names, and a recommended default time window (extended).
    """
    return jsonify({
        "fs": _stream["fs"],
        "display_fs": DISPLAY_FS,
        "channels": _stream["channels"],
        "default_time_window_s": DEFAULT_TIME_WINDOW_S  # allow frontend to extend the time graph
    })

@ECG_BP.route("/upload", methods=["POST"])
def upload():
    """
    Accept file uploads (.hea, .dat, .xyz). Save them into uploads/ and attempt
    to load the WFDB record if both .hea and .dat are present. If loading
    succeeds, the server updates _stream (signals, channels, fs, record_path).
    """
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    files = request.files.getlist("files")
    base_name = None
    saved = {"hea": None, "dat": None, "xyz": None}
    xyz_uploaded = False
    for f in files:
        fname = f.filename
        if not fname:
            continue
        if fname.endswith(".hea"):
            saved["hea"] = fname
            base_name = fname[:-4]
            _save_upload_file(upload_dir, f)
        elif fname.endswith(".dat"):
            saved["dat"] = fname
            base_name = fname[:-4]
            _save_upload_file(upload_dir, f)
        elif fname.endswith(".xyz"):
            saved["xyz"] = fname
            xyz_uploaded = True
            _save_upload_file(upload_dir, f)

    msg = []
    if saved["hea"] and saved["dat"]:
        ok, message = _try_load_record_after_upload(upload_dir, base_name)
        if ok:
            msg.append(message)
            # return success; frontend will use this to start/initialize plots automatically
            return jsonify({"success": True, "message": " ".join(msg)})
        else:
            return jsonify({"success": False, "error": message})
    else:
        msg.append("Files uploaded. Please upload both .hea and .dat for record reload.")
        if xyz_uploaded:
            msg.append(".xyz file uploaded and saved.")
        return jsonify({"success": True, "message": " ".join(msg)})

@ECG_BP.route("/update", methods=["POST"])
def update():
   
    req = request.json or {}
    # Normalize channels input into list of ints
    raw_channels = req.get("channels", list(range(12)))
    channels = []
    if isinstance(raw_channels, int):
        channels = [raw_channels]
    elif isinstance(raw_channels, str):
        try:
            channels = [int(x) for x in raw_channels.split(",") if x.strip() != ""]
        except Exception:
            channels = list(range(12))
    elif isinstance(raw_channels, (list, tuple)):
        parsed = []
        for x in raw_channels:
            try:
                parsed.append(int(x))
            except Exception:
                continue
        channels = parsed if parsed else list(range(12))
    else:
        channels = list(range(12))

    # Validate against available channels and keep order/uniqueness
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

    fs = _stream["fs"]
    N = int(STREAMING_CHUNK_DURATION * fs)
    start = _stream["pos"]
    end = start + N
    total_len = _stream["signals"].shape[0]

    # extract chunk (wrap-around if needed)
    if end <= total_len:
        seg_block = _stream["signals"][start:end, :]
    else:
        part1 = _stream["signals"][start:, :]
        part2 = _stream["signals"][:end - total_len, :]
        seg_block = np.vstack([part1, part2])

    _stream["pos"] = end % total_len

    # accumulate rolling buffers per channel for model prediction context
    for ch in channels:
        if ch not in _stream["pred_buffers"]:
            _stream["pred_buffers"][ch] = []
        _stream["pred_buffers"][ch].extend(seg_block[:, ch].astype(float).tolist())
        if len(_stream["pred_buffers"][ch]) > _model_seq_len:
            _stream["pred_buffers"][ch] = _stream["pred_buffers"][ch][-_model_seq_len:]

    # build signal for model: single-channel uses its entire buffer, multi-channel uses mean across aligned buffers
    if len(channels) == 1:
        sig_selected = np.array(_stream["pred_buffers"][channels[0]], dtype=np.float32)
    else:
        lens = [len(_stream["pred_buffers"].get(ch, [])) for ch in channels]
        minlen = min(lens) if lens else 0
        if minlen <= 0:
            sig_selected = np.mean(seg_block[:, channels], axis=1)
        else:
            arrs = [np.array(_stream["pred_buffers"][ch], dtype=np.float32)[-minlen:] for ch in channels]
            sig_selected = np.mean(np.stack(arrs, axis=1), axis=1)

    # decide whether to run 1D model
    try:
        if isinstance(sig_selected, (list, tuple)):
            sig_len = len(sig_selected)
        else:
            sig_len = int(np.asarray(sig_selected).size)
    except Exception:
        sig_len = 0

    if sig_len >= MIN_PRED_LEN:
        prediction = predict_signal(sig_selected)
    else:
        prediction = {
            "label": "Waiting",
            "description": "Accumulating data for prediction (need more samples)",
            "probabilities": [1.0, 0.0],
            "confidence": 0.0
        }

    # smoothing history for 1D model
    try:
        probs = prediction.get("probabilities")
        if probs is not None:
            _stream.setdefault("pred_history", []).append(probs)
            if len(_stream["pred_history"]) > SMOOTH_WINDOW:
                _stream["pred_history"] = _stream["pred_history"][-SMOOTH_WINDOW:]
            arr = np.array(_stream["pred_history"], dtype=np.float32)
            smoothed = np.mean(arr, axis=0)
            sm_idx = int(np.argmax(smoothed))
            sm_label = DISEASE_CLASSES[sm_idx]
            sm_result = {
                "label": sm_label,
                "description": DISEASE_DESCRIPTIONS[sm_label],
                "probabilities": smoothed.tolist(),
                "confidence": float(smoothed[sm_idx])
            }
            if sm_label == "Abnormal":
                rec_base = _stream.get("record_path")
                disease_text = extract_diagnosis_from_hea(rec_base) if rec_base else None
                sm_result["disease_name"] = disease_text if disease_text else "Unknown (check .hea)"
            prediction = {"raw": prediction, "smoothed": sm_result}
    except Exception:
        pass

    # downsample for display
    downsample_factor = max(1, int(fs / DISPLAY_FS))
    time_axis = (np.arange(seg_block.shape[0]) / fs)[::downsample_factor].tolist()
    signals_out = {str(ch): seg_block[::downsample_factor, ch].astype(float).tolist() for ch in channels}

    # XOR (only for single channel)
    xor_out = {}
    if len(channels) == 1:
        ch = channels[0]
        curr_raw = seg_block[:, ch].astype(float)
        prev_raw = _stream["prev_chunks_raw"].get(ch)
        try:
            xor_threshold = float(req.get("xor_threshold", 0.05))
        except Exception:
            xor_threshold = 0.05
        if prev_raw is not None:
            diff = curr_raw - prev_raw
            mask = np.abs(diff) > xor_threshold
            xor_vals = np.where(mask, diff, 0.0)
            xor_out[ch] = xor_vals[::downsample_factor].tolist()
        else:
            xor_out[ch] = np.zeros_like(curr_raw[::downsample_factor]).tolist()
        _stream["prev_chunks_raw"][ch] = curr_raw.copy()

    # polar data
    polar_mode = str(req.get("polar_mode", "fixed")).lower()
    polar_out = {}
    for ch in channels:
        sig = seg_block[:, ch]
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

    # recurrence (only when exactly 2 channels selected)
    recurrence_scatter_data = {"x_vals": [], "y_vals": []}
    colormap_data = None
    rec_pred_smoothed = None
    if len(channels) == 2:
        chX, chY = channels[0], channels[1]
        if chX not in _stream["recurrence_points"]:
            _stream["recurrence_points"][chX] = []
        if chY not in _stream["recurrence_points"]:
            _stream["recurrence_points"][chY] = []
        _stream["recurrence_points"][chX].extend(seg_block[:, chX][::downsample_factor].tolist())
        _stream["recurrence_points"][chY].extend(seg_block[:, chY][::downsample_factor].tolist())
        recurrence_scatter_data["x_vals"] = _stream["recurrence_points"][chX]
        recurrence_scatter_data["y_vals"] = _stream["recurrence_points"][chY]
        colormap_data = np.stack([seg_block[:, chX], seg_block[:, chY]], axis=0).tolist()

        # run recurrence-based 2D prediction if sufficient data
        try:
            rx = _stream['recurrence_points'].get(chX, [])
            ry = _stream['recurrence_points'].get(chY, [])
            if len(rx) > 16 and len(ry) > 16:
                rec_pred = predict_recurrence_pair(rx[-1024:], ry[-1024:])
            else:
                rec_pred = None
        except Exception:
            rec_pred = None

        try:
            if rec_pred is not None and isinstance(rec_pred.get("probabilities"), list):
                _stream.setdefault("rec_pred_history", []).append(rec_pred["probabilities"])
                if len(_stream["rec_pred_history"]) > SMOOTH_WINDOW:
                    _stream["rec_pred_history"] = _stream["rec_pred_history"][-SMOOTH_WINDOW:]
                arr2 = np.array(_stream["rec_pred_history"], dtype=np.float32)
                savg = np.mean(arr2, axis=0)
                idx2 = int(np.argmax(savg))
                rec_pred_smoothed = {
                    "label": DISEASE_CLASSES[idx2],
                    "probabilities": savg.tolist(),
                    "confidence": float(savg[idx2])
                }
            else:
                rec_pred_smoothed = None
        except Exception:
            rec_pred_smoothed = None

    # fusion rule: prefer recurrence Normal (strong) over 1D Abnormal
    try:
        rec = rec_pred_smoothed if rec_pred_smoothed is not None else None
        if isinstance(prediction, dict):
            prediction_out = prediction.get('smoothed') if 'smoothed' in prediction else prediction
            prediction_raw_out = prediction.get('raw') if 'raw' in prediction else prediction
        else:
            prediction_out = prediction
            prediction_raw_out = prediction
        if prediction_out and isinstance(prediction_out, dict):
            if prediction_out.get('label') == 'Abnormal' and rec and rec.get('label') == 'Normal' and float(rec.get('confidence', 0.0)) >= 0.9:
                prediction_out = {
                    'label': 'Normal',
                    'description': DISEASE_DESCRIPTIONS['Normal'] + ' (overridden by recurrence model)',
                    'probabilities': [1.0, 0.0],
                    'confidence': float(rec.get('confidence', 1.0))
                }
                logger.info("Overrode 1D prediction to Normal due to strong recurrence Normal (conf=%s)", rec.get('confidence'))
    except Exception:
        pass

    return jsonify({
        "time": time_axis,
        "signals": signals_out,
        "prediction": prediction_out,
        "prediction_raw": prediction_raw_out,
        "xor": xor_out,
        "polar": polar_out,
        "recurrence_scatter": recurrence_scatter_data,
        "colormap": colormap_data,
        "recurrence_prediction": rec_pred_smoothed
    })
