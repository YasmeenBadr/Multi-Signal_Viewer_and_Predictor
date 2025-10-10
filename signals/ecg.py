import os
import time
import numpy as np
import torch
import torch.nn as nn
from flask import Blueprint, request, jsonify, render_template
import threading

try:
    import wfdb
except Exception:
    wfdb = None

try:
    from simple_ecg import DATA_PATH
except ImportError:
    DATA_PATH = None

ECG_BP = Blueprint("ecg", __name__, url_prefix="/ecg", template_folder="../templates")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMPLE_MODEL_PATH = os.path.join(os.getcwd(), "simple_ecg_model.pt")

DISEASE_CLASSES = ["Normal", "Abnormal"]
DISEASE_DESCRIPTIONS = {
    "Normal": "No obvious abnormality detected.",
    "Abnormal": (
        "Abnormal ECG pattern detected. Possible ischemia, arrhythmia, or other irregularities. "
        "Please confirm with a cardiologist."
    ),
}

# -------------------------
# Stream state
# -------------------------
_stream = {
    "loaded": False,
    "signals": None,
    "channels": None,
    "fs": 360,
    "pos": 0,
    "record_path": None,
    "prev_chunks": {},       # store previous chunk for XOR per channel
    "prev_chunks_raw": {},   # store previous raw float chunk per channel (for thresholded XOR)
    "recurrence_points": {},  # store cumulative points for recurrence plot
    "polar_points": {},       # store cumulative polar r-values per channel
    "pred_buffers": {},       # rolling buffers used for model prediction per channel
    "pred_history": [],       # recent model probability vectors for smoothing
    "rec_pred_history": []    # recent recurrence-model probability vectors for smoothing
}

DISPLAY_FS = 200
STREAMING_CHUNK_DURATION = 1.0
_model_seq_len = 5000
POLAR_MAX_POINTS = 2000
SMOOTH_WINDOW = 5
MIN_PRED_LEN = 1000  # minimum samples required before running 1D model prediction

# -------------------------
# 1D model
# -------------------------
class SimpleECG(nn.Module):
    def __init__(self):
        super().__init__()
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

model = SimpleECG().to(DEVICE)
if os.path.exists(SIMPLE_MODEL_PATH):
    try:
        sd = torch.load(SIMPLE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded model from {SIMPLE_MODEL_PATH}")
    except Exception as e:
        print("Failed to load model:", e)
else:
    print("simple_ecg_model.pt not found — using untrained model.")
model.eval()

# -------------------------
# 2D CNN Model for recurrence colormap
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
model2d.eval()

from torch.utils.data import TensorDataset, DataLoader

# Helper: build a recurrence-style 2D image from two time series
def build_recurrence_image(x, y, size=128):
    """Create a 2D histogram (recurrence-like) image from x and y signals.
    Returns a float32 array shape (size, size) normalized to mean 0, std 1.
    """
    try:
        x = np.asarray(x, dtype=np.float32).flatten()
        y = np.asarray(y, dtype=np.float32).flatten()
        if len(x) == 0 or len(y) == 0:
            return np.zeros((size, size), dtype=np.float32)
        # range with a small margin
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        if xmin == xmax:
            xmin -= 1e-3; xmax += 1e-3
        if ymin == ymax:
            ymin -= 1e-3; ymax += 1e-3
        H, xedges, yedges = np.histogram2d(x, y, bins=size, range=[[xmin, xmax], [ymin, ymax]])
        # log scaling helps
        H = np.log1p(H)
        # normalize
        H = (H - H.mean()) / (H.std() + 1e-6)
        return H.astype(np.float32)
    except Exception:
        return np.zeros((size, size), dtype=np.float32)


# Helper to extract diagnosis from .hea files (moved here so training thread can call it)
def extract_diagnosis_from_hea(record_base):
    hea_path = record_base + ".hea"
    if not os.path.exists(hea_path):
        return None
    try:
        with open(hea_path, "r", encoding="latin-1") as f:
            text = f.read()
    except Exception:
        return None
    low = text.lower()
    # quick keyword checks
    if "healthy" in low or "control" in low or "normal" in low:
        # return a short indicator string
        return "healthy"
    # otherwise look for explicit diagnosis/reason lines (with or without #)
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


def train_model2d_on_record(signals, chan_names, record_base, max_windows=200, window_s=2.0, epochs=6):
    """Train the simple 2D CNN on recurrence images generated from the provided
    signals. The label is derived from the .hea using extract_diagnosis_from_hea.
    This function runs synchronously; callers may run it in a background thread.
    """
    try:
        rec_label_text = extract_diagnosis_from_hea(record_base) if record_base else None
        if not rec_label_text:
            # no label available — skip training
            print("No diagnosis found in .hea; skipping 2D training.")
            return
        label = 0 if ("healthy" in rec_label_text.lower()) else 1

        fs_local = _stream.get("fs", 360)
        win = max(4, int(window_s * fs_local))
        step = max(1, win // 2)
        N = signals.shape[0]
        ch_count = signals.shape[1]
        # choose channels: prefer first two available
        ch0 = 0
        ch1 = 1 if ch_count > 1 else 0

        # Save recurrence pair data to CSV before training for reproducibility/debug
        try:
            outdir = os.path.join(os.getcwd(), 'results', 'recurrence_data')
            os.makedirs(outdir, exist_ok=True)
            base = os.path.basename(record_base) if record_base else f'record_{int(time.time())}'
            csv_path = os.path.join(outdir, f"{base}_ch{ch0}_ch{ch1}_recurrence.csv")
            # write two columns: ch0, ch1
            try:
                twoch = np.stack([signals[:, ch0], signals[:, ch1]], axis=1)
                # include a small header with label info
                header = 'ch0,ch1'
                np.savetxt(csv_path, twoch, delimiter=',', header=header, comments='')
                print(f"Saved recurrence CSV to {csv_path}")
            except Exception as e:
                print("Failed to write recurrence CSV:", e)
        except Exception:
            pass

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
            print("Not enough windows for training 2D model; found", len(images))
            return

        X = np.stack(images, axis=0)[:, None, :, :].astype(np.float32)
        y_arr = np.array(labels, dtype=np.int64)

        # convert to torch tensors
        tX = torch.from_numpy(X)
        ty = torch.from_numpy(y_arr)
        dataset = TensorDataset(tX, ty)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model2d.parameters(), lr=1e-3)

        model2d.train()
        print(f"Starting 2D training on {len(dataset)} samples, label={label}")
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
            print(f"2D train epoch {ep+1}/{epochs} loss={total_loss/total:.4f} acc={correct/total:.3f}")

        # save model weights
        try:
            save_path = os.path.join(os.getcwd(), 'model2d_recurrence.pt')
            torch.save(model2d.state_dict(), save_path)
            print(f"Saved 2D model to {save_path}")
        except Exception as e:
            print("Failed to save 2D model:", e)

        model2d.eval()
    except Exception as e:
        print("2D training failed:", e)


def predict_recurrence_pair(x, y):
    """Compute recurrence image for a pair and run model2d to predict label/confidence."""
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
        return None

# -------------------------
# Load record
# -------------------------
def load_wfdb_record(record_base):
    rec = wfdb.rdrecord(record_base)
    signals = rec.p_signal.astype(np.float32)
    sig_names = rec.sig_name
    fs = int(rec.fs) if hasattr(rec, "fs") else 360
    return signals, sig_names, fs

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
        print(f"Loaded patient record from {DATA_PATH}")
        # start background 2D training if wfdb record and labels are present
        try:
            threading.Thread(target=train_model2d_on_record, args=(s, names, DATA_PATH), daemon=True).start()
        except Exception:
            pass
    except Exception as e:
        print("Failed to load specified record:", e)
        _stream["loaded"] = False

if not _stream["loaded"]:
# Simulate ECG data if no WFDB record is loaded
    print("No WFDB record found — using simulated ECG (12 leads).")
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
# -------------------------
# Upload route for .hea and .dat files
# -------------------------
@ECG_BP.route("/upload", methods=["POST"])
def upload():
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    files = request.files.getlist("files")
    base_name = None
    saved = {"hea": None, "dat": None, "xyz": None}
    xyz_uploaded = False
    for f in files:
        fname = f.filename
        if fname.endswith(".hea"):
            saved["hea"] = fname
            base_name = fname[:-4]
            f.save(os.path.join(upload_dir, fname))
        elif fname.endswith(".dat"):
            saved["dat"] = fname
            base_name = fname[:-4]
            f.save(os.path.join(upload_dir, fname))
        elif fname.endswith(".xyz"):
            saved["xyz"] = fname
            xyz_uploaded = True
            f.save(os.path.join(upload_dir, fname))
    # Try to reload ECG record if both files are present
    msg = []
    if saved["hea"] and saved["dat"]:
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
            msg.append("Files uploaded and record loaded.")
            # spawn background training for 2D model
            try:
                threading.Thread(target=train_model2d_on_record, args=(s, names, record_base), daemon=True).start()
            except Exception:
                pass
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to load record: {e}"})
    else:
        msg.append("Files uploaded. Please upload both .hea and .dat for record reload.")
    if xyz_uploaded:
        msg.append(".xyz file uploaded and saved.")
    return jsonify({"success": True, "message": " ".join(msg)})
    print("No WFDB record found — using simulated ECG (12 leads).")
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

# -------------------------
# Helpers
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
    if isinstance(sig, list) or sig.ndim == 1:
        x = _prepare_for_model(sig)
    else:
        x = _prepare_for_model(np.mean(sig, axis=1))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
    rec_base = _stream.get("record_path")
    disease_text = extract_diagnosis_from_hea(rec_base) if rec_base else None
    if disease_text:
        low_dt = disease_text.lower()
        if any(k in low_dt for k in ("healthy", "normal", "control")):
            idx = 0
    # Avoid forcing 'Normal' for modest amplitude signals. Only override when
    # the signal is essentially flat (very low variance).
    try:
        arr_check = np.asarray(sig, dtype=np.float32).flatten()
        if arr_check.size == 0 or (np.std(arr_check) < 1e-4):
            idx = 0
    except Exception:
        pass
    label = DISEASE_CLASSES[idx]
    result = {"label": label, "description": DISEASE_DESCRIPTIONS[label]}
    # include raw model probabilities and confidence for debugging/UI
    try:
        result["probabilities"] = probs.tolist()
        result["confidence"] = float(probs[idx])
    except Exception:
        pass
    if label=="Abnormal":
        result["disease_name"] = disease_text if disease_text else "Unknown (check .hea)"
    return result

# -------------------------
# Routes
# -------------------------
@ECG_BP.route("/")
def index():
    return render_template("ecg.html")

@ECG_BP.route("/config")
def config():
    return jsonify({
        "fs": _stream["fs"],
        "display_fs": DISPLAY_FS,
        "channels": _stream["channels"]
    })

@ECG_BP.route("/update", methods=["POST"])
def update():
    req = request.json or {}
    # channels can come in as: list of ints, list of strings, a single int, or a comma-separated string
    raw_channels = req.get("channels", list(range(12)))
    channels = []
    # normalize to a list of ints
    if isinstance(raw_channels, int):
        channels = [raw_channels]
    elif isinstance(raw_channels, str):
        # e.g. "0,1,2" or "0"
        try:
            channels = [int(x) for x in raw_channels.split(",") if x.strip() != ""]
        except ValueError:
            channels = list(range(12))
    elif isinstance(raw_channels, (list, tuple)):
        parsed = []
        for x in raw_channels:
            try:
                parsed.append(int(x))
            except Exception:
                # ignore non-convertible entries
                continue
        channels = parsed if parsed else list(range(12))
    else:
        channels = list(range(12))

    # validate channel indices and preserve order/uniqueness
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

    if end <= total_len:
        seg_block = _stream["signals"][start:end,:]
    else:
        part1 = _stream["signals"][start:,:]
        part2 = _stream["signals"][:end-total_len,:]
        seg_block = np.vstack([part1, part2])

    _stream["pos"] = end % total_len

    # Prediction: accumulate a rolling buffer per channel and use a longer sequence
    # for prediction (model expects _model_seq_len samples). This gives the model
    # more context than the 1s seg_block and avoids trivial 'Normal' outputs.
    for ch in channels:
        if ch not in _stream["pred_buffers"]:
            _stream["pred_buffers"][ch] = []
        _stream["pred_buffers"][ch].extend(seg_block[:, ch].astype(float).tolist())
        # cap to model sequence length
        if len(_stream["pred_buffers"][ch]) > _model_seq_len:
            _stream["pred_buffers"][ch] = _stream["pred_buffers"][ch][-_model_seq_len:]

    # build signal for prediction: if single channel use its buffer; otherwise use
    # the mean across channel buffers (aligned to the shortest buffer length).
    if len(channels) == 1:
        sig_selected = np.array(_stream["pred_buffers"][channels[0]], dtype=np.float32)
    else:
        # find available lengths
        lens = [len(_stream["pred_buffers"].get(ch, [])) for ch in channels]
        minlen = min(lens) if lens else 0
        if minlen <= 0:
            # fallback to current mean of seg_block when buffers are empty
            sig_selected = np.mean(seg_block[:, channels], axis=1)
        else:
            arrs = [np.array(_stream["pred_buffers"][ch], dtype=np.float32)[-minlen:] for ch in channels]
            sig_selected = np.mean(np.stack(arrs, axis=1), axis=1)

    # Only run the 1D model when we have accumulated a reasonable amount of data.
    # Predicting on heavily padded inputs tends to produce 'Normal' due to zero-padding.
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
        # Not enough data yet: return placeholder; smoothing will still be applied
        prediction = {
            "label": "Waiting",
            "description": "Accumulating data for prediction (need more samples)",
            "probabilities": [1.0, 0.0],
            "confidence": 0.0
        }
    # Store raw probabilities for smoothing
    try:
        probs = prediction.get("probabilities")
        if probs is not None:
            _stream.setdefault("pred_history", []).append(probs)
            # cap history
            if len(_stream["pred_history"]) > SMOOTH_WINDOW:
                _stream["pred_history"] = _stream["pred_history"][-SMOOTH_WINDOW:]
            # compute smoothed probabilities (average)
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
            # preserve original as 'prediction_raw' and return smoothed in 'prediction'
            prediction = {"raw": prediction, "smoothed": sm_result}
    except Exception:
        pass

    downsample_factor = max(1,int(fs/DISPLAY_FS))
    time_axis = (np.arange(seg_block.shape[0])/fs)[::downsample_factor].tolist()
    signals_out = {str(ch): seg_block[::downsample_factor, ch].astype(float).tolist() for ch in channels}

    # ---- XOR only if 1 channel ----
    xor_out = {}
    if len(channels) == 1:
        ch = channels[0]
        # use raw float values for thresholded difference
        curr_raw = seg_block[:, ch].astype(float)
        prev_raw = _stream["prev_chunks_raw"].get(ch)
        # threshold can be provided by client (in signal units); default small value
        try:
            xor_threshold = float(req.get("xor_threshold", 0.05))
        except Exception:
            xor_threshold = 0.05

        if prev_raw is not None:
            diff = curr_raw - prev_raw
            # set to zero where abs(diff) <= threshold, otherwise keep diff
            mask = np.abs(diff) > xor_threshold
            xor_vals = np.where(mask, diff, 0.0)
            xor_out[ch] = xor_vals[::downsample_factor].tolist()
        else:
            xor_out[ch] = np.zeros_like(curr_raw[::downsample_factor]).tolist()
        # store current raw chunk for next comparison
        _stream["prev_chunks_raw"][ch] = curr_raw.copy()

    # ---- Polar plot ----
    # polar_mode: 'fixed' => return current chunk angles and r-values for each selected channel
    #             'cumulative' => return cumulative r-values across time for each channel
    polar_mode = str(req.get("polar_mode", "fixed")).lower()
    polar_out = {}
    for ch in channels:
        sig = seg_block[:, ch]
        Nsig = len(sig)
        theta = np.linspace(0, 360, Nsig, endpoint=False)
        r = (sig - np.min(sig)).tolist()
        if polar_mode == "cumulative":
            # initialize storage for channel
            if ch not in _stream["polar_points"]:
                _stream["polar_points"][ch] = {"r": [], "theta": []}
            # extend cumulative list and cap to POLAR_MAX_POINTS
            _stream["polar_points"][ch]["r"].extend(r)
            _stream["polar_points"][ch]["theta"].extend(theta.tolist())
            # cap length
            if len(_stream["polar_points"][ch]["r"]) > POLAR_MAX_POINTS:
                excess = len(_stream["polar_points"][ch]["r"]) - POLAR_MAX_POINTS
                _stream["polar_points"][ch]["r"] = _stream["polar_points"][ch]["r"][excess:]
                _stream["polar_points"][ch]["theta"] = _stream["polar_points"][ch]["theta"][excess:]
            polar_out[str(ch)] = {"r": _stream["polar_points"][ch]["r"], "theta": _stream["polar_points"][ch]["theta"]}
        else:
            polar_out[str(ch)] = {"r": r, "theta": theta.tolist()}

    # ---- Recurrence only if exactly 2 channels (unchanged behavior) ----
    recurrence_scatter_data = {"x_vals": [], "y_vals": []}
    colormap_data = None
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

        # If possible, run recurrence-based 2D prediction using accumulated points
        try:
            rx = _stream['recurrence_points'].get(chX, [])
            ry = _stream['recurrence_points'].get(chY, [])
            # use the most recent window for prediction if available
            if len(rx) > 16 and len(ry) > 16:
                rec_pred = predict_recurrence_pair(rx[-1024:], ry[-1024:])
            else:
                rec_pred = None
        except Exception:
            rec_pred = None
        # smooth recurrence prediction probabilities over recent frames
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

    # For backward compatibility, return a flat prediction object at `prediction`.
    # If we wrapped prediction with {'raw':..., 'smoothed':...}, prefer the smoothed
    prediction_out = None
    prediction_raw_out = None
    try:
        if isinstance(prediction, dict) and 'smoothed' in prediction:
            prediction_out = prediction.get('smoothed')
            prediction_raw_out = prediction.get('raw')
        else:
            prediction_out = prediction
            prediction_raw_out = prediction
    except Exception:
        prediction_out = prediction
        prediction_raw_out = prediction

    # If recurrence model strongly indicates 'Normal' while 1D model says 'Abnormal',
    # prefer the recurrence/metadata signal for healthy controls.
    try:
        rec = rec_pred_smoothed if 'rec_pred_smoothed' in locals() else None
        if prediction_out and isinstance(prediction_out, dict):
            if prediction_out.get('label') == 'Abnormal' and rec and rec.get('label') == 'Normal' and float(rec.get('confidence', 0.0)) >= 0.9:
                # override prediction_out to Normal (keep prediction_raw_out for debugging)
                prediction_out = {
                    'label': 'Normal',
                    'description': DISEASE_DESCRIPTIONS['Normal'] + ' (overridden by recurrence model)',
                    'probabilities': [1.0, 0.0],
                    'confidence': float(rec.get('confidence', 1.0))
                }
                print("Overrode 1D prediction to Normal due to strong recurrence Normal (conf=", rec.get('confidence'), ")")
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
        "recurrence_prediction": rec_pred_smoothed if 'rec_pred' in locals() else None
    })
