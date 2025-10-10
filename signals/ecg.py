# signals/ecg.py
# Flask blueprint that streams ECG signals (12 leads) and predicts disease names.
# It supports WFDB records in data1/ and uses the patient record defined in simple_ecg.py.

import os
import numpy as np
import torch
import torch.nn as nn
from flask import Blueprint, request, jsonify, render_template

try:
    import wfdb
except Exception:
    wfdb = None

try:
    from simple_ecg import DATA_PATH
except ImportError:
    DATA_PATH = None

ECG_BP = Blueprint("ecg", __name__, url_prefix="/ecg", template_folder="../templates")
bp = ECG_BP

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
    "pred_buffers": {}        # rolling buffers used for model prediction per channel
}

DISPLAY_FS = 200
STREAMING_CHUNK_DURATION = 1.0
_model_seq_len = 5000
POLAR_MAX_POINTS = 2000

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
# Load ECG record (from DATA_PATH or fallback)
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
def extract_diagnosis_from_hea(record_base):
    hea_path = record_base + ".hea"
    if not os.path.exists(hea_path): return None
    try:
        with open(hea_path,"r",encoding="latin-1") as f:
            for line in f:
                if line.startswith("#") and ("diagnosis" in line.lower() or "reason" in line.lower()):
                    return line.strip("#").split(":", 1)[-1].strip()
    except Exception:
        return None
    return None

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
    if disease_text and "healthy" in disease_text.lower():
        idx = 0
    # Avoid forcing 'Normal' for modest amplitude signals. Only override when
    # the signal is essentially flat (very low absolute amplitude).
    try:
        if np.max(np.abs(sig)) < 0.01:
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

    prediction = predict_signal(sig_selected)

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
        if 0 <= ch < seg_block.shape[1]:
            signals_out[str(ch)] = seg_block[::downsample_factor, ch].tolist()

    return jsonify({
        "time": time_axis,
        "signals": signals_out,
        "prediction": prediction
    })
