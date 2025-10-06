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
    "recurrence_points": {}  # store cumulative points for recurrence plot
}

DISPLAY_FS = 200
STREAMING_CHUNK_DURATION = 1.0
_model_seq_len = 5000

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
    except Exception as e:
        print("Failed to load specified record:", e)
        _stream["loaded"] = False

if not _stream["loaded"]:
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
    if not os.path.exists(hea_path):
        return None
    try:
        with open(hea_path,"r",encoding="latin-1") as f:
            for line in f:
                if line.startswith("#") and ("diagnosis" in line.lower() or "reason" in line.lower()):
                    return line.strip("#").split(":",1)[-1].strip()
    except:
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
    if np.mean(np.abs(sig)) < 0.05:
        idx = 0
    label = DISEASE_CLASSES[idx]
    result = {"label": label,"description": DISEASE_DESCRIPTIONS[label]}
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
    channels = req.get("channels", list(range(12)))
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

    # Prediction
    if len(channels) == 1:
        sig_selected = seg_block[:, channels[0]]
    else:
        sig_selected = np.mean(seg_block[:, channels], axis=1)
    prediction = predict_signal(sig_selected)

    downsample_factor = max(1,int(fs/DISPLAY_FS))
    time_axis = (np.arange(seg_block.shape[0])/fs)[::downsample_factor].tolist()
    signals_out = {str(ch): seg_block[::downsample_factor,ch].astype(float).tolist() for ch in channels if 0<=ch<seg_block.shape[1]}

    # ---- XOR only if 1 channel ----
    xor_out = {}
    if len(channels) == 1:
        ch = channels[0]
        chunk = ((seg_block[:,ch]-np.min(seg_block[:,ch]))*1000).astype(np.int32)
        prev = _stream["prev_chunks"].get(ch)
        if prev is not None:
            xor = np.bitwise_xor(chunk, prev)
            xor_out[ch] = xor[::downsample_factor].tolist()
        else:
            xor_out[ch] = np.zeros_like(chunk[::downsample_factor]).tolist()
        _stream["prev_chunks"][ch] = chunk.copy()

    # ---- Polar plot ----
    polar_out = {}
    for ch in channels:
        sig = seg_block[:, ch]
        Nsig = len(sig)
        theta = np.linspace(0, 360, Nsig)
        r = sig - np.min(sig)
        polar_out[str(ch)] = {"r": r.tolist(), "theta": theta.tolist()}

    # ---- Recurrence only if 2 channels ----
    recurrence_scatter_data = {"x_vals": [], "y_vals": []}
    colormap_data = None
    if len(channels) == 2:
        chX, chY = channels[:2]
        if chX not in _stream["recurrence_points"]:
            _stream["recurrence_points"][chX] = []
            _stream["recurrence_points"][chY] = []
        _stream["recurrence_points"][chX].extend(seg_block[:,chX][::downsample_factor].tolist())
        _stream["recurrence_points"][chY].extend(seg_block[:,chY][::downsample_factor].tolist())
        recurrence_scatter_data["x_vals"] = _stream["recurrence_points"][chX]
        recurrence_scatter_data["y_vals"] = _stream["recurrence_points"][chY]
        colormap_data = np.stack([seg_block[:, chX], seg_block[:, chY]], axis=0).tolist()

    return jsonify({
        "time": time_axis,
        "signals": signals_out,
        "prediction": prediction,
        "xor": xor_out,
        "polar": polar_out,
        "recurrence_scatter": recurrence_scatter_data,
        "colormap": colormap_data
    })
