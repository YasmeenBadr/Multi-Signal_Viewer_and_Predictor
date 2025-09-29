# signals/ecg.py
from flask import Blueprint, request, jsonify, render_template
import wfdb
import numpy as np

bp = Blueprint("ecg", __name__, template_folder="../templates")

# --- Load ECG data once ---
dat_file = r"data\s0010_re"
print(f"Loading ECG from {dat_file}...")
record = wfdb.rdrecord(dat_file)
fs = record.fs
signals = record.p_signal.T  # shape = (channels, samples)
print(f"Loaded ECG with {signals.shape[0]} channels, fs={fs} Hz")

current_index = 0  # global streaming pointer

# Home route
@bp.route("/", methods=["GET"])
def ecg_home():
    return render_template("ecg.html")

# Update route
@bp.route("/update", methods=["POST"])
def update():
    global current_index
    data = request.get_json()
    channels = data.get("channels", [])
    width = int(data.get("width", 5))  # seconds
    samples = width * int(fs)

    start = current_index
    stop = start + samples
    if stop > signals.shape[1]:
        stop = signals.shape[1]
        current_index = 0  # loop back
    else:
        current_index = stop

    picked_signals = {str(ch): signals[ch, start:stop].tolist() for ch in channels}
    time_axis = np.arange(start, stop) / fs

    return jsonify({"time": time_axis.tolist(), "signals": picked_signals})
