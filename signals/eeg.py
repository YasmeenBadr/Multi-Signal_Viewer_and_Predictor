from flask import Blueprint, request, jsonify, render_template
import mne
import numpy as np

bp = Blueprint("eeg", __name__, template_folder="../templates")

# Load EDF once
edf_file = r"data\S001R01.edf"
print(f"Extracting EDF parameters from {edf_file}...")
raw = mne.io.read_raw_edf(edf_file, preload=True)
fs = int(raw.info["sfreq"])
print(f"Loaded EEG with {len(raw.ch_names)} channels, fs={fs} Hz")

current_index = 0  # global pointer for streaming

# 👉 Add this route to serve the EEG page
@bp.route("/", methods=["GET"])
def eeg_home():
    return render_template("eeg.html")

@bp.route("/update", methods=["POST"])
def update():
    global current_index
    data = request.get_json()
    channels = data.get("channels", [])
    width = int(data.get("width", 5))  # seconds
    samples = width * fs

    # Get data slice
    start = current_index
    stop = start + samples
    if stop > raw.n_times:
        stop = raw.n_times
        current_index = 0  # loop back to start
    else:
        current_index = stop

    picked = raw.get_data(picks=channels, start=start, stop=stop)

    # Build response
    time_axis = np.arange(start, stop) / fs
    signals = {str(ch): picked[i].tolist() for i, ch in enumerate(channels)}

    return jsonify({"time": time_axis.tolist(), "signals": signals})
