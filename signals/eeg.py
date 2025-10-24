from flask import Blueprint, request, jsonify, render_template, current_app
import mne
import numpy as np
from scipy.signal import butter, lfilter, filtfilt 
import os
import sys
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from .resampling import decimate_with_aliasing

bp = Blueprint("eeg", __name__, template_folder="../templates")

# --- GLOBAL STATE (To hold the loaded data) ---
class EEGData:
    def __init__(self):
        self.raw = None
        self.fs = 160
        self.fs_native = 160
        self.n_times = 0
        self.ch_names = []
        self.current_index = 0

eeg_data = EEGData()
INITIAL_OFFSET_SAMPLES = 0  # Will be calculated after file load
CHUNK_SAMPLES = 16          # Default, will be recalculated

# Persistent aliasing phase state (per target fs) for EEG decimation
EEG_ALIAS_PHASE = {}

# EEG frequency bands and helpers for band power
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50),
}

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    if fs <= 0 or highcut <= 0 or lowcut >= highcut or lowcut >= nyq:
        return None, None
    low = max(1e-6, float(lowcut) / nyq)
    high = min(0.999, float(highcut) / nyq)
    try:
        if lowcut <= 0.5:
            from scipy.signal import butter
            b, a = butter(order, high, btype='lowpass')
        else:
            from scipy.signal import butter
            b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    except Exception:
        return None, None

def calculate_band_power(data, fs):
    try:
        x = np.asarray(data, dtype=float)
        if x.size == 0 or fs <= 0:
            return {band: 0.0 for band in BANDS}
        powers = {}
        from scipy.signal import filtfilt
        for band, (low, high) in BANDS.items():
            b, a = butter_bandpass(low, high, fs, order=2)
            if b is None or a is None:
                powers[band] = 0.0
                continue
            try:
                y = filtfilt(b, a, x)
                p = float(np.mean(y**2)) if y.size else 0.0
            except Exception:
                p = 0.0
            powers[band] = p
        return powers
    except Exception:
        return {band: 0.0 for band in BANDS}

# --- SERVER-SIDE XOR STATE ---
# Maintains rolling buffers and previous window per channel for XOR mode
_XOR_BUFFERS: Dict[int, List[float]] = {}
_XOR_PREV_WINDOWS: Dict[int, List[float]] = {}

# ... (Your existing code remains the same)

# --- NEW UPLOAD ROUTE ---
@bp.route("/upload", methods=["POST"])
def upload_file():

    global INITIAL_OFFSET_SAMPLES, CHUNK_SAMPLES
    
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    if file:
        # Save the file temporarily
        filename = file.filename
        # Use a safe path, e.g., 'uploads' directory
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        
        # NOTE: For a real app, you should check file extension and secure filenames
        try:
            file.save(filepath)
            
            # Load the EEG file with MNE (support both EDF and FIF)
            if filename.lower().endswith('.edf'):
                eeg_data.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            elif filename.lower().endswith('.fif') or filename.lower().endswith('.fif.gz'):
                eeg_data.raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
            else:
                return jsonify({"success": False, "message": "Unsupported file format. Please use .edf or .fif files."}), 400
            eeg_data.fs = int(eeg_data.raw.info["sfreq"])
            eeg_data.fs_native = int(eeg_data.raw.info["sfreq"])  # store native FS
            eeg_data.n_times = eeg_data.raw.n_times
            eeg_data.ch_names = eeg_data.raw.ch_names
            
            # Recalculate streaming parameters
            INITIAL_OFFSET_SAMPLES = eeg_data.fs * 10
            eeg_data.current_index = INITIAL_OFFSET_SAMPLES if eeg_data.n_times > INITIAL_OFFSET_SAMPLES else 0 
            CHUNK_SAMPLES = int(eeg_data.fs / 10) # 16 samples per update for 100ms interval

            print(f"File loaded. Channels: {len(eeg_data.ch_names)}, fs: {eeg_data.fs} Hz")
            
            # Optionally delete the file if you don't need it anymore, 
            # but keep it for continuous streaming.
            
            # Map channel indices to names for the frontend
            ch_info = {i: name for i, name in enumerate(eeg_data.ch_names)}
            
            return jsonify({
                "success": True, 
                "message": f"File {filename} loaded successfully.",
                "channels": ch_info,
                "fs": eeg_data.fs
            })
            
        except Exception as e:
            error_msg = f"Error processing file: {e}"
            print(f"FATAL ERROR: {error_msg}")
            return jsonify({"success": False, "message": error_msg}), 500

# ... (Your existing code remains the same)

# --- FLASK ROUTES (update needs to use the global state) ---

@bp.route("/", methods=["GET"])
def eeg_home():
    # Pass a default empty list or load a default file if needed
    # For now, we'll just render the template
    return render_template("eeg.html")

@bp.route("/set_sampling", methods=["POST"])
def set_sampling():
    try:
        global CHUNK_SAMPLES
        data = request.get_json(silent=True) or {}
        new_fs = float(data.get("sampling_freq", data.get("frequency", eeg_data.fs)))
        fs_native = int(getattr(eeg_data, "fs_native", eeg_data.fs))
        new_fs = max(1.0, min(float(fs_native), float(new_fs)))
        eeg_data.fs = int(new_fs)
        CHUNK_SAMPLES = max(1, int(eeg_data.fs / 10))
        EEG_ALIAS_PHASE.clear()
        return jsonify({"success": True, "current_sampling": int(eeg_data.fs)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/reset_sampling", methods=["POST"])
def reset_sampling():
    try:
        global CHUNK_SAMPLES
        fs_native = int(getattr(eeg_data, "fs_native", eeg_data.fs))
        eeg_data.fs = fs_native
        CHUNK_SAMPLES = max(1, int(eeg_data.fs / 10))
        EEG_ALIAS_PHASE.clear()
        return jsonify({"success": True, "current_sampling": int(eeg_data.fs)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/update", methods=["POST"])
def update():
    if eeg_data.raw is None:
        return jsonify({"n_samples": 0, "signals": {}, "band_power": {}, "message": "No file loaded."})
        
    # Make sure to use the eeg_data global object
    global CHUNK_SAMPLES
    
    data = request.get_json()
    channels = data.get("channels", [])
    mode = data.get("mode", "time")
    width = float(data.get("width", 5))
    
    # Determine native-sample window to extract so that the downsampled output ~ CHUNK_SAMPLES at current fs
    fs_cur = max(1, int(eeg_data.fs))
    fs_native = max(1, int(getattr(eeg_data, "fs_native", eeg_data.fs)))
    samples_to_send_native = int(round(CHUNK_SAMPLES * (fs_native / fs_cur)))
    samples_to_send_native = max(1, samples_to_send_native)

    start = eeg_data.current_index
    stop = start + samples_to_send_native
    
    # Handle wrap-around for looping
    if stop > eeg_data.n_times:
        stop = eeg_data.n_times
        samples_to_send_native = stop - start
        eeg_data.current_index = INITIAL_OFFSET_SAMPLES if eeg_data.n_times > INITIAL_OFFSET_SAMPLES else 0 
    else:
        eeg_data.current_index = stop

    if samples_to_send_native <= 0:
        eeg_data.current_index = INITIAL_OFFSET_SAMPLES if eeg_data.n_times > INITIAL_OFFSET_SAMPLES else 0 
        return jsonify({"n_samples": 0, "signals": {}, "band_power": {}})

    # Get data for ALL selected channels
    picked = eeg_data.raw.get_data(picks=channels, start=start, stop=stop)

    # Apply aliasing decimation from native fs to current fs per chunk, preserving phase continuity
    # picked shape: (channels, samples) -> transpose to (samples, channels)
    if fs_cur < fs_native and picked.size > 0:
        try:
            chunk_native = picked.T.astype(np.float32)
            chunk_ds = decimate_with_aliasing(
                chunk_native,
                native_fs=fs_native,
                target_fs=fs_cur,
                pos_native=int(start),
                phase_state=EEG_ALIAS_PHASE,
            )
            # shape back to (channels, samples)
            picked = chunk_ds.T
        except Exception as _eeg_res_err:
            # Fallback: basic stride decimation per channel
            stride = max(1, int(round(fs_native / fs_cur)))
            picked = picked[:, ::stride]

    # NEW LOGIC: Calculate and AVERAGE band power across all selected channels
    band_power_data = {}
    if picked.shape[0] > 0:
        # Check if "cycle" mode is *requested* by the frontend (optional optimization)
        # Since we don't have the mode here, we compute if it's a single channel,
        # but the frontend's main use for band power is "cycle" mode.
        # Since the 'cycle' mode only selects one channel, we can optimize by only running the calc if 1 channel is selected.
        # But for robustness, the code returns averaged band power if multiple channels are selected.
        
        all_channel_powers = [calculate_band_power(picked[i], eeg_data.fs) for i in range(picked.shape[0])]
        
        if all_channel_powers:
            for band in BANDS.keys():
                avg_power = np.mean([cp.get(band, 0.0) for cp in all_channel_powers])
                band_power_data[band] = float(avg_power)
    
    # Build response (after any decimation)
    signals = {str(ch): picked[i].tolist() for i, ch in enumerate(channels)}

    response = {
        "n_samples": picked.shape[1],
        "signals": signals,
        "band_power": band_power_data
    }

    # Server-side XOR computation for single-channel XOR mode
    try:
        if mode == "xor" and len(channels) == 1 and picked.shape[0] == 1:
            ch = int(channels[0])
            new_samples = signals[str(ch)]

            # Initialize buffers if not present
            if ch not in _XOR_BUFFERS:
                _XOR_BUFFERS[ch] = []
            if ch not in _XOR_PREV_WINDOWS:
                _XOR_PREV_WINDOWS[ch] = []

            # Rolling buffer to maintain last window seconds of data (use current fs)
            chunk_size = max(1, int(width * eeg_data.fs))

            buf = _XOR_BUFFERS[ch]
            buf.extend(new_samples)
            if len(buf) > chunk_size:
                del buf[0:len(buf) - chunk_size]

            xor_series = buf.copy()
            if len(buf) == chunk_size:
                prev_window = _XOR_PREV_WINDOWS.get(ch, [])
                if len(prev_window) == chunk_size:
                    # Binary XOR based on mid-level threshold, comparing current window
                    # with reversed previous window (to mimic forward vs reverse pairing)
                    combined = np.array(buf + prev_window, dtype=float)
                    y_min = float(np.min(combined)) if combined.size > 0 else 0.0
                    y_max = float(np.max(combined)) if combined.size > 0 else 1.0
                    y_range = max(1e-9, y_max - y_min)
                    threshold = (y_max + y_min) / 2.0

                    mapped_high = y_min - 0.1 * y_range + 0.85 * (1.2 * y_range)
                    mapped_low = y_min - 0.1 * y_range + 0.15 * (1.2 * y_range)

                    xor_series = []
                    for i in range(chunk_size):
                        cur_val = buf[i]
                        prev_val = prev_window[chunk_size - 1 - i]
                        cur_bit = 1 if cur_val > threshold else 0
                        prev_bit = 1 if prev_val > threshold else 0
                        bit = cur_bit ^ prev_bit
                        xor_series.append(mapped_high if bit == 1 else mapped_low)

                # Update previous window after computing
                _XOR_PREV_WINDOWS[ch] = buf[-chunk_size:].copy()

            response["xor"] = xor_series
    except Exception as xor_err:
        print(f"XOR computation error: {xor_err}")

    return jsonify(response)


# --- PREDICTION ROUTE ---
def run_all_predictions(eeg_1d: np.ndarray):
    x = np.asarray(eeg_1d, dtype=np.float32).flatten()
    if x.size == 0 or not np.isfinite(x).any():
        return {
            "epilepsy": {"label": "Normal", "confidence": 0.5},
            "alzheimer": {"label": "Normal", "confidence": 0.5},
            "sleep_disorder": {"label": "Normal", "confidence": 0.5},
            "parkinson": {"label": "Healthy", "confidence": 0.5},
        }
    x = (x - np.mean(x)) / (np.std(x) + 1e-6)
    var = float(np.var(x))
    k = float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 0.0
    e_conf = max(0.5, min(0.95, 0.5 + k * 0.1))
    a_conf = max(0.5, min(0.95, 0.5 + (0.2 - min(0.2, var)) * 0.5))
    s_conf = max(0.5, min(0.95, 0.5 + (var) * 0.05))
    p_conf = max(0.5, min(0.95, 0.5 + k * 0.05))
    return {
        "epilepsy": {"label": "Normal" if e_conf < 0.7 else "Epilepsy", "confidence": e_conf},
        "alzheimer": {"label": "Normal" if a_conf < 0.7 else "Alzheimer", "confidence": a_conf},
        "sleep_disorder": {"label": "Normal" if s_conf < 0.8 else "Sleep Disorder", "confidence": s_conf},
        "parkinson": {"label": "Healthy" if p_conf < 0.8 else "Parkinson", "confidence": p_conf},
    }
@bp.route("/predict", methods=["POST"])
def predict_diseases():
    """Run disease predictions on current EEG data"""
    if eeg_data.raw is None:
        return jsonify({"success": False, "message": "No file loaded."}), 400
    
    try:
        data = request.get_json()
        channels = data.get("channels", [])
        downsample_factor = data.get("downsample_factor", 1)
        
        if not channels:
            return jsonify({"success": False, "message": "No channels selected."}), 400
        
        # Get current EEG data from the first selected channel
        start = eeg_data.current_index
        stop = start + CHUNK_SAMPLES
        
        if stop > eeg_data.n_times:
            stop = eeg_data.n_times
        
        if stop <= start:
            return jsonify({"success": False, "message": "No data available for prediction."}), 400
        
        # Get data for the first selected channel
        picked = eeg_data.raw.get_data(picks=[channels[0]], start=start, stop=stop)
        
        if picked.shape[0] == 0 or picked.shape[1] == 0:
            return jsonify({"success": False, "message": "No valid data for prediction."}), 400
        
        # Use the first channel's data for prediction
        eeg_data_for_prediction = picked[0]  # Shape: (samples,)
        
        # Apply downsampling (aliasing effect)
        if downsample_factor > 1:
            # Simple downsampling by taking every nth sample
            eeg_data_for_prediction = eeg_data_for_prediction[::int(downsample_factor)]
            print(f"Applied {downsample_factor}x downsampling. New length: {len(eeg_data_for_prediction)}")
        
        # Run all predictions
        prediction_results = run_all_predictions(eeg_data_for_prediction)
        
        return jsonify({
            "success": True,
            "predictions": prediction_results,
            "channel_used": channels[0],
            "data_length": len(eeg_data_for_prediction),
            "downsample_factor": downsample_factor
        })
        
    except Exception as e:
        error_msg = f"Error in prediction: {e}"
        print(f"PREDICTION ERROR: {error_msg}")
        return jsonify({"success": False, "message": error_msg}), 500