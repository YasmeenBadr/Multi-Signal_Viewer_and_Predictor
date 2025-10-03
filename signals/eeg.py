from flask import Blueprint, request, jsonify, render_template, current_app
import mne
import numpy as np
from scipy.signal import butter, lfilter, filtfilt 
import os
import sys

bp = Blueprint("eeg", __name__, template_folder="../templates")

# --- GLOBAL STATE (To hold the loaded data) ---
class EEGData:
    def __init__(self):
        self.raw = None
        self.fs = 160
        self.n_times = 0
        self.ch_names = []
        self.current_index = 0

eeg_data = EEGData()
INITIAL_OFFSET_SAMPLES = 0  # Will be calculated after file load
CHUNK_SAMPLES = 16          # Default, will be recalculated

# Define EEG Frequency Bands (Kept for band power calc)
BANDS = {
    'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 
    'Beta': (13, 30), 'Gamma': (30, 50)
}

# --- HELPER FUNCTIONS (butter_bandpass and calculate_band_power remain the same) ---

def butter_bandpass(lowcut, highcut, fs, order=2): 
    # ... (Your existing butter_bandpass function) ...
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if lowcut == 0.5 and highcut == 4:
        b, a = butter(order, high, btype='lowpass')
    elif lowcut > 0 and highcut < nyq:
        b, a = butter(order, [low, high], btype='bandpass')
    else:
        return None, None 
        
    return b, a

def calculate_band_power(data, fs):
    # ... (Your existing calculate_band_power function) ...
    band_powers = {}
    SCALING_FACTOR = 10000000000000.0 
    
    for band, (low, high) in BANDS.items():
        if high <= low or low >= fs/2:
            band_powers[band] = 0.0
            continue
            
        b, a = butter_bandpass(low, high, fs, order=2) 
        
        if b is None or a is None:
            band_powers[band] = 0.0
            continue
        
        try:
            filtered_data = filtfilt(b, a, data.astype(float))
            power_value = np.mean(filtered_data**2)
            scaled_power = (power_value if np.isfinite(power_value) else 0.0) * SCALING_FACTOR
            band_powers[band] = scaled_power
        except Exception as e:
            print(f"Error calculating band power for {band}: {e}", file=sys.stderr)
            band_powers[band] = 0.0

    return band_powers


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
            
            # Load the EDF file with MNE
            eeg_data.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            eeg_data.fs = int(eeg_data.raw.info["sfreq"])
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


# --- FLASK ROUTES (update needs to use the global state) ---

@bp.route("/", methods=["GET"])
def eeg_home():
    # Pass a default empty list or load a default file if needed
    # For now, we'll just render the template
    return render_template("eeg.html")

@bp.route("/update", methods=["POST"])
def update():
    if eeg_data.raw is None:
        return jsonify({"n_samples": 0, "signals": {}, "band_power": {}, "message": "No file loaded."})
        
    # Make sure to use the eeg_data global object
    global CHUNK_SAMPLES
    
    data = request.get_json()
    channels = data.get("channels", [])
    
    samples_to_send = CHUNK_SAMPLES 

    start = eeg_data.current_index
    stop = start + samples_to_send
    
    # Handle wrap-around for looping
    if stop > eeg_data.n_times:
        stop = eeg_data.n_times
        samples_to_send = stop - start
        eeg_data.current_index = INITIAL_OFFSET_SAMPLES if eeg_data.n_times > INITIAL_OFFSET_SAMPLES else 0 
    else:
        eeg_data.current_index = stop

    if samples_to_send <= 0:
        eeg_data.current_index = INITIAL_OFFSET_SAMPLES if eeg_data.n_times > INITIAL_OFFSET_SAMPLES else 0 
        return jsonify({"n_samples": 0, "signals": {}, "band_power": {}})

    # Get data for ALL selected channels
    picked = eeg_data.raw.get_data(picks=channels, start=start, stop=stop)

    # NEW LOGIC: Calculate and AVERAGE band power across all selected channels
    band_power_data = {}
    if picked.shape[0] > 0:
        # Check if "cycle" mode is *requested* by the frontend (optional optimization)
        # Since we don't have the mode here, we compute if it's a single channel,
        # but the frontend's main use for band power is "cycle" mode.
        # However, the Topomap mode uses raw signals, so we only need band power for 'cycle'.
        # Since the 'cycle' mode only selects one channel, we can optimize by only running the calc if 1 channel is selected.
        # But for robustness, the code returns averaged band power if multiple channels are selected.
        
        # If the Topomap mode is using band power, this logic would need adjustment.
        # Based on the plan, Topomap uses raw signal amplitude, so this part is fine as is.
        
        all_channel_powers = [calculate_band_power(picked[i], eeg_data.fs) for i in range(picked.shape[0])]
        
        if all_channel_powers:
            for band in BANDS.keys():
                avg_power = np.mean([cp.get(band, 0.0) for cp in all_channel_powers])
                band_power_data[band] = float(avg_power)
    
    # Build response
    signals = {str(ch): picked[i].tolist() for i, ch in enumerate(channels)}

    return jsonify({
        "n_samples": picked.shape[1], 
        "signals": signals,
        "band_power": band_power_data 
    })