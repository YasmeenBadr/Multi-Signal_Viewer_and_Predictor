from flask import Blueprint, request, jsonify, render_template
import mne
import numpy as np
from scipy.signal import butter, lfilter, filtfilt 
import sys

bp = Blueprint("eeg", __name__, template_folder="../templates")

# --- INITIALIZATION ---
edf_file = r"data\s001R01.edf"
print(f"Extracting EDF parameters from {edf_file}...")

try:
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False) 
except FileNotFoundError:
    print(f"FATAL ERROR: EDF file not found at {edf_file}. Please check your path.")
    info = mne.create_info(ch_names=[f'Ch{i+1}' for i in range(8)], sfreq=160, ch_types='eeg')
    raw = mne.io.RawArray(np.zeros((8, 1000)), info)
except Exception as e:
    print(f"FATAL ERROR during MNE loading: {e}")
    info = mne.create_info(ch_names=[f'Ch{i+1}' for i in range(8)], sfreq=160, ch_types='eeg')
    raw = mne.io.RawArray(np.zeros((8, 1000)), info)


fs = int(raw.info["sfreq"])
print(f"Loaded EEG with {len(raw.ch_names)} channels, fs={fs} Hz")

# FIX: Start streaming 10 seconds into the file to skip potentially flat setup data
INITIAL_OFFSET_SAMPLES = fs * 10
current_index = INITIAL_OFFSET_SAMPLES if raw.n_times > INITIAL_OFFSET_SAMPLES else 0 

CHUNK_SAMPLES = int(fs / 10) # 16 samples per update for 100ms interval

# Define EEG Frequency Bands
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# --- BAND POWER FUNCTIONS ---

def butter_bandpass(lowcut, highcut, fs, order=2): 
    """Returns the coefficients for a Butterworth filter."""
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
    """
    Calculates the average power (variance) for each EEG frequency band 
    and applies a massive scaling factor for frontend visibility.
    """
    band_powers = {}
    # FINAL SCALING FACTOR: Ensures visibility on the plot
    SCALING_FACTOR = 10000000000.0 
    
    for band, (low, high) in BANDS.items():
        if high <= low or low >= fs/2:
            band_powers[band] = 0.0
            continue
            
        b, a = butter_bandpass(low, high, fs, order=2) 
        
        if b is None or a is None:
            band_powers[band] = 0.0
            continue
        
        try:
            # Use filtfilt for stable, zero-phase filtering
            filtered_data = filtfilt(b, a, data.astype(float))
            # Power is Mean Squared Amplitude
            power_value = np.mean(filtered_data**2)
            
            # Apply scaling for plot visibility
            scaled_power = (power_value if np.isfinite(power_value) else 0.0) * SCALING_FACTOR
            
            band_powers[band] = scaled_power
        except Exception as e:
            print(f"Error calculating band power for {band}: {e}", file=sys.stderr)
            band_powers[band] = 0.0

    return band_powers

# --- FLASK ROUTES ---

@bp.route("/", methods=["GET"])
def eeg_home():
    return render_template("eeg.html")

@bp.route("/update", methods=["POST"])
def update():
    global current_index
    data = request.get_json()
    channels = data.get("channels", [])
    
    samples_to_send = CHUNK_SAMPLES 

    start = current_index
    stop = start + samples_to_send
    
    # Handle wrap-around for looping
    if stop > raw.n_times:
        stop = raw.n_times
        samples_to_send = stop - start
        current_index = INITIAL_OFFSET_SAMPLES if raw.n_times > INITIAL_OFFSET_SAMPLES else 0 
    else:
        current_index = stop

    if samples_to_send <= 0:
        current_index = INITIAL_OFFSET_SAMPLES if raw.n_times > INITIAL_OFFSET_SAMPLES else 0 
        return jsonify({"n_samples": 0, "signals": {}, "band_power": {}})

    # Get data for ALL selected channels
    picked = raw.get_data(picks=channels, start=start, stop=stop)

    # NEW LOGIC: Calculate and AVERAGE band power across all selected channels
    band_power_data = {}
    if picked.shape[0] > 0:
        all_channel_powers = [calculate_band_power(picked[i], fs) for i in range(picked.shape[0])]
        
        if all_channel_powers:
            for band in BANDS.keys():
                # Get power for 'band' from every channel and compute the mean
                avg_power = np.mean([cp.get(band, 0.0) for cp in all_channel_powers])
                band_power_data[band] = float(avg_power)
    
    # Build response
    signals = {str(ch): picked[i].tolist() for i, ch in enumerate(channels)}

    return jsonify({
        "n_samples": picked.shape[1], 
        "signals": signals,
        "band_power": band_power_data  # This now contains averaged power
    })