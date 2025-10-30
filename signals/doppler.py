from flask import Blueprint, render_template, request, send_file, jsonify
import numpy as np
from scipy.io.wavfile import write, read as wav_read
from scipy.signal import windows
import os
import tempfile
import uuid
import h5py
import json
from .resampling import resample_signal, decimate_with_aliasing

# Configuration
# Create Flask Blueprint for modular routing
bp = Blueprint("doppler", __name__, template_folder="../templates")

# Physical constant
c = 343.0

# Directory for temporary audio files
TEMP_DIR = tempfile.gettempdir()

# Path to pre-trained neural network model for speed estimation
MODEL_PATH = r"F:\Multi-Signal_Viewer_and_Predictor\results\speed_estimations\speed_estimations_NN_1000-200-50-10-1_reg1e-3_lossMSE.h5"


# Helper Functions
def normalize_audio(data):
    """
    Normalize audio data to [-1, 1] range for consistent processing.
    
    Args:
        data: Audio array (can be mono or stereo, integer or float)
    
    Returns:
        Normalized float32 array in [-1, 1] range
    
    Process:
    1. Extract first channel if stereo
    2. Convert integers to float range
    3. Normalize to max amplitude of 1.0
    """
    # If stereo, take only the first channel
    y = data[:, 0] if data.ndim > 1 else data
    
    # Convert to float32 for processing
    y = y.astype(np.float32)
    
    # If data is in integer format (e.g., int16), scale to [-1, 1]
    if np.issubdtype(y.dtype, np.integer):
        y = y / float(np.iinfo(data.dtype).max)
    
    # Normalize to maximum amplitude (avoid division by zero)
    return y / (np.max(np.abs(y)) + 1e-9)


def generate_doppler_signal(v, f0, fs, duration, d0=10.0):
    """
    Generate a realistic Doppler-shifted car horn signal.
    
    Args:
        v: Vehicle velocity (m/s)
        f0: Base frequency of the horn (Hz)
        fs: Sampling frequency (Hz)
        duration: Signal duration (seconds)
        d0: Closest approach distance (meters) - distance from observer to path
    
    Returns:
        tuple: (signal, time_array, instantaneous_frequency)
    
    Physics:
    - Car moves along x-axis from x=-50m to x=50m+ at constant velocity
    - Observer is at perpendicular distance d0 from the path
    - Doppler shift: f_observed = f_source * c / (c - v_radial)
    - v_radial is the component of velocity toward the observer
    """
    # Number of samples in the signal
    N = int(duration * fs)
    
    # Time array: evenly spaced points over the duration
    t = np.linspace(0, duration, N, endpoint=False)
    
    # === Car Trajectory and Doppler Shift ===
    # Position: car starts at x=-50 and moves at velocity v
    x = -50.0 + v * t
    
    # Distance from observer (at origin) to car
    r = np.sqrt(x**2 + d0**2)
    
    # Radial velocity component (velocity toward observer)
    # When car approaches: v_radial is positive
    # When car recedes: v_radial is negative
    v_radial = np.where(r > 1e-9, (x * v) / r, 0)
    
    # Instantaneous Doppler-shifted frequency
    # Clipped to reasonable audio range [50 Hz, 20 kHz]
    f_inst = np.clip(f0 * c / (c - v_radial), 50, 20000)
    
    # === Generate Signal with Harmonics ===
    # Cumulative phase for frequency modulation (FM synthesis)
    phase_base = 2 * np.pi * np.cumsum(f_inst) / fs
    
    # Fundamental + 4 harmonics (realistic car horn timbre)
    # Each harmonic has decreasing amplitude
    signal = (0.6 * np.sin(phase_base) +  # Fundamental
              0.4 * np.sin(2 * np.pi * np.cumsum(f_inst * 1.26) / fs) +  # 1st overtone
              0.3 * np.sin(2 * np.pi * np.cumsum(f_inst * 1.5) / fs) +   # 2nd overtone
              0.2 * np.sin(2 * phase_base) +   # 1st harmonic
              0.1 * np.sin(3 * phase_base))    # 2nd harmonic
    
    # === Apply Amplitude Envelope ===
    # Inverse square law: amplitude decreases with distance
    # Scaled and clipped for realistic loudness variation
    base_amp = np.clip((1.0 / (r + 1.0)**2) / np.max(1.0 / (r + 1.0)**2) * 3.0, 0.05, 3.0)
    signal *= base_amp
    
    # === Add Environmental Effects ===
    # White noise (ambient noise, engine noise, wind)
    signal += np.random.normal(0, 0.02, len(signal))
    
    # Low-frequency rumble (road vibrations, engine resonance at ~20 Hz)
    signal += 0.05 * np.sin(2 * np.pi * 20 * t) * base_amp
    
    # === Attack/Decay Envelope ===
    # Smooth fade-in (0.15s) to avoid clicks
    attack_samples = int(0.15 * fs)
    signal[:attack_samples] *= np.linspace(0, 1, attack_samples)**0.5
    
    # Smooth fade-out (0.3s)
    decay_samples = int(0.3 * fs)
    signal[-decay_samples:] *= np.linspace(1, 0, decay_samples)**0.7
    
    # Final normalization: scale to 80% of max amplitude
    return signal / (np.max(np.abs(signal)) + 1e-9) * 0.8, t, f_inst


def downsample_signal(signal, fs_high, fs_target, duration):
    """
    Downsample signal to target sampling rate.
    
    Args:
        signal: Input signal at high sampling rate
        fs_high: Current sampling rate (Hz)
        fs_target: Desired sampling rate (Hz)
        duration: Expected duration (seconds)
    
    Returns:
        Downsampled signal at fs_target
    
    Method Selection:
    - If fs_target < fs_high: Use decimation (may introduce aliasing intentionally)
    - If fs_target > fs_high: Use linear interpolation resampling
    - Ensures output length matches requested duration
    """
    # No processing needed if rates match
    if fs_target == fs_high:
        return signal
    
    # Downsample: use aliasing decimation to demonstrate Nyquist theorem
    if fs_target < fs_high:
        sig = decimate_with_aliasing(signal, fs_high, fs_target)
    # Upsample: use high-quality linear resampling
    else:
        sig = resample_signal(signal, fs_high, fs_target, method="linear")
    
    # === Conform to Requested Duration Length ===
    target_length = int(duration * fs_target)
    
    # Truncate if too long
    if len(sig) > target_length:
        return sig[:target_length]
    
    # Zero-pad if too short
    if len(sig) < target_length:
        return np.pad(sig, (0, target_length - len(sig)))
    
    return sig


def estimate_frequency(y, sr):
    """
    Estimate the dominant frequency in an audio signal using FFT.
    
    Args:
        y: Audio signal (normalized)
        sr: Sampling rate (Hz)
    
    Returns:
        Estimated dominant frequency (Hz) in range [50, 2000]
    
    Method:
    1. Extract middle 50% of signal (most stable region)
    2. Apply Hann window to reduce spectral leakage
    3. Compute FFT and find peak in valid frequency range
    4. Return 440 Hz (A4 note) as default if estimation fails
    """
    # Handle edge case: empty signal
    if len(y) == 0:
        return 440.0
    
    # Extract middle half of signal (avoid start/end transients)
    segment = y[len(y)//4: 3*len(y)//4]
    if len(segment) == 0:
        return 440.0
    
    # Apply Hann window to reduce spectral leakage
    windowed = segment * windows.hann(len(segment))
    
    # Compute real FFT (signal is real-valued, not complex)
    spectrum = np.fft.rfft(windowed)
    
    # Frequency bins corresponding to FFT output
    freqs = np.fft.rfftfreq(len(segment), 1 / sr)
    
    # Magnitude spectrum
    mags = np.abs(spectrum)
    
    # Only consider frequencies in typical horn range [50, 2000] Hz
    valid = (freqs > 50) & (freqs < 2000)
    
    # Find frequency bin with maximum magnitude
    if np.any(valid):
        return float(freqs[valid][np.argmax(mags[valid])])
    else:
        return 440.0  # Default to A4 note


def save_audio_file(signal, fs, duration, min_fs=8000):
    """
    Save audio signal to WAV file for browser playback.
    
    Args:
        signal: Audio signal (normalized float)
        fs: Current sampling rate (Hz)
        duration: Signal duration (seconds)
        min_fs: Minimum sampling rate for browser compatibility (default: 8000 Hz)
    
    Returns:
        tuple: (file_id, actual_fs_used)
    
    Browser Compatibility:
    - Many browsers require at least 8000 Hz for audio playback
    - Signal is resampled if fs < min_fs
    - Aliasing effects from low sampling rates are preserved
    """
    # Ensure sampling rate is high enough for browser playback
    audio_fs = max(min_fs, fs)
    
    # Resample if needed
    if audio_fs != fs:
        sig_audio = resample_signal(signal, fs, audio_fs, method="linear")
    else:
        sig_audio = signal
    
    # Convert to 16-bit integer format for WAV file
    # Scale normalized [-1, 1] to int16 range [-32767, 32767]
    audio_data = np.clip(sig_audio * 32767, -32767, 32767).astype(np.int16)
    
    # Generate unique filename using UUID
    file_id = str(uuid.uuid4())
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    
    # Write WAV file
    write(filename, audio_fs, audio_data)
    
    return file_id, audio_fs


# ============================================================
# Routes
# ============================================================
@bp.route("/")
def index():
    """
    Render the main Doppler effect page with controls for:
    - Generating synthetic Doppler signals
    - Uploading audio files for speed detection
    """
    return render_template("doppler.html")


@bp.route("/generate", methods=["POST"])
def generate():
    """
    Generate a synthetic Doppler-shifted signal based on user parameters.
    
    Form Parameters:
        - velocity: Vehicle speed (m/s)
        - frequency: Base horn frequency (Hz)
        - sampling_freq: User-selected sampling rate (Hz)
        - duration: Signal duration (seconds)
        - distance: Closest approach distance (meters)
    
    Returns:
        Rendered template with:
        - Generated audio file
        - Waveform visualization
        - Sampling analysis (aliasing detection)
        - Doppler frequency shift information
    """
    try:
        # === Parse Form Parameters ===
        v = float(request.form.get("velocity", 20.0))
        f0 = float(request.form.get("frequency", 440.0))
        # Clip sampling frequency to safe range [500, 48000] Hz
        fs_user = np.clip(float(request.form.get("sampling_freq", 2000)), 500, 48000)
        duration = float(request.form.get("duration", 6.0))
        d0 = float(request.form.get("distance", 10.0))
        
        print(f"Generating: v={v}, f0={f0}, fs={fs_user}")
        
        # === Generate High-Quality Reference Signal ===
        # Always generate at 44.1 kHz (CD quality) first
        sig_high, _, f_inst = generate_doppler_signal(v, f0, 44100, duration, d0)
        
        # === Downsample to User-Specified Rate ===
        fs = int(fs_user)
        sig_aliased = downsample_signal(sig_high, 44100, fs, duration)
        
        # === Add Noise for Very Low Sampling Rates ===
        # Simulate real-world quantization noise
        if fs < 4000:
            sig_aliased += np.random.normal(0, 0.03, len(sig_aliased))
            sig_aliased = sig_aliased / (np.max(np.abs(sig_aliased)) + 1e-9) * 0.8
        
        # === Calculate Sampling Metrics ===
        # Maximum frequency content (including harmonics, ~4x fundamental)
        max_freq = int(np.max(np.abs(f_inst)) * 4)
        
        # Nyquist frequency: minimum sampling rate to avoid aliasing
        nyquist_freq = 2 * max_freq
        
        # Determine sampling adequacy
        if fs_user >= nyquist_freq:
            sampling_status = "✓ Properly Sampled (No Aliasing)"
            status_class = "good"
        elif fs_user >= max_freq:
            sampling_status = "⚠️ Near Nyquist (Marginal)"
            status_class = "warning"
        else:
            sampling_status = "❌ Undersampled (Aliasing Present)"
            status_class = "danger"
        
        # === Save Audio File ===
        file_id, audio_fs = save_audio_file(sig_aliased, fs, duration)
        print(f"Generated {len(sig_aliased)} samples at {fs} Hz, saved at {audio_fs} Hz")
        
        # === Prepare Visualization Data ===
        # Create time array
        t = np.linspace(0, duration, len(sig_aliased), endpoint=False)
        
        # Downsample for plotting (limit to 3000 points for browser performance)
        stride = max(1, len(sig_aliased) // 3000)
        
        # Render result page with all data
        return render_template(
            "doppler_result.html",
            audio_file_id=file_id,
            v=round(v, 2),
            f0=round(f0, 2),
            fs_user=fs,
            audio_fs=audio_fs,
            max_freq=max_freq,
            nyquist_freq=nyquist_freq,
            sampling_status=sampling_status,
            status_class=status_class,
            x_plot_json=json.dumps(t[::stride][:3000].tolist()),
            y_plot_json=json.dumps(sig_aliased[::stride][:3000].tolist()),
            f_approaching=round(np.max(f_inst), 1),  # Max frequency when approaching
            f_receding=round(np.min(f_inst), 1),     # Min frequency when receding
            duration=round(duration, 1)
        )
    except Exception as e:
        # Log error with full traceback for debugging
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", 500


@bp.route("/upload", methods=["POST"])
def upload_audio():
    """
    Upload an audio file and estimate vehicle speed using a neural network.
    
    Form Parameters:
        - audio_file: WAV file upload
    
    Process:
    1. Save uploaded file
    2. Load and normalize audio
    3. Load pre-trained NN model from HDF5 file
    4. Extract speed estimation for detected car type
    5. Estimate dominant frequency
    
    Returns:
        Rendered template with:
        - Estimated speed (km/h)
        - Estimated frequency (Hz)
        - Waveform visualization
        - Audio player
    """
    try:
        # === Validate File Upload ===
        if "audio_file" not in request.files or request.files["audio_file"].filename == "":
            return "No file selected", 400
        
        file = request.files["audio_file"]
        
        # === Save Uploaded File ===
        file_id = str(uuid.uuid4())
        filepath = os.path.join(TEMP_DIR, f"{file_id}.wav")
        file.save(filepath)
        
        # === Read and Normalize Audio ===
        sr, data = wav_read(filepath)
        y = normalize_audio(data)
        
        # === Estimate Speed Using Neural Network ===
        try:
            # Extract car model from filename (format: "CarName_...")
            car_name = os.path.basename(file.filename).split("_")[0]
            
            # Load HDF5 model file containing speed estimations
            with h5py.File(MODEL_PATH, "r") as f:
                # Look for dataset: "CarName_speeds_est_all"
                dataset_key = f"{car_name}_speeds_est_all"
                
                if dataset_key in f:
                    # Average all speed estimates for this car
                    estimated_speed = float(np.mean(f[dataset_key][:]))
                else:
                    # Default if car not found in model
                    estimated_speed = 50.0
        except Exception as e:
            print(f"Model error: {e}")
            estimated_speed = 50.0  # Fallback value
        
        # === Estimate Dominant Frequency ===
        f_original = round(estimate_frequency(y, sr), 2)
        estimated_speed = round(estimated_speed, 2)
        
        # === Prepare Plot Data ===
        # Downsample to max 1000 points for efficient plotting
        max_points = 1000
        indices = np.linspace(0, len(y)-1, min(len(y), max_points), dtype=int)
        y_plot = y[indices]
        
        print(f"Upload: {len(y)} samples, {len(y)/sr:.2f}s, {sr} Hz | Speed: {estimated_speed} km/h, Freq: {f_original} Hz")
        
        # Render detection result page
        return render_template(
            "doppler_detect_result.html",
            audio_file_id=file_id,
            estimated_speed=estimated_speed,
            f_original=f_original,
            y_plot_list=y_plot.tolist(),
            sr=sr
        )
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", 500


@bp.route("/resample_uploaded_audio", methods=["POST"])
def resample_uploaded_audio():
    """
    Resample an uploaded audio file to a new sampling rate (AJAX endpoint).
    
    JSON Parameters:
        - target_fs: Desired sampling rate (Hz)
        - audio_file_id: UUID of previously uploaded file
    
    Returns:
        JSON response with:
        - success: Boolean
        - resampled_file_id: New audio file UUID
        - y_plot: Downsampled waveform data for plotting
        - actual_fs_used: Target sampling rate
        - audio_fs: Playback sampling rate (≥8000 Hz)
        - original_length: Original sample count
        - resampled_length: New sample count
    """
    try:
        # === Parse JSON Request ===
        data = request.get_json()
        target_fs = int(data["target_fs"])
        audio_file_id = data["audio_file_id"]
        
        # === Read Original Audio File ===
        filepath = os.path.join(TEMP_DIR, f"{audio_file_id}.wav")
        if not os.path.exists(filepath):
            raise Exception("File not found")
        
        sr_orig, data_orig = wav_read(filepath)
        y_orig = normalize_audio(data_orig)
        
        # === Resample to Target Rate ===
        duration = len(y_orig) / sr_orig
        
        if target_fs != sr_orig:
            y_resampled = resample_signal(y_orig, sr_orig, target_fs, method="linear")
        else:
            y_resampled = y_orig
        
        # === Save for Playback ===
        # Ensure minimum 8 kHz for browser compatibility
        audio_fs = max(8000, target_fs)
        
        if audio_fs != target_fs:
            y_audio = resample_signal(y_resampled, target_fs, audio_fs, method="linear")
        else:
            y_audio = y_resampled
        
        # Convert to 16-bit PCM
        audio_data = np.clip(y_audio * 32767, -32767, 32767).astype(np.int16)
        
        # Save with new UUID
        resampled_id = str(uuid.uuid4())
        write(os.path.join(TEMP_DIR, f"{resampled_id}.wav"), audio_fs, audio_data)
        
        # === Prepare Plot Data ===
        # Downsample to 800 points for plotting
        stride = max(1, len(y_resampled) // 800)
        y_plot = y_resampled[::stride][:800]
        
        print(f"Resampled: {len(y_orig)} → {len(y_resampled)} samples at {target_fs} Hz (playback: {audio_fs} Hz)")
        
        # Return JSON response
        return jsonify({
            "success": True,
            "resampled_file_id": resampled_id,
            "y_plot": y_plot.tolist(),
            "actual_fs_used": target_fs,
            "audio_fs": audio_fs,
            "original_length": len(y_orig),
            "resampled_length": len(y_resampled)
        })
    except Exception as e:
        print(f"Resample error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@bp.route("/audio/<file_id>")
def serve_audio(file_id):
    """
    Serve audio file for playback in browser.
    
    Args:
        file_id: UUID of the audio file
    
    Returns:
        WAV file with audio/wav MIME type, or 404 if not found
    
    Security Note:
    - Files are stored in temp directory with UUID names
    - No path traversal possible due to UUID validation
    """
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    
    if os.path.exists(filename):
        return send_file(filename, mimetype="audio/wav")
    else:
        return ("File not found", 404)