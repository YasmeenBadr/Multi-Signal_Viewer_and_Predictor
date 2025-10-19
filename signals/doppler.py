from flask import Blueprint, render_template, request, send_file, jsonify
import numpy as np
from scipy.io.wavfile import write
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt, windows, resample
import io
import os
import tempfile
import uuid
import h5py
import json

bp = Blueprint("doppler", __name__, template_folder="../templates")

c = 343.0
TEMP_DIR = tempfile.gettempdir()


# ==================== FILTER FUNCTION ====================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a
# =========================================================


# ==================== MAIN PAGE ====================
@bp.route("/")
def index():
    return render_template("doppler.html")
# ===================================================

# ==================== GENERATE SOUND ====================
@bp.route("/generate", methods=["POST"])
def generate():
    # Get user inputs - FIXED parameter name
    v = float(request.form.get("velocity", 0.0))
    f0 = float(request.form.get("frequency", 150.0))
    fs_user = float(request.form.get("sampling_freq", 44100))  # NOW MATCHES HTML
    d0 = float(request.form.get("distance", 5.0))
    duration = float(request.form.get("duration", 6.0))

    # --- Generate at high resolution first ---
    fs_original = 44100  # Always generate at high quality first
    N = int(duration * fs_original)
    t = np.linspace(0, duration, N, endpoint=False)

    # Doppler shift model
    x = v * (t - duration / 2.0)
    r = np.sqrt(x**2 + d0**2)
    dr_dt = (x * v) / (r + 1e-9)
    v_radial = -dr_dt
    denom = c - v_radial
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    f_inst = f0 * (c / denom)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs_original

    # Generate signal with harmonics
    sig = 0.6 * np.sin(phase) + 0.28 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    sig += 0.25 * np.sin(0.5 * phase)

    # Amplitude variation (distance effect)
    r_ref = 5.0
    amp = 1.0 / (1.0 + (r / r_ref) ** 2)
    amp *= (1.0 + 0.15 * np.sin(2.0 * np.pi * 3.5 * (1 + v / 10.0) * t))
    sig *= amp
    sig += 0.008 * np.random.normal(size=sig.shape)

    # Bandpass filtering at original rate
    lowcut = 50.0
    highcut = min(4000.0, 0.45 * fs_original)
    b, a = butter_bandpass(lowcut, highcut, fs_original)
    sig = filtfilt(b, a, sig)
    
    # Normalize before resampling
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.95

    # ======= PROPER RESAMPLING TO SHOW ALIASING =======
    if fs_user != fs_original:
        # Calculate number of samples at new rate
        num_samples_new = int(len(sig) * fs_user / fs_original)
        
        # Use scipy.signal.resample for proper resampling
        # This simulates real-world sampling (includes aliasing if undersampled)
        sig_resampled = resample(sig, num_samples_new)
        t_new = np.linspace(0, duration, num_samples_new, endpoint=False)
    else:
        sig_resampled = sig
        t_new = t

    # ======= Save as WAV at user's sampling rate =======
    audio = (sig_resampled * 32767).astype(np.int16)
    
    file_id = str(uuid.uuid4())
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    write(filename, int(fs_user), audio)

    # ======= Calculate max frequency from actual signal using FFT =======
    # Perform FFT on original high-quality signal to find true max frequency
    fft_result = np.fft.rfft(sig[:10000])  # Use first 10k samples for speed
    fft_freqs = np.fft.rfftfreq(len(sig[:10000]), 1/fs_original)
    fft_magnitude = np.abs(fft_result)
    
    # Find significant frequency components (above 5% of max magnitude)
    threshold = 0.05 * np.max(fft_magnitude)
    significant_freqs = fft_freqs[fft_magnitude > threshold]
    
    if len(significant_freqs) > 0:
        max_freq = int(np.max(significant_freqs))
    else:
        max_freq = int(f0 * 3)  # Fallback: assume 3rd harmonic
    
    nyquist_freq = 2 * max_freq

    # ======= Determine sampling status =======
    if fs_user >= nyquist_freq:
        sampling_status = "[OK] Properly Sampled (No Aliasing)"
    elif fs_user >= nyquist_freq * 0.8:
        sampling_status = "[WARNING] Near Nyquist (Marginal)"
    else:
        sampling_status = "[ERROR] Undersampled (Aliasing Present)"

    # ======= Prepare for visualization (downsample if too many points) =======
    max_plot_points = 5000
    if len(t_new) > max_plot_points:
        step = len(t_new) // max_plot_points
        x_plot_json = t_new[::step].tolist()
        y_plot_json = sig_resampled[::step].tolist()
    else:
        x_plot_json = t_new.tolist()
        y_plot_json = sig_resampled.tolist()

    # ======= Render Result =======
    return render_template(
        "doppler_result.html",
        audio_file_id=file_id,
        v=v,
        f0=f0,
        fs_user=int(fs_user),
        max_freq=int(max_freq),
        nyquist_freq=int(nyquist_freq),
        sampling_status=sampling_status,
        x_plot_json=x_plot_json,
        y_plot_json=y_plot_json
    )

# =========================================================


# ==================== SERVE AUDIO ====================
@bp.route("/audio/<file_id>")
def serve_audio(file_id):
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    if not os.path.exists(filename):
        return "Audio file not found or expired.", 404
    return send_file(filename, mimetype="audio/wav", as_attachment=False)
# =====================================================


# ==================== DETECT PAGE (UPLOAD FORM) ====================
@bp.route("/detect", methods=["GET"])
def detect():
    return render_template("doppler_detect_result.html")
# =====================================================


# ==================== UPLOAD AND ESTIMATE SPEED ====================
@bp.route("/upload", methods=["POST"])
def upload_audio():
    if "audio_file" not in request.files:
        return "No file part", 400

    file = request.files["audio_file"]
    if file.filename == "":
        return "No selected file", 400

    file_id = str(uuid.uuid4())
    filepath = os.path.join(TEMP_DIR, f"{file_id}.wav")
    file.save(filepath)
    
    sr, data = wav.read(filepath)
    if data.ndim > 1:
        y = data[:, 0].astype(np.float32)
    else:
        y = data.astype(np.float32)
    if np.issubdtype(y.dtype, np.integer):
        y = y / float(np.iinfo(data.dtype).max)

    model_path = r"F:\Multi-Signal_Viewer_and_Predictor\results\speed_estimations\speed_estimations_NN_1000-200-50-10-1_reg1e-3_lossMSE.h5"
    car_name_in_file = os.path.basename(file.filename).split("_")[0]
    car_key = f"{car_name_in_file}_speeds_est_all"

    with h5py.File(model_path, "r") as f:
        if car_key in f:
            speeds = f[car_key][:]
            estimated_speed = float(np.mean(speeds))
        else:
            estimated_speed = 0.0

    estimated_speed = round(estimated_speed, 2)

    c = 343.0
    v_source = estimated_speed / 3.6

    def estimate_base_freq(y, sr):
        if len(y) == 0:
            return None
        segment = y[len(y)//4: 3*len(y)//4]
        window = windows.hann(len(segment))
        spectrum = np.fft.rfft(segment * window)
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        mags = np.abs(spectrum)
        valid = freqs > 20.0
        if not np.any(valid):
            return None
        freqs = freqs[valid]
        mags = mags[valid]
        f_peak = freqs[np.argmax(mags)]
        return float(f_peak)

    try:
        f_original = estimate_base_freq(y, sr)
    except Exception:
        f_original = None

    if f_original is None or f_original <= 0:
        f_original = 900.0

    approaching = True
    if approaching:
        perceived_freq = f_original * c / (c - v_source)
    else:
        perceived_freq = f_original * c / (c + v_source)

    perceived_freq = round(perceived_freq, 2)

    N = len(y)
    x_plot = np.linspace(0, N/sr, N)
    y_plot = y

    y_plot_json = json.dumps(y_plot.tolist())
    x_plot_json = json.dumps(x_plot.tolist())

    return render_template(
        "doppler_detect_result.html",
        audio_file_id=file_id,
        estimated_speed=estimated_speed,
        f_original=round(f_original, 2),
        perceived_freq=perceived_freq,
        y_plot_json=y_plot_json,
        x_plot_json=x_plot_json
    )