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

# ============================================================
# Configuration
# ============================================================
bp = Blueprint("doppler", __name__, template_folder="../templates")
c = 343.0  # Speed of sound (m/s)
TEMP_DIR = tempfile.gettempdir()
MODEL_PATH = r"F:\Multi-Signal_Viewer_and_Predictor\results\speed_estimations\speed_estimations_NN_1000-200-50-10-1_reg1e-3_lossMSE.h5"


# ============================================================
# Helper Functions
# ============================================================
def normalize_audio(data):
    """Normalize audio data to [-1, 1] range"""
    y = data[:, 0] if data.ndim > 1 else data
    y = y.astype(np.float32)
    if np.issubdtype(y.dtype, np.integer):
        y = y / float(np.iinfo(data.dtype).max)
    return y / (np.max(np.abs(y)) + 1e-9)


def generate_doppler_signal(v, f0, fs, duration, d0=10.0):
    """Generate Doppler-shifted car horn signal"""
    N = int(duration * fs)
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Car trajectory and Doppler shift
    x = -50.0 + v * t
    r = np.sqrt(x**2 + d0**2)
    v_radial = np.where(r > 1e-9, (x * v) / r, 0)
    f_inst = np.clip(f0 * c / (c - v_radial), 50, 20000)
    
    # Generate harmonics
    phase_base = 2 * np.pi * np.cumsum(f_inst) / fs
    signal = (0.6 * np.sin(phase_base) +
              0.4 * np.sin(2 * np.pi * np.cumsum(f_inst * 1.26) / fs) +
              0.3 * np.sin(2 * np.pi * np.cumsum(f_inst * 1.5) / fs) +
              0.2 * np.sin(2 * phase_base) +
              0.1 * np.sin(3 * phase_base))
    
    # Apply amplitude envelope
    base_amp = np.clip((1.0 / (r + 1.0)**2) / np.max(1.0 / (r + 1.0)**2) * 3.0, 0.05, 3.0)
    signal *= base_amp
    
    # Add environmental effects
    signal += np.random.normal(0, 0.02, len(signal))
    signal += 0.05 * np.sin(2 * np.pi * 20 * t) * base_amp
    
    # Attack/decay envelope
    attack_samples = int(0.15 * fs)
    decay_samples = int(0.3 * fs)
    signal[:attack_samples] *= np.linspace(0, 1, attack_samples)**0.5
    signal[-decay_samples:] *= np.linspace(1, 0, decay_samples)**0.7
    
    return signal / (np.max(np.abs(signal)) + 1e-9) * 0.8, t, f_inst


def downsample_signal(signal, fs_high, fs_target, duration):
    """Downsample signal with optional decimation or resampling"""
    if fs_target == fs_high:
        return signal
    # Use aliasing decimation when reducing sample rate, otherwise high-quality resample
    if fs_target < fs_high:
        sig = decimate_with_aliasing(signal, fs_high, fs_target)
    else:
        sig = resample_signal(signal, fs_high, fs_target, method="scipy", aa=True)
    # Conform to requested duration length
    target_length = int(duration * fs_target)
    if len(sig) > target_length:
        return sig[:target_length]
    if len(sig) < target_length:
        return np.pad(sig, (0, target_length - len(sig)))
    return sig


def estimate_frequency(y, sr):
    """Estimate base frequency using FFT"""
    if len(y) == 0:
        return 440.0
    
    segment = y[len(y)//4: 3*len(y)//4]
    if len(segment) == 0:
        return 440.0
    
    spectrum = np.fft.rfft(segment * windows.hann(len(segment)))
    freqs = np.fft.rfftfreq(len(segment), 1 / sr)
    mags = np.abs(spectrum)
    
    valid = (freqs > 50) & (freqs < 2000)
    return float(freqs[valid][np.argmax(mags[valid])]) if np.any(valid) else 440.0


def save_audio_file(signal, fs, duration, min_fs=8000):
    """Save audio file with minimum sampling rate for playback"""
    audio_fs = max(min_fs, fs)
    sig_audio = resample_signal(signal, fs, audio_fs, method="scipy", aa=True) if audio_fs != fs else signal
    audio_data = np.clip(sig_audio * 32767, -32767, 32767).astype(np.int16)
    
    file_id = str(uuid.uuid4())
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    write(filename, audio_fs, audio_data)
    return file_id, audio_fs


# ============================================================
# Routes
# ============================================================
@bp.route("/")
def index():
    return render_template("doppler.html")


@bp.route("/generate", methods=["POST"])
def generate():
    try:
        v = float(request.form.get("velocity", 20.0))
        f0 = float(request.form.get("frequency", 440.0))
        fs_user = np.clip(float(request.form.get("sampling_freq", 2000)), 500, 48000)
        duration = float(request.form.get("duration", 6.0))
        d0 = float(request.form.get("distance", 10.0))
        
        print(f"Generating: v={v}, f0={f0}, fs={fs_user}")
        
        # Generate high-quality signal
        sig_high, _, f_inst = generate_doppler_signal(v, f0, 44100, duration, d0)
        
        # Downsample
        fs = int(fs_user)
        sig_aliased = downsample_signal(sig_high, 44100, fs, duration)
        
        # Add noise for low sampling rates
        if fs < 4000:
            sig_aliased += np.random.normal(0, 0.03, len(sig_aliased))
            sig_aliased = sig_aliased / (np.max(np.abs(sig_aliased)) + 1e-9) * 0.8
        
        # Calculate metrics
        max_freq = int(np.max(np.abs(f_inst)) * 4)
        nyquist_freq = 2 * max_freq
        
        if fs_user >= nyquist_freq:
            sampling_status, status_class = "✓ Properly Sampled (No Aliasing)", "good"
        elif fs_user >= max_freq:
            sampling_status, status_class = "⚠️ Near Nyquist (Marginal)", "warning"
        else:
            sampling_status, status_class = "❌ Undersampled (Aliasing Present)", "danger"
        
        # Save audio
        file_id, audio_fs = save_audio_file(sig_aliased, fs, duration)
        print(f"Generated {len(sig_aliased)} samples at {fs} Hz, saved at {audio_fs} Hz")
        
        # Prepare visualization
        t = np.linspace(0, duration, len(sig_aliased), endpoint=False)
        stride = max(1, len(sig_aliased) // 3000)
        
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
            f_approaching=round(np.max(f_inst), 1),
            f_receding=round(np.min(f_inst), 1),
            duration=round(duration, 1)
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", 500


@bp.route("/upload", methods=["POST"])
def upload_audio():
    try:
        if "audio_file" not in request.files or request.files["audio_file"].filename == "":
            return "No file selected", 400
        
        file = request.files["audio_file"]
        file_id = str(uuid.uuid4())
        filepath = os.path.join(TEMP_DIR, f"{file_id}.wav")
        file.save(filepath)
        
        # Read and normalize
        sr, data = wav_read(filepath)
        y = normalize_audio(data)
        
        # Estimate speed
        try:
            car_name = os.path.basename(file.filename).split("_")[0]
            with h5py.File(MODEL_PATH, "r") as f:
                estimated_speed = float(np.mean(f[f"{car_name}_speeds_est_all"][:])) if f"{car_name}_speeds_est_all" in f else 50.0
        except Exception as e:
            print(f"Model error: {e}")
            estimated_speed = 50.0
        
        # Estimate frequency
        f_original = round(estimate_frequency(y, sr), 2)
        estimated_speed = round(estimated_speed, 2)
        
        # Prepare plot data
        max_points = 1000
        indices = np.linspace(0, len(y)-1, min(len(y), max_points), dtype=int)
        y_plot = y[indices]
        
        print(f"Upload: {len(y)} samples, {len(y)/sr:.2f}s, {sr} Hz | Speed: {estimated_speed} km/h, Freq: {f_original} Hz")
        
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
    try:
        data = request.get_json()
        target_fs = int(data["target_fs"])
        audio_file_id = data["audio_file_id"]
        
        # Read original
        filepath = os.path.join(TEMP_DIR, f"{audio_file_id}.wav")
        if not os.path.exists(filepath):
            raise Exception("File not found")
        
        sr_orig, data_orig = wav_read(filepath)
        y_orig = normalize_audio(data_orig)
        
        # Resample
        duration = len(y_orig) / sr_orig
        y_resampled = resample_signal(y_orig, sr_orig, target_fs, method="scipy", aa=True) if target_fs != sr_orig else y_orig
        
        # Save for playback (minimum 8kHz)
        audio_fs = max(8000, target_fs)
        y_audio = resample_signal(y_resampled, target_fs, audio_fs, method="scipy", aa=True) if audio_fs != target_fs else y_resampled
        
        audio_data = np.clip(y_audio * 32767, -32767, 32767).astype(np.int16)
        resampled_id = str(uuid.uuid4())
        write(os.path.join(TEMP_DIR, f"{resampled_id}.wav"), audio_fs, audio_data)
        
        # Plot data
        stride = max(1, len(y_resampled) // 800)
        y_plot = y_resampled[::stride][:800]
        
        print(f"Resampled: {len(y_orig)} → {len(y_resampled)} samples at {target_fs} Hz (playback: {audio_fs} Hz)")
        
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
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    return send_file(filename, mimetype="audio/wav") if os.path.exists(filename) else ("File not found", 404)


@bp.route("/detect", methods=["GET"])
def detect():
    return render_template("doppler_detect_result.html")