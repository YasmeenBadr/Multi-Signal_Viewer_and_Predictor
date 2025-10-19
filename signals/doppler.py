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

bp = Blueprint("doppler",__name__, template_folder="../templates")

c = 343.0
TEMP_DIR = tempfile.gettempdir()

# ==================== MAIN PAGE ====================
@bp.route("/")
def index():
    return render_template("doppler.html")

# ==================== GENERATE SOUND ====================
@bp.route("/generate", methods=["POST"])
def generate():
    try:
        # Get user inputs
        v = float(request.form.get("velocity", 20.0))
        f0 = float(request.form.get("frequency", 440.0))
        fs_user = float(request.form.get("sampling_freq", 2000))
        d0 = float(request.form.get("distance", 10.0))
        duration = float(request.form.get("duration", 6.0))

        print(f"Generating: v={v}, f0={f0}, fs={fs_user}")

        # Set reasonable limits for audio playback
        fs_user = max(500, min(fs_user, 48000))
        fs = int(fs_user)
        
        # ALWAYS generate at high quality first for comparison
        fs_high = 44100
        N_high = int(duration * fs_high)
        t_high = np.linspace(0, duration, N_high, endpoint=False)

        # ==================== DOPPLER SHIFT MODEL ====================
        t_closest = duration / 2.0
        x0 = -50.0
        x = x0 + v * t_high
        
        r = np.sqrt(x**2 + d0**2)
        
        v_radial = np.zeros_like(x)
        for i in range(len(x)):
            if r[i] > 1e-9:
                v_radial[i] = (x[i] * v) / r[i]
            else:
                v_radial[i] = 0
        
        f_inst = f0 * c / (c - v_radial)
        f_inst = np.clip(f_inst, 50, 20000)
        
        phase = 2.0 * np.pi * np.cumsum(f_inst) / fs_high

        # ==================== REALISTIC CAR HORN GENERATION ====================
        phase1 = 2.0 * np.pi * np.cumsum(f_inst) / fs_high
        phase2 = 2.0 * np.pi * np.cumsum(f_inst * 1.26) / fs_high  
        phase3 = 2.0 * np.pi * np.cumsum(f_inst * 1.5) / fs_high

        sig_high = (0.6 * np.sin(phase1) +
                    0.4 * np.sin(phase2) +
                    0.3 * np.sin(phase3) +
                    0.2 * np.sin(2 * phase1) +
                    0.1 * np.sin(3 * phase1))

        noise_level = 0.02
        environment_noise = np.random.normal(0, noise_level, len(sig_high))
        sig_high += environment_noise

        base_amp = 1.0 / (r + 1.0)**2
        base_amp = base_amp / np.max(base_amp) * 3.0
        base_amp = np.clip(base_amp, 0.05, 3.0)

        sig_high *= base_amp

        attack_time = 0.15
        decay_time = 0.3
        attack_samples = int(attack_time * fs_high)
        decay_samples = int(decay_time * fs_high)
        attack_env = np.linspace(0, 1, attack_samples)**0.5
        decay_env = np.linspace(1, 0, decay_samples)**0.7
        sig_high[:attack_samples] *= attack_env
        sig_high[-decay_samples:] *= decay_env

        buzz_freq = 20
        buzz = 0.05 * np.sin(2 * np.pi * buzz_freq * t_high)
        sig_high += buzz * base_amp

        sig_high = sig_high / (np.max(np.abs(sig_high)) + 1e-9) * 0.8

        # ==================== GENERATE ALIASED SIGNAL ====================
        if fs != fs_high:
            decimation_factor = max(1, fs_high // fs)
            if decimation_factor > 1:
                sig_aliased = sig_high[::decimation_factor]
                target_length = int(duration * fs)
                if len(sig_aliased) > target_length:
                    sig_aliased = sig_aliased[:target_length]
                else:
                    sig_aliased = np.pad(sig_aliased, (0, target_length - len(sig_aliased)))
            else:
                sig_aliased = resample(sig_high, int(duration * fs))
        else:
            sig_aliased = sig_high
        
        t = np.linspace(0, duration, len(sig_aliased), endpoint=False)
        if fs < 4000:
            noise = np.random.normal(0, 0.03, len(sig_aliased))
            sig_aliased = sig_aliased + noise
            sig_aliased = sig_aliased / (np.max(np.abs(sig_aliased)) + 1e-9) * 0.8
        
        print(f"Generated {len(sig_aliased)} samples at {fs} Hz")

        # ==================== CALCULATE METRICS ====================
        max_freq_inst = np.max(np.abs(f_inst))
        max_freq = int(max_freq_inst * 4)
        nyquist_freq = 2 * max_freq
        
        if fs_user >= nyquist_freq:
            sampling_status = "✓ Properly Sampled (No Aliasing)"
            status_class = "good"
        elif fs_user >= max_freq:
            sampling_status = "⚠️ Near Nyquist (Marginal)"
            status_class = "warning"
        else:
            sampling_status = "❌ Undersampled (Aliasing Present)"
            status_class = "danger"
        
        print(f"Max freq: {max_freq} Hz, Nyquist: {nyquist_freq} Hz, Status: {sampling_status}")

        # ==================== SAVE AUDIO FILE ====================
        audio_fs = max(8000, fs)
        if audio_fs != fs:
            num_samples_audio = int(duration * audio_fs)
            sig_audio = resample(sig_aliased, num_samples_audio)
        else:
            sig_audio = sig_aliased
            
        audio = np.clip(sig_audio * 32767, -32767, 32767).astype(np.int16)
        
        file_id = str(uuid.uuid4())
        filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
        write(filename, audio_fs, audio)
        print(f"Saved audio: {filename} at {audio_fs} Hz")

        # ==================== PREPARE VISUALIZATION ====================
        t_vis = t
        sig_vis = sig_aliased

        max_points = 3000
        if len(t_vis) > max_points:
            stride = len(t_vis) // max_points
            if stride < 1:
                stride = 1
            t_plot = t_vis[::stride]
            sig_plot = sig_vis[::stride]
            if len(t_plot) > max_points:
                t_plot = t_plot[:max_points]
                sig_plot = sig_plot[:max_points]
        else:
            t_plot = t_vis
            sig_plot = sig_vis

        x_list = [float(x) for x in t_plot]
        y_list = [float(y) for y in sig_plot]

        print(f"Full signal visualization: {len(x_list)} points over {duration} seconds")

        return render_template(
            "doppler_result.html",
            audio_file_id=file_id,
            v=round(v, 2),
            f0=round(f0, 2),
            fs_user=int(fs_user),
            audio_fs=int(audio_fs),
            max_freq=max_freq,
            nyquist_freq=nyquist_freq,
            sampling_status=sampling_status,
            status_class=status_class,
            x_plot_json=json.dumps(x_list),
            y_plot_json=json.dumps(y_list),
            f_approaching=round(np.max(f_inst), 1),
            f_receding=round(np.min(f_inst), 1),
            duration=round(duration, 1)
        )
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating sound: {str(e)}", 500

# ==================== SERVE AUDIO ====================
@bp.route("/audio/<file_id>")
def serve_audio(file_id):
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    if not os.path.exists(filename):
        return "Audio file not found or expired.", 404
    return send_file(filename, mimetype="audio/wav", as_attachment=False)

# ==================== DETECT PAGE ====================
@bp.route("/detect", methods=["GET"])
def detect():
    return render_template("doppler_detect_result.html")

# ==================== UPLOAD AND ESTIMATE SPEED ====================
@bp.route("/upload", methods=["POST"])
def upload_audio():
    try:
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

        # Use your actual AI model here
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
        # FIX: Use proper downsampling for the original signal display
        max_plot_points = 1000
        if N > max_plot_points:
            step = N // max_plot_points
            x_plot = np.linspace(0, N/sr, N)[::step]
            y_plot = y[::step]
        else:
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
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return f"Error processing file: {str(e)}", 500
    # ==================== GENERATE DOWNSAMPLED AUDIO ====================
@bp.route("/generate_downsampled_audio", methods=["POST"])
def generate_downsampled_audio():
    try:
        data = request.get_json()
        target_fs = int(data['target_fs'])
        estimated_speed = float(data['estimated_speed'])
        estimated_freq = float(data['estimated_freq'])
        
        # Generate NEW car sound using the detected parameters and selected sampling rate
        # Use lower sampling rates to demonstrate aliasing
        duration = 6.0
        
        # FIX: Use the exact target_fs without minimum limits to demonstrate aliasing
        car_sound, t = generate_car_sound_from_detection(
            estimated_speed, 
            estimated_freq, 
            target_fs,  # Use the exact sampling rate user selected
            duration
        )
        
        # Convert to WAV format
        audio_data = np.clip(car_sound * 32767, -32767, 32767).astype(np.int16)
        
        # Save generated audio
        generated_file_id = str(uuid.uuid4())
        generated_filename = os.path.join(TEMP_DIR, f"{generated_file_id}.wav")
        write(generated_filename, target_fs, audio_data)
        
        # Prepare plot data
        max_plot_points = 1000
        if len(t) > max_plot_points:
            stride = len(t) // max_plot_points
            t_plot = t[::stride]
            y_plot = car_sound[::stride]
        else:
            t_plot = t
            y_plot = car_sound
        
        return jsonify({
            'success': True,
            'generated_file_id': generated_file_id,
            'x_plot': t_plot.tolist(),
            'y_plot': y_plot.tolist(),
            'actual_fs_used': target_fs
        })
        
    except Exception as e:
        print(f"Error in generate_downsampled_audio: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_car_sound_from_detection(speed, frequency, fs, duration):
    """Generate a car Doppler sound based on detection results and sampling rate"""
    c = 343.0
    d0 = 10.0
    
    # Create time array at the specified sampling rate
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Car trajectory
    x0 = -50.0
    x = x0 + (speed / 3.6) * t
    
    # Distance from observer
    r = np.sqrt(x**2 + d0**2)
    
    # Radial velocity and instantaneous frequency
    v_radial = (x * (speed / 3.6)) / (r + 1e-9)
    f_inst = frequency * c / (c - v_radial)
    f_inst = np.clip(f_inst, 50, 10000)
    
    # Generate phase and signal - CRITICAL: This is where aliasing happens!
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    
    # Create horn sound with harmonics
    signal = (0.6 * np.sin(phase) +
              0.4 * np.sin(2 * phase) +
              0.2 * np.sin(3 * phase))
    
    # Amplitude envelope
    amp = 1.0 / (r + 1.0)**2
    amp = amp / np.max(amp) * 2.0
    signal *= amp
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.8
    
    return signal, t