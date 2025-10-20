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

# ============================================================
# Blueprint Setup
# ============================================================
bp = Blueprint("doppler", __name__, template_folder="../templates")

c = 343.0  # Speed of sound (m/s)
TEMP_DIR = tempfile.gettempdir()


# ============================================================
# ROUTE: Main Doppler Page
# ============================================================
@bp.route("/")
def index():
    return render_template("doppler.html")


# ============================================================
# ROUTE: Generate Doppler Sound
# ============================================================
@bp.route("/generate", methods=["POST"])
def generate():
    try:
        # ---------- Get Inputs ----------
        v = float(request.form.get("velocity", 20.0))
        f0 = float(request.form.get("frequency", 440.0))
        fs_user = float(request.form.get("sampling_freq", 2000))
        d0 = float(request.form.get("distance", 10.0))
        duration = float(request.form.get("duration", 6.0))

        print(f"Generating: v={v}, f0={f0}, fs={fs_user}")

        # ---------- Sampling ----------
        fs_user = max(500, min(fs_user, 48000))
        fs = int(fs_user)
        fs_high = 44100  # Always generate high-quality reference
        N_high = int(duration * fs_high)
        t_high = np.linspace(0, duration, N_high, endpoint=False)

        # ---------- Doppler Model ----------
        x0 = -50.0
        x = x0 + v * t_high
        r = np.sqrt(x**2 + d0**2)

        v_radial = np.where(r > 1e-9, (x * v) / r, 0)
        f_inst = f0 * c / (c - v_radial)
        f_inst = np.clip(f_inst, 50, 20000)
        phase = 2 * np.pi * np.cumsum(f_inst) / fs_high

        # ---------- Generate Realistic Car Horn ----------
        phase1 = 2 * np.pi * np.cumsum(f_inst) / fs_high
        phase2 = 2 * np.pi * np.cumsum(f_inst * 1.26) / fs_high
        phase3 = 2 * np.pi * np.cumsum(f_inst * 1.5) / fs_high

        sig_high = (
            0.6 * np.sin(phase1)
            + 0.4 * np.sin(phase2)
            + 0.3 * np.sin(phase3)
            + 0.2 * np.sin(2 * phase1)
            + 0.1 * np.sin(3 * phase1)
        )

        # Add environmental noise
        sig_high += np.random.normal(0, 0.02, len(sig_high))

        # Amplitude based on distance
        base_amp = 1.0 / (r + 1.0) ** 2
        base_amp = base_amp / np.max(base_amp) * 3.0
        base_amp = np.clip(base_amp, 0.05, 3.0)
        sig_high *= base_amp

        # Attack/decay envelope
        attack_time, decay_time = 0.15, 0.3
        attack_env = np.linspace(0, 1, int(attack_time * fs_high)) ** 0.5
        decay_env = np.linspace(1, 0, int(decay_time * fs_high)) ** 0.7
        sig_high[: len(attack_env)] *= attack_env
        sig_high[-len(decay_env):] *= decay_env

        # Add car “buzz”
        sig_high += 0.05 * np.sin(2 * np.pi * 20 * t_high) * base_amp
        sig_high = sig_high / (np.max(np.abs(sig_high)) + 1e-9) * 0.8

        # ---------- Downsampling / Aliasing Simulation ----------
        if fs != fs_high:
            decimation_factor = max(1, fs_high // fs)
            if decimation_factor > 1:
                sig_aliased = sig_high[::decimation_factor]
                target_length = int(duration * fs)
                sig_aliased = (
                    sig_aliased[:target_length]
                    if len(sig_aliased) > target_length
                    else np.pad(sig_aliased, (0, target_length - len(sig_aliased)))
                )
            else:
                sig_aliased = resample(sig_high, int(duration * fs))
        else:
            sig_aliased = sig_high

        t = np.linspace(0, duration, len(sig_aliased), endpoint=False)

        # Add slight noise for very low fs
        if fs < 4000:
            sig_aliased = sig_aliased + np.random.normal(0, 0.03, len(sig_aliased))
            sig_aliased = sig_aliased / (np.max(np.abs(sig_aliased)) + 1e-9) * 0.8

        print(f"Generated {len(sig_aliased)} samples at {fs} Hz")

        # ---------- Sampling Metrics ----------
        max_freq_inst = np.max(np.abs(f_inst))
        max_freq = int(max_freq_inst * 4)
        nyquist_freq = 2 * max_freq

        if fs_user >= nyquist_freq:
            sampling_status, status_class = "✓ Properly Sampled (No Aliasing)", "good"
        elif fs_user >= max_freq:
            sampling_status, status_class = "⚠️ Near Nyquist (Marginal)", "warning"
        else:
            sampling_status, status_class = "❌ Undersampled (Aliasing Present)", "danger"

        print(
            f"Max freq: {max_freq} Hz, Nyquist: {nyquist_freq} Hz, Status: {sampling_status}"
        )

        # ---------- Save Audio ----------
        audio_fs = max(8000, fs)
        sig_audio = (
            resample(sig_aliased, int(duration * audio_fs))
            if audio_fs != fs
            else sig_aliased
        )
        audio = np.clip(sig_audio * 32767, -32767, 32767).astype(np.int16)

        file_id = str(uuid.uuid4())
        filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
        write(filename, audio_fs, audio)
        print(f"Saved audio: {filename} at {audio_fs} Hz")

        # ---------- Prepare Visualization ----------
        max_points = 3000
        stride = max(1, len(sig_aliased) // max_points)
        t_plot = t[::stride][:max_points]
        sig_plot = sig_aliased[::stride][:max_points]

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
            x_plot_json=json.dumps(t_plot.tolist()),
            y_plot_json=json.dumps(sig_plot.tolist()),
            f_approaching=round(np.max(f_inst), 1),
            f_receding=round(np.min(f_inst), 1),
            duration=round(duration, 1),
        )

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return f"Error generating sound: {str(e)}", 500


# ============================================================
# ROUTE: Serve Audio File
# ============================================================
@bp.route("/audio/<file_id>")
def serve_audio(file_id):
    filename = os.path.join(TEMP_DIR, f"{file_id}.wav")
    if not os.path.exists(filename):
        return "Audio file not found or expired.", 404
    return send_file(filename, mimetype="audio/wav", as_attachment=False)


# ============================================================
# ROUTE: Detect Page
# ============================================================
@bp.route("/detect", methods=["GET"])
def detect():
    return render_template("doppler_detect_result.html")


# ============================================================
# ROUTE: Upload and Estimate Speed
# ==================== UPLOAD AND ESTIMATE SPEED ====================
# ... (keep all your code the same until upload_audio function)
# ... (keep all your code the same until upload_audio function)

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

        # Read audio file
        sr, data = wav.read(filepath)
        
        # Process audio data
        if data.ndim > 1:
            y = data[:, 0].astype(np.float32)
        else:
            y = data.astype(np.float32)
            
        if np.issubdtype(y.dtype, np.integer):
            y = y / float(np.iinfo(data.dtype).max)

        # Normalize
        y = y / (np.max(np.abs(y)) + 1e-9)

        # ---------- Load Model and Estimate Speed ----------
        try:
            model_path = r"F:\Multi-Signal_Viewer_and_Predictor\results\speed_estimations\speed_estimations_NN_1000-200-50-10-1_reg1e-3_lossMSE.h5"
            car_name_in_file = os.path.basename(file.filename).split("_")[0]
            car_key = f"{car_name_in_file}_speeds_est_all"

            with h5py.File(model_path, "r") as f:
                estimated_speed = float(np.mean(f[car_key][:])) if car_key in f else 50.0
        except Exception as e:
            print(f"Model loading failed: {e}")
            estimated_speed = 50.0

        estimated_speed = round(estimated_speed, 2)

        # ---------- Frequency Estimation ----------
        def estimate_base_freq(y, sr):
            if len(y) == 0:
                return 440.0
                
            start_idx = len(y) // 4
            end_idx = 3 * len(y) // 4
            segment = y[start_idx:end_idx]
            
            if len(segment) == 0:
                return 440.0
                
            window = windows.hann(len(segment))
            spectrum = np.fft.rfft(segment * window)
            freqs = np.fft.rfftfreq(len(segment), 1 / sr)
            mags = np.abs(spectrum)
            
            valid_indices = (freqs > 50) & (freqs < 2000)
            if not np.any(valid_indices):
                return 440.0
                
            valid_freqs = freqs[valid_indices]
            valid_mags = mags[valid_indices]
            f_peak = valid_freqs[np.argmax(valid_mags)]
            
            return float(f_peak)

        try:
            f_original = estimate_base_freq(y, sr)
        except Exception as e:
            print(f"Frequency estimation failed: {e}")
            f_original = 440.0

        f_original = round(f_original, 2)

        # ---------- Prepare Plot Data ----------
        N = len(y)
        duration = N / sr
        
        max_plot_points = 1000
        if N > max_plot_points:
            indices = np.linspace(0, N-1, max_plot_points, dtype=int)
            y_plot = y[indices]
        else:
            y_plot = y

        print(f"=== UPLOAD SUCCESS ===")
        print(f"Audio: {N} samples, {duration:.2f}s, {sr} Hz")
        print(f"Plot: {len(y_plot)} amplitude points")
        print(f"Detection: speed={estimated_speed} km/h, freq={f_original} Hz")
        print(f"Y data sample: {y_plot[:5]}")

        # ✅ FIX 1: Don't double-encode JSON - convert to list directly
        return render_template(
            "doppler_detect_result.html",
            audio_file_id=file_id,
            estimated_speed=estimated_speed,
            f_original=f_original,
            y_plot_list=y_plot.tolist(),  # ✅ Pass as Python list
            sr=sr
        )

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error processing file: {str(e)}", 500


# ✅ FIX 3: Actually resample the ORIGINAL uploaded audio
@bp.route("/resample_uploaded_audio", methods=["POST"])
def resample_uploaded_audio():
    try:
        print("=== RESAMPLE_UPLOADED_AUDIO CALLED ===")
        
        data = request.get_json()
        print(f"Received data: {data}")
        
        target_fs = int(data["target_fs"])
        audio_file_id = data["audio_file_id"]

        print(f"Resampling original file {audio_file_id} to {target_fs} Hz")

        # Read the ORIGINAL uploaded file
        original_filepath = os.path.join(TEMP_DIR, f"{audio_file_id}.wav")
        
        if not os.path.exists(original_filepath):
            raise Exception("Original file not found")
        
        sr_orig, data_orig = wav.read(original_filepath)
        
        # Convert to mono if stereo
        if data_orig.ndim > 1:
            y_orig = data_orig[:, 0].astype(np.float32)
        else:
            y_orig = data_orig.astype(np.float32)
            
        # Normalize to -1 to 1
        if np.issubdtype(y_orig.dtype, np.integer):
            y_orig = y_orig / float(np.iinfo(data_orig.dtype).max)
        
        y_orig = y_orig / (np.max(np.abs(y_orig)) + 1e-9)
        
        print(f"Original: {len(y_orig)} samples at {sr_orig} Hz")
        
        # ✅ RESAMPLE to target frequency
        duration = len(y_orig) / sr_orig
        num_samples_new = int(duration * target_fs)
        
        if target_fs != sr_orig:
            y_resampled = resample(y_orig, num_samples_new)
        else:
            y_resampled = y_orig
        
        print(f"Resampled: {len(y_resampled)} samples at {target_fs} Hz")
        
        # ✅ FIX: For audio playback, upsample to at least 8000 Hz if needed
        audio_fs = max(8000, target_fs)  # Browsers need minimum 8kHz
        
        if audio_fs != target_fs:
            num_samples_audio = int(duration * audio_fs)
            y_audio = resample(y_resampled, num_samples_audio)
            print(f"Upsampled for playback: {audio_fs} Hz")
        else:
            y_audio = y_resampled
        
        # Save audio file at playable sampling rate
        audio_data = np.clip(y_audio * 32767, -32767, 32767).astype(np.int16)
        
        resampled_file_id = str(uuid.uuid4())
        resampled_filename = os.path.join(TEMP_DIR, f"{resampled_file_id}.wav")
        write(resampled_filename, audio_fs, audio_data)
        
        print(f"✅ Saved audio: {resampled_filename} at {audio_fs} Hz (display: {target_fs} Hz)")

        # Prepare plot data - use the TRUE resampled signal (not upsampled)
        max_plot_points = 800
        if len(y_resampled) > max_plot_points:
            stride = len(y_resampled) // max_plot_points
            y_plot = y_resampled[::stride][:max_plot_points]
        else:
            y_plot = y_resampled

        print(f"✅ Returning {len(y_plot)} plot points")

        return jsonify({
            "success": True,
            "resampled_file_id": resampled_file_id,
            "y_plot": y_plot.tolist(),
            "actual_fs_used": target_fs,  # The TRUE sampling rate
            "audio_fs": audio_fs,  # The playback sampling rate
            "original_length": len(y_orig),
            "resampled_length": len(y_resampled)
        })

    except Exception as e:
        print(f"❌ Error in resample_uploaded_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
# ============================================================
# ==================== GENERATE DOWNSAMPLED AUDIO ====================
@bp.route("/generate_downsampled_audio", methods=["POST"])
def generate_downsampled_audio():
    try:
        print("=== GENERATE_DOWNSAMPLED_AUDIO CALLED ===")
        
        data = request.get_json()
        print(f"Received data: {data}")
        
        target_fs = int(data["target_fs"])
        estimated_speed = float(data["estimated_speed"])
        estimated_freq = float(data["estimated_freq"])

        print(f"Generating: speed={estimated_speed}, freq={estimated_freq}, fs={target_fs}")

        # Generate Doppler car sound
        duration = 4.0
        c = 343.0
        d0 = 10.0
        
        # Create time array
        t = np.linspace(0, duration, int(duration * target_fs), endpoint=False)
        
        # Car trajectory - passes closest at t = duration/2
        t_closest = duration / 2.0
        x0 = -50.0
        x = x0 + (estimated_speed / 3.6) * (t - t_closest)
        
        # Distance from observer
        r = np.sqrt(x**2 + d0**2)
        
        # Radial velocity and instantaneous frequency
        v_radial = np.zeros_like(x)
        for i in range(len(x)):
            if r[i] > 1e-9:
                v_radial[i] = (x[i] * (estimated_speed / 3.6)) / r[i]
        
        f_inst = estimated_freq * c / (c - v_radial)
        f_inst = np.clip(f_inst, 50, 20000)
        
        # Generate phase and signal
        phase = 2.0 * np.pi * np.cumsum(f_inst) / target_fs
        
        # Create horn sound with harmonics
        signal = (0.6 * np.sin(phase) +
                  0.3 * np.sin(2 * phase) + 
                  0.1 * np.sin(3 * phase))
        
        # Amplitude envelope based on distance
        amp = 1.0 / (1.0 + r/20.0)
        amp = amp / np.max(amp)
        signal *= amp
        
        # Add aliasing artifacts at low sampling rates
        if target_fs < 2000:
            noise_freq = min(target_fs * 0.8, 1000)
            signal += 0.1 * np.sin(2 * np.pi * noise_freq * t)
        
        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.8

        # Convert to audio
        audio_data = np.int16(signal * 32767)
        
        # Save file
        generated_file_id = str(uuid.uuid4())
        generated_filename = os.path.join(TEMP_DIR, f"{generated_file_id}.wav")
        write(generated_filename, target_fs, audio_data)
        
        print(f"✅ Saved audio: {generated_filename} at {target_fs} Hz")

        # Prepare plot data
        max_plot_points = 800
        if len(t) > max_plot_points:
            stride = len(t) // max_plot_points
            t_plot = t[::stride]
            y_plot = signal[::stride]
        else:
            t_plot = t
            y_plot = signal

        # Ensure we don't exceed max points
        if len(t_plot) > max_plot_points:
            t_plot = t_plot[:max_plot_points]
            y_plot = y_plot[:max_plot_points]

        print(f"✅ Returning {len(t_plot)} plot points")

        return jsonify({
            "success": True,
            "generated_file_id": generated_file_id,
            "x_plot": t_plot.tolist(),
            "y_plot": y_plot.tolist(),
            "actual_fs_used": target_fs,
        })

    except Exception as e:
        print(f"❌ Error in generate_downsampled_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

# FUNCTION: Generate Doppler Car Sound for Detection
# ============================================================
def generate_car_sound_from_detection(speed, frequency, fs, duration):
    """Generate a car Doppler sound based on detection results and sampling rate"""
    c = 343.0
    d0 = 10.0
    
    # Create time array at the specified sampling rate
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Car trajectory - passes closest at t = duration/2
    t_closest = duration / 2.0
    x0 = -50.0
    x = x0 + (speed / 3.6) * (t - t_closest)
    
    # Distance from observer
    r = np.sqrt(x**2 + d0**2)
    
    # Radial velocity and instantaneous frequency
    v_radial = np.zeros_like(x)
    for i in range(len(x)):
        if r[i] > 1e-9:
            v_radial[i] = (x[i] * (speed / 3.6)) / r[i]
    
    f_inst = frequency * c / (c - v_radial)
    f_inst = np.clip(f_inst, 50, 20000)
    
    # Generate phase and signal
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    
    # Create horn sound with harmonics
    signal = (0.6 * np.sin(phase) +
              0.3 * np.sin(2 * phase) + 
              0.1 * np.sin(3 * phase))
    
    # Amplitude envelope based on distance
    amp = 1.0 / (1.0 + r/20.0)  # Softer attenuation
    amp = amp / np.max(amp)
    signal *= amp
    
    # Add aliasing artifacts at low sampling rates
    if fs < 2000:
        # Add some high-frequency noise to simulate aliasing
        noise_freq = min(fs * 0.8, 1000)  # Frequency that will alias
        signal += 0.1 * np.sin(2 * np.pi * noise_freq * t)
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.8
    
    return signal, t















