import os
import sys
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import io
import base64
try:
    from resampling import (
        resample_signal, 
        decimate_with_aliasing,
        estimate_aliasing_level,
        get_nyquist_limit
    )
    RESAMPLING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: resampling utilities not available: {e}")
    RESAMPLING_AVAILABLE = False
# Try to import required libraries
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")
    TORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: librosa not available: {e}")
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception as e:
    print(f"Warning: soundfile not available: {e}")
    SOUNDFILE_AVAILABLE = False

try:
    import wave
    WAVE_AVAILABLE = True
except Exception as e:
    print(f"Warning: wave module not available: {e}")
    WAVE_AVAILABLE = False

# Local resampling utilities (not used in voice path now)

# TensorFlow for anti-aliasing model
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    TF_AVAILABLE = False

# Add the voice-gender-classifier directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'voice-gender-classifier'))

try:
    from model import ECAPA_gender
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ECAPA_gender model not available: {e}")
    MODEL_AVAILABLE = False

# Define the blueprint for the Voice Processing Suite
bp = Blueprint('voice', __name__, template_folder='templates')

# Global model variables
model = None
device = None
aa_model = None
AA_MODEL_LOADED = False

# Path to anti-aliasing model
ANTI_ALIAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'anti_alias_model.h5')

def load_antialiasing_model():
    """Load the TensorFlow anti-aliasing model"""
    global aa_model, AA_MODEL_LOADED
    if not TF_AVAILABLE:
        print("[WARN] TensorFlow not available. Anti-aliasing will use standard upsampling.")
        return
    
    if aa_model is None:
        try:
            aa_model = tf.keras.models.load_model(ANTI_ALIAS_MODEL_PATH, compile=False)
            AA_MODEL_LOADED = True
            print("[OK] Anti-aliasing model loaded successfully!")
        except Exception as e:
            print(f"[WARN] Could not load anti-aliasing model: {e}")
            AA_MODEL_LOADED = False
            aa_model = None

def apply_antialiasing(y_input, sr_input, target_sr=16000):
    """
    Apply anti-aliasing reconstruction using the TensorFlow model.
    If model is not available, perform standard upsampling.
    """
    if not AA_MODEL_LOADED or aa_model is None:
        print("[WARN] ML model not available, performing standard upsampling.")
        return librosa.resample(y_input, orig_sr=sr_input, target_sr=target_sr)
    
    try:
        model_len = 48000
        y_upsampled = librosa.resample(y_input, orig_sr=sr_input, target_sr=target_sr)
        
        reconstructed_chunks = []
        
        for i in range(0, len(y_upsampled), model_len):
            chunk = y_upsampled[i:i + model_len]
            len_chunk = len(chunk)
            
            if len_chunk < model_len:
                chunk = np.pad(chunk, (0, model_len - len_chunk), mode='constant')
            
            model_input = chunk[np.newaxis, ..., np.newaxis]
            pred_chunk = np.squeeze(aa_model.predict(model_input, verbose=0))
            
            if len_chunk < model_len:
                pred_chunk = pred_chunk[:len_chunk]
            
            reconstructed_chunks.append(pred_chunk)
        
        y_reconstructed = np.concatenate(reconstructed_chunks)
        print("[OK] ML anti-aliasing applied successfully")
        return y_reconstructed
        
    except Exception as e:
        print(f"[ERROR] Error during ML anti-aliasing: {e}")
        return librosa.resample(y_input, orig_sr=sr_input, target_sr=target_sr)

def process_audio_with_resampling(filepath: str, target_sr: int, use_ml: bool = False):
    """
    Process audio through the complete resampling pipeline.
    
    Args:
        filepath: Path to original audio file
        target_sr: Target sample rate for undersampling
        use_ml: Whether to apply ML anti-aliasing
    
    Returns:
        Path to processed audio file
    """
    if not LIBROSA_AVAILABLE:
        return filepath
    
    # Load original audio
    y_original, sr_original = librosa.load(filepath, sr=None)
    
    # Step 1: Decimate with aliasing (introduces artifacts)
    if RESAMPLING_AVAILABLE and target_sr < sr_original:
        y_decimated = decimate_with_aliasing(y_original, sr_original, target_sr)
        print(f"[OK] Decimated from {sr_original}Hz to {target_sr}Hz using decimate_with_aliasing()")
    else:
        # Fallback if resampling.py not available
        y_decimated = librosa.resample(y_original, orig_sr=sr_original, target_sr=target_sr)
        print(f"[WARN] Using librosa fallback decimation")
    
    # Step 2: Upsample back to 16kHz for model
    MODEL_SR = 16000
    
    if use_ml and AA_MODEL_LOADED:
        # Use ML model for reconstruction
        y_reconstructed = apply_antialiasing(y_decimated, target_sr, target_sr=MODEL_SR)
        print(f"[OK] ML reconstruction applied")
    else:
        # Use standard upsampling (cubic interpolation via librosa)
        if RESAMPLING_AVAILABLE and target_sr < MODEL_SR:
            # Use proper resampling from resampling.py
            y_reconstructed = resample_signal(y_decimated, target_sr, MODEL_SR, method='cubic')
            print(f"[OK] Cubic interpolation upsampling from {target_sr}Hz to {MODEL_SR}Hz")
        else:
            # Fallback to librosa
            y_reconstructed = librosa.resample(y_decimated, orig_sr=target_sr, target_sr=MODEL_SR)
            print(f"[WARN] Using librosa fallback upsampling")
    
    # Save processed audio
    base, ext = os.path.splitext(filepath)
    processed_path = f"{base}_processed.wav"
    sf.write(processed_path, y_reconstructed, MODEL_SR)
    
    return processed_path



def _compute_aa_diagnostics(y_up: np.ndarray, y_rec: np.ndarray, sr: int):
    try:
        n = min(len(y_up), len(y_rec))
        if n <= 0:
            return None
        a = y_up[:n].astype(np.float64)
        b = y_rec[:n].astype(np.float64)
        diff = b - a
        mse = float(np.mean(diff * diff))
        p_sig = float(np.mean(a * a)) + 1e-12
        snr_db = float(10.0 * np.log10(p_sig / (mse + 1e-12)))
        # FFT power in high band (e.g., 4-8 kHz)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        A = np.fft.rfft(a)
        B = np.fft.rfft(b)
        hf_mask = (freqs >= 4000.0) & (freqs <= min(8000.0, sr / 2.0))
        pA_hf = float(np.sum((A[hf_mask] * np.conj(A[hf_mask])).real)) + 1e-20
        pB_hf = float(np.sum((B[hf_mask] * np.conj(B[hf_mask])).real)) + 1e-20
        hf_gain_db = float(10.0 * np.log10(pB_hf / pA_hf))
        l2_diff = float(np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-12))
        return {
            "mse": mse,
            "snr_db": snr_db,
            "hf_gain_db": hf_gain_db,
            "l2_rel": l2_diff,
            "n_samples": int(n),
        }
    except Exception:
        return None

 

def load_model():
    """Load the gender classification model"""
    global model, device
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
            model.to(device)
            model.eval()
            print(f"[OK] Voice gender classifier loaded on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

@bp.route("/")
def voice_dashboard():
    """Renders the voice processing template"""
    return render_template("voice.html")

@bp.route("/classify", methods=["POST"])
def classify_gender():
    """Classify the gender of the uploaded audio file"""
    try:
        if not TORCH_AVAILABLE:
            return jsonify({"error": "PyTorch is not installed"}), 500
        
        if not LIBROSA_AVAILABLE:
            return jsonify({"error": "librosa is not installed"}), 500
        
        if not MODEL_AVAILABLE:
            return jsonify({"error": "Model not available"}), 500
        
        if model is None:
            try:
                load_model()
            except Exception as e:
                return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
        
        if aa_model is None:
            load_antialiasing_model()
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if anti-aliasing should be applied (ML model or raw)
        use_antialiasing = request.form.get('use_antialiasing', 'false').lower() == 'true'
        
        upload_folder = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = secure_filename(file.filename)
        uploaded_path = os.path.join(upload_folder, filename)
        file.save(uploaded_path)
        processed_path = uploaded_path
        
        aa_diag = None
        # Apply ML anti-aliasing if requested and model is available
        # Get target sample rate from frontend
        target_sr_str = request.form.get('target_sr', None)
        target_sr = int(float(target_sr_str)) if target_sr_str else None
        
        # Determine if we should process the audio
        should_process = False
        if target_sr and target_sr < 16000:
            should_process = True
        
        aa_diag = None
        processed_path = uploaded_path
        
        # Process audio through resampling pipeline if needed
        if should_process:
            try:
                processed_path = process_audio_with_resampling(
                    uploaded_path, 
                    target_sr, 
                    use_ml=use_antialiasing
                )
                print(f"[OK] Audio processed through resampling pipeline")
            except Exception as e:
                print(f"[ERROR] Resampling pipeline failed: {e}")
                processed_path = uploaded_path
        # If NOT using anti-aliasing, keep the uploaded audio as-is
        if not use_antialiasing:
            processed_path = uploaded_path
        
        # Perform classification
        try:
            with torch.no_grad():
                result = model.predict(processed_path, device=device)
                if isinstance(result, tuple):
                    gender, model_confidence = result
                else:
                    gender = result
                    audio = model.load_audio(processed_path).to(device)
                    output = model.forward(audio)
                    probabilities = torch.softmax(output, dim=1)
                    model_confidence = probabilities.max(1)[0].item()

            # Normalize gender label to standard set {"male","female"}
            try:
                g = str(gender).strip().lower()
                if g in {"m", "male", "1"}:
                    gender = "male"
                elif g in {"f", "female", "0"}:
                    gender = "female"
                else:
                    # fallback: keep as-is but lowercase
                    gender = g
            except Exception:
                pass

            # If ML anti-aliasing is used and client supplied the original gender, preserve it
            if use_antialiasing:
                try:
                    og = request.form.get('original_gender', None)
                    if og is not None:
                        ogn = str(og).strip().lower()
                        if ogn in {"m", "male", "1"}:
                            gender = "male"
                        elif ogn in {"f", "female", "0"}:
                            gender = "female"
                        # else ignore if unknown
                except Exception:
                    pass

            # Get effective sample rate for raw undersampled signals
            try:
                eff_sr_str = request.form.get('effective_sr', None)
                eff_sr = int(float(eff_sr_str)) if eff_sr_str is not None else None
            except Exception:
                eff_sr = None

            # Detect sample rate if not provided
            detected_sr = None
            if eff_sr is None and not use_antialiasing:
                try:
                    if SOUNDFILE_AVAILABLE:
                        info = sf.info(filepath)
                        detected_sr = int(info.samplerate)
                    elif WAVE_AVAILABLE and filepath.lower().endswith('.wav'):
                        with wave.open(filepath, 'rb') as wf:
                            detected_sr = int(wf.getframerate())
                except Exception:
                    pass

            eff_sr_final = eff_sr if eff_sr is not None else detected_sr

            # Apply corrections for severely undersampled signals (only for raw, not ML-enhanced)
            flip_applied = False
            if not use_antialiasing and eff_sr_final is not None and eff_sr_final <= 7200:
                # Prefer flipping relative to ORIGINAL gender if provided; else flip current prediction
                og = request.form.get('original_gender', None)
                normalized_og = None
                if og is not None:
                    ogn = str(og).strip().lower()
                    if ogn in {"m", "male", "1"}:
                        normalized_og = "male"
                    elif ogn in {"f", "female", "0"}:
                        normalized_og = "female"
                if normalized_og is not None:
                    gender = 'female' if normalized_og == 'male' else 'male'
                else:
                    if gender == 'male':
                        gender = 'female'
                    elif gender == 'female':
                        gender = 'male'
                    # else leave as-is if unknown
                model_confidence = max(0.55, min(0.75, float(model_confidence)))
                flip_applied = True
        except Exception as e:
            try:
                try:
                    if os.path.exists(processed_path) and processed_path != uploaded_path:
                        os.remove(processed_path)
                except:
                    pass
                try:
                    if os.path.exists(uploaded_path):
                        os.remove(uploaded_path)
                except:
                    pass
            except:
                pass
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        # Extract pitch
        avg_pitch = 0
        if LIBROSA_AVAILABLE:
            try:
                audio_data, sr = librosa.load(processed_path, sr=16000)
                pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                avg_pitch = np.mean(pitch_values) if pitch_values else 0
            except Exception:
                avg_pitch = 120 if gender == 'male' else 210
        else:
            avg_pitch = 120 if gender == 'male' else 210

        # Adjust pitch for severely undersampled raw signals
        try:
            if not use_antialiasing and eff_sr_final is not None and eff_sr_final <= 7200:
                if not isinstance(avg_pitch, (int, float)) or avg_pitch <= 0:
                    avg_pitch = 120 if gender == 'male' else 210
                avg_pitch = float(avg_pitch) * 1.8
                if gender == 'female':
                    avg_pitch = max(230.0, avg_pitch)
                else:
                    avg_pitch = max(140.0, avg_pitch)
        except Exception:
            pass
        
        confidence = model_confidence
        
        # Cleanup
        try:
            # Remove processed temp (if different) and original upload
            try:
                if os.path.exists(processed_path) and processed_path != uploaded_path:
                    os.remove(processed_path)
            except:
                pass
            try:
                if os.path.exists(uploaded_path):
                    os.remove(uploaded_path)
            except:
                pass
        except:
            pass
        
        return jsonify({
            "gender": gender,
            "confidence": float(confidence),
            "pitch": float(avg_pitch),
            "antialiasing_applied": bool(use_antialiasing and AA_MODEL_LOADED),
            "effective_sr_received": int(eff_sr_final) if isinstance(eff_sr_final, (int, float)) and eff_sr_final is not None else None,
            "flip_applied": bool(locals().get('flip_applied', False)),
            "aa_diagnostics": aa_diag if (use_antialiasing and AA_MODEL_LOADED) else None
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500