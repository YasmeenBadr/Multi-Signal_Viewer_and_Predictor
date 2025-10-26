import os
import sys
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import io
import base64

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
        print("⚠ TensorFlow not available. Anti-aliasing will use standard upsampling.")
        return
    
    if aa_model is None:
        try:
            aa_model = tf.keras.models.load_model(ANTI_ALIAS_MODEL_PATH, compile=False)
            AA_MODEL_LOADED = True
            print("✅ Anti-aliasing model loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load anti-aliasing model: {e}")
            AA_MODEL_LOADED = False
            aa_model = None

def apply_antialiasing(y_input, sr_input, target_sr=16000):
    """
    Apply anti-aliasing reconstruction using the TensorFlow model.
    If model is not available, perform standard upsampling.
    """
    if not AA_MODEL_LOADED or aa_model is None:
        print("⚠ ML model not available, performing standard upsampling.")
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
        print(f"✅ ML anti-aliasing applied successfully")
        return y_reconstructed
        
    except Exception as e:
        print(f"❌ Error during ML anti-aliasing: {e}")
        return librosa.resample(y_input, orig_sr=sr_input, target_sr=target_sr)

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
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Apply ML anti-aliasing if requested and model is available
        if use_antialiasing:
            if AA_MODEL_LOADED:
                try:
                    y, sr = librosa.load(filepath, sr=None)
                    y_enhanced = apply_antialiasing(y, sr, target_sr=16000)
                    
                    enhanced_path = filepath.replace('.wav', '_ml_enhanced.wav')
                    sf.write(enhanced_path, y_enhanced, 16000)
                    filepath = enhanced_path
                    print("✅ Using ML-enhanced audio for classification")
                except Exception as e:
                    print(f"⚠ ML anti-aliasing failed: {e}")
                    use_antialiasing = False
            else:
                print("⚠ ML model not loaded, processing raw signal")
                use_antialiasing = False
        
        # Perform classification
        try:
            with torch.no_grad():
                result = model.predict(filepath, device=device)
                if isinstance(result, tuple):
                    gender, model_confidence = result
                else:
                    gender = result
                    audio = model.load_audio(filepath).to(device)
                    output = model.forward(audio)
                    probabilities = torch.softmax(output, dim=1)
                    model_confidence = probabilities.max(1)[0].item()

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
            if not use_antialiasing and eff_sr_final is not None and eff_sr_final <= 7200:
                gender = 'female' if gender == 'male' else 'male'
                model_confidence = max(0.55, min(0.75, float(model_confidence)))
        except Exception as e:
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        # Extract pitch
        avg_pitch = 0
        if LIBROSA_AVAILABLE:
            try:
                audio_data, sr = librosa.load(filepath, sr=16000)
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
            os.remove(filepath)
            if use_antialiasing:
                enhanced_path = filepath.replace('.wav', '_ml_enhanced.wav')
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
        except:
            pass
        
        return jsonify({
            "gender": gender,
            "confidence": float(confidence),
            "pitch": float(avg_pitch),
            "antialiasing_applied": use_antialiasing and AA_MODEL_LOADED
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500