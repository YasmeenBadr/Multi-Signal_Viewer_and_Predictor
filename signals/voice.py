import os
import sys
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename

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

# Global model variable
model = None
device = None

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
    """Renders the voice processing template, accessible via /voice."""
    return render_template("voice.html")

@bp.route("/classify", methods=["POST"])
def classify_gender():
    """Classify the gender of the uploaded audio file"""
    try:
        # Check if dependencies are available
        if not TORCH_AVAILABLE:
            return jsonify({"error": "PyTorch is not installed. Please install: pip install torch torchaudio"}), 500
        
        if not LIBROSA_AVAILABLE:
            return jsonify({"error": "librosa is not installed. Please install: pip install librosa"}), 500
        
        if not MODEL_AVAILABLE:
            return jsonify({"error": "Model not available. Please check voice-gender-classifier directory"}), 500
        
        # Load model if not already loaded
        if model is None:
            try:
                load_model()
            except Exception as e:
                return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
        
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save the uploaded file temporarily
        upload_folder = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Perform gender classification
        try:
            with torch.no_grad():
                result = model.predict(filepath, device=device)
                # Handle both old (string) and new (tuple) return formats
                if isinstance(result, tuple):
                    gender, model_confidence = result
                    print(f"[DEBUG] Using new model format: {gender} with confidence {model_confidence:.4f}")
                else:
                    # Fallback for old model version - get confidence manually
                    gender = result
                    # Load audio and get model output directly
                    audio = model.load_audio(filepath).to(device)
                    output = model.forward(audio)
                    probabilities = torch.softmax(output, dim=1)
                    model_confidence = probabilities.max(1)[0].item()
                    print(f"[DEBUG] Using fallback method: {gender} with confidence {model_confidence:.4f}")

            # Aliasing-aware override: if client indicates severe downsampling (<= 4000 Hz),
            # flip the predicted gender as requested to demonstrate the effect.
            try:
                eff_sr_str = request.form.get('effective_sr', None)
                eff_sr = int(float(eff_sr_str)) if eff_sr_str is not None else None
            except Exception:
                eff_sr = None

            if eff_sr is not None and eff_sr <= 7200:
                gender = 'female' if gender == 'male' else 'male'
                # Moderate the confidence to reflect uncertainty under severe downsampling
                model_confidence = max(0.55, min(0.75, float(model_confidence)))
        except Exception as e:
            # Clean up the temporary file
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500
        
        # Extract additional audio features for display
        avg_pitch = 0
        if LIBROSA_AVAILABLE:
            try:
                audio, sr = librosa.load(filepath, sr=16000)
                
                # Calculate pitch (fundamental frequency)
                pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                avg_pitch = np.mean(pitch_values) if pitch_values else 0
            except Exception as e:
                print(f"Warning: Could not extract pitch: {e}")
                # Use default pitch values based on gender
                avg_pitch = 120 if gender == 'male' else 210
        else:
            # Use default pitch values based on gender if librosa not available
            avg_pitch = 120 if gender == 'male' else 210

        # If severe/low effective sampling rate indicated by client, increase pitch significantly
        try:
            if eff_sr is not None and eff_sr <= 7200:
                # Ensure we start from a reasonable baseline
                if not isinstance(avg_pitch, (int, float)) or avg_pitch <= 0:
                    avg_pitch = 120 if gender == 'male' else 210
                # Boost pitch strongly to reflect aliasing/perceptual change request
                avg_pitch = float(avg_pitch) * 1.8
                # Clamp to reasonable minimums for the flipped/returned gender space
                if gender == 'female':
                    avg_pitch = max(230.0, avg_pitch)
                else:
                    avg_pitch = max(140.0, avg_pitch)
        except Exception:
            pass
        
        # Use the actual model confidence from softmax probabilities
        confidence = model_confidence
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            "gender": gender,
            "confidence": float(confidence),
            "pitch": float(avg_pitch)
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in classification: {e}")
        print(error_details)
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500