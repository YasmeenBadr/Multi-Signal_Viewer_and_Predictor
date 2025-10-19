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
            print(f"✓ Voice gender classifier loaded on {device}")
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
                gender = model.predict(filepath, device=device)
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
        
        # Calculate confidence based on pitch ranges
        # Typical male: 85-180 Hz, Female: 165-255 Hz
        if gender == 'male':
            confidence = 0.9 if avg_pitch < 165 else 0.6
        else:
            confidence = 0.9 if avg_pitch > 165 else 0.6
        
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