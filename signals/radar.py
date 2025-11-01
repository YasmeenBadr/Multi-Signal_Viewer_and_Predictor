from flask import Blueprint, render_template, request, jsonify
import librosa  # Audio loading/processing library
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification # HuggingFace transformers
import os
import uuid  # For generating unique filenames
import time  # For file cleanup based on age
from scipy.io import wavfile  # For writing WAV files
import numpy as np
from .resampling import resample_signal, decimate_with_aliasing # Custom resampling functions

bp = Blueprint('radar', __name__, template_folder='templates')

# Load model and processor (happens once when server starts)
MODEL_ID = "preszzz/drone-audio-detection-05-17-trial-0"
processor = AutoProcessor.from_pretrained(MODEL_ID)  # Prepares audio for model input
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID) # The actual AI model
labels = model.config.id2label


# Directory where temporary audio files will be saved
TEMP_DIR = os.path.join('static', 'temp')

def validate_file():
   
    # Check if 'file' field exists in the uploaded form data
    if 'file' not in request.files:
        return None, (jsonify({'error': 'No file part'}), 400)
    file = request.files['file']

    # Check if user actually selected a file (empty filename means no file)
    if file.filename == '':
        return None, (jsonify({'error': 'No selected file'}), 400)
    return file, None


def run_inference(audio_data):
    """  
    Run the AI model on audio data to predict if it contains a drone.
    
    Args:
        audio_data: numpy array of audio samples at 16kHz
    
    Returns:
        (predicted_class, confidence) tuple
     """
    
     # Preprocess audio: convert to format model expects (PyTorch tensors)
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
     # Run model WITHOUT calculating gradients (faster, uses less memory)
    with torch.no_grad():
        outputs = model(**inputs)  # Get raw model predictions (logits)

     # Convert logits to probabilities (0 to 1) using softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
     # Find which class has highest probability
    pred_idx = torch.argmax(probs, dim=-1).item()
    # Return class name and its confidence score
    return labels[pred_idx], probs[0][pred_idx].item()


def save_audio(audio_data, sample_rate, prefix):
    """
    Save audio data to temporary WAV file
     Args:
        audio_data: numpy array of audio samples (normalized -1 to 1)
        sample_rate: sampling rate in Hz
        prefix: filename prefix like "original" or "downsampled"
    
    Returns:
        filename (without path) for URL construction
    """

    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    # Generate unique filename: "original_a3f9b2c1.wav"
    filename = f'{prefix}_{uuid.uuid4().hex[:8]}.wav'

     # Full path: "static/temp/original_a3f9b2c1.wav"
    filepath = os.path.join(TEMP_DIR, filename)

     # Convert normalized audio (-1 to 1) to 16-bit integers for audio quality standards
    # WAV format requires integer samples
    wavfile.write(filepath, sample_rate, (audio_data * 32767).astype(np.int16))
    return filename


@bp.route('/')
def index():
    """
    Home page route: serves the HTML interface.
    URL: http://localhost:5000/radar/
    """
    return render_template('radar.html')
   


@bp.route('/analyze', methods=['POST'])
def analyze():

    # Check if file was uploaded correctly
    file, error = validate_file()
    if error:
        return error
    
    try:
        # Load audio with librosa
        # sr=None means "keep original sample rate"
        audio_data, sr = librosa.load(file.stream, sr=None)
        
        # Resample to 16kHz using resampling.py utility (with anti-aliasing filter)
        # Model was trained on 16kHz audio, so we must match that
        audio_16k = resample_signal(audio_data, sr, 16000, method='linear')
        
        # Run inference to get prediction
        predicted_class, confidence = run_inference(audio_16k)
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 #500 for internal server error


@bp.route('/analyze_downsampled', methods=['POST'])
def analyze_downsampled():
  
    file, error = validate_file()
    if error:
        return error
    
    # Validate target sample rate
    try:

    # Get target sample rate from form (default: 9000 Hz)
    # This is what user selected with the slider
        target_sr = int(request.form.get('target_sr', 9000))

    # Sanity check: must be between 1-16 kHz
        if not 1000 <= target_sr <= 16000:
            return jsonify({'error': 'Sample rate must be between 1000 and 16000 Hz'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid sample rate value'}), 400
    
    try:
        # Load original audio  at its native sample rate
        audio_data, original_sr = librosa.load(file.stream, sr=None)
        
        # === Original Analysis (proper 16kHz resampling) ===
        audio_16k = resample_signal(audio_data, original_sr, 16000, method='linear')
        pred_orig, conf_orig = run_inference(audio_16k)
        
        # === Downsampled Analysis (with aliasing for educational demonstration) ===
        # Step 1: Decimate with aliasing (simulates bad hardware/undersampling)
        # HIGH FREQUENCIES FOLD BACK (aliasing) - this is intentional for demo!
        audio_down = decimate_with_aliasing(audio_data, original_sr, target_sr)
        
        # Step 2: Upsample back to 16kHz for model inference
        # Note: Aliasing artifacts from step 1 are preserved
        audio_up = resample_signal(audio_down, target_sr, 16000, method='linear')
        pred_down, conf_down = run_inference(audio_up)
        
        # === Save audio files for comparison ===
        orig_file = save_audio(audio_16k, 16000, 'original')
        down_file = save_audio(audio_down, target_sr, 'downsampled')
        
        # === Calculate metrics ===
        nyquist_freq = target_sr / 2
        confidence_drop = conf_orig - conf_down
        classification_changed = pred_orig != pred_down
        
        # Determine sampling quality
        if target_sr >= 16000:
            sampling_status = "✓ Properly Sampled"
        elif target_sr >= 8000:
             # Marginal: might work but close to Nyquist limit
            sampling_status = "⚠️ Marginal (Nyquist: {}Hz)".format(int(nyquist_freq))
        else:
            sampling_status = "❌ Severely Undersampled (Nyquist: {}Hz)".format(int(nyquist_freq))
        

        # ========================================
        # RETURN COMPREHENSIVE COMPARISON
        # ========================================

        return jsonify({
            "original": {
                "predicted_class": pred_orig,
                "confidence": round(conf_orig, 4),
                "audio_url": f"/static/temp/{orig_file}",
                "sample_rate": 16000
            },
            "downsampled": {
                "predicted_class": pred_down,
                "confidence": round(conf_down, 4),
                "sample_rate": target_sr,
                "audio_url": f"/static/temp/{down_file}",
                "nyquist_frequency": int(nyquist_freq),
                "sampling_status": sampling_status
            },
            "comparison": {
                "confidence_drop": round(confidence_drop, 4),
                "confidence_drop_percent": round(confidence_drop * 100, 2),
                "classification_changed": classification_changed,
                "aliasing_present": target_sr < original_sr
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/cleanup_temp', methods=['POST'])
def cleanup_temp():
    """Clean up temporary audio files older than 1 hour"""
    try:
        # Check if temp directory exists
        if os.path.exists(TEMP_DIR):
            current_time = time.time() # Current timestamp in seconds
            cleaned_count = 0

            # Loop through all files in temp directory
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                 # Only process actual files (not directories)
                 # Get file modification time
                 # and If older than 1 hour (3600 seconds), delete it
                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 3600:
                    os.remove(filepath)
                    cleaned_count += 1
            return jsonify({'status': 'success', 'files_cleaned': cleaned_count})
        # Directory doesn't exist, nothing to clean
        return jsonify({'status': 'success', 'files_cleaned': 0})
    except Exception as e:
        return jsonify({'error': str(e)}), 500