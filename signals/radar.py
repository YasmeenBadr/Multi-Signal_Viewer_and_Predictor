from flask import Blueprint, render_template, request, jsonify
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import os
import uuid
import time
from scipy.io import wavfile
import numpy as np
from .resampling import resample_signal, decimate_with_aliasing

bp = Blueprint('radar', __name__, template_folder='templates')

# Load model and processor
MODEL_ID = "preszzz/drone-audio-detection-05-17-trial-0"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
labels = model.config.id2label

TEMP_DIR = os.path.join('static', 'temp')


def validate_file():
    """Validate uploaded file from request"""
    if 'file' not in request.files:
        return None, (jsonify({'error': 'No file part'}), 400)
    file = request.files['file']
    if file.filename == '':
        return None, (jsonify({'error': 'No selected file'}), 400)
    return file, None


def run_inference(audio_data):
    """Run model inference on audio data at 16kHz"""
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    return labels[pred_idx], probs[0][pred_idx].item()


def save_audio(audio_data, sample_rate, prefix):
    """Save audio data to temporary WAV file"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    filename = f'{prefix}_{uuid.uuid4().hex[:8]}.wav'
    filepath = os.path.join(TEMP_DIR, filename)
    wavfile.write(filepath, sample_rate, (audio_data * 32767).astype(np.int16))
    return filename


@bp.route('/')
def index():
    return render_template('radar.html')


@bp.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded audio file for drone detection"""
    file, error = validate_file()
    if error:
        return error
    
    try:
        # Load audio with librosa
        audio_data, sr = librosa.load(file.stream, sr=None)
        
        # Resample to 16kHz using resampling.py utility
        audio_16k = resample_signal(audio_data, sr, 16000, method='linear')
        
        # Run inference
        predicted_class, confidence = run_inference(audio_16k)
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/analyze_downsampled', methods=['POST'])
def analyze_downsampled():
    """
    Educational endpoint: Compare drone detection at original vs downsampled rates.
    Uses aliasing decimation to demonstrate sampling theorem effects.
    """
    file, error = validate_file()
    if error:
        return error
    
    # Validate target sample rate
    try:
        target_sr = int(request.form.get('target_sr', 8000))
        if not 1000 <= target_sr <= 16000:
            return jsonify({'error': 'Sample rate must be between 1000 and 16000 Hz'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid sample rate value'}), 400
    
    try:
        # Load original audio
        audio_data, original_sr = librosa.load(file.stream, sr=None)
        
        # === Original Analysis (proper 16kHz resampling) ===
        audio_16k = resample_signal(audio_data, original_sr, 16000, method='linear')
        pred_orig, conf_orig = run_inference(audio_16k)
        
        # === Downsampled Analysis (with aliasing for educational demonstration) ===
        # Step 1: Decimate with aliasing (simulates bad hardware/undersampling)
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
            sampling_status = "⚠️ Marginal (Nyquist: {}Hz)".format(int(nyquist_freq))
        else:
            sampling_status = "❌ Severely Undersampled (Nyquist: {}Hz)".format(int(nyquist_freq))
        
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
        if os.path.exists(TEMP_DIR):
            current_time = time.time()
            cleaned_count = 0
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 3600:
                    os.remove(filepath)
                    cleaned_count += 1
            return jsonify({'status': 'success', 'files_cleaned': cleaned_count})
        return jsonify({'status': 'success', 'files_cleaned': 0})
    except Exception as e:
        return jsonify({'error': str(e)}), 500