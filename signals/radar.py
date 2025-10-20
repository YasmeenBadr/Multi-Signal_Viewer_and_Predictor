from flask import Blueprint, render_template, request, jsonify
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import os
import uuid
import time
from scipy.io import wavfile
import numpy as np

bp = Blueprint('radar', __name__, template_folder='templates')

# Load model and processor
MODEL_ID = "preszzz/drone-audio-detection-05-17-trial-0"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
labels = model.config.id2label

TEMP_DIR = os.path.join('static', 'temp')


def validate_file():
    if 'file' not in request.files:
        return None, (jsonify({'error': 'No file part'}), 400)
    file = request.files['file']
    if file.filename == '':
        return None, (jsonify({'error': 'No selected file'}), 400)
    return file, None


def run_inference(audio_data):
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    return labels[pred_idx], probs[0][pred_idx].item()


def save_audio(audio_data, sample_rate, prefix):
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
    file, error = validate_file()
    if error:
        return error
    
    try:
        audio_data, sr = librosa.load(file.stream, sr=None)
        audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        predicted_class, confidence = run_inference(audio_16k)
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/analyze_downsampled', methods=['POST'])
def analyze_downsampled():
    file, error = validate_file()
    if error:
        return error
    
    try:
        target_sr = int(request.form.get('target_sr', 8000))
        if not 1000 <= target_sr <= 16000:
            return jsonify({'error': 'Sample rate must be between 1000 and 16000 Hz'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid sample rate value'}), 400
    
    try:
        audio_data, original_sr = librosa.load(file.stream, sr=None)
        
        # Original analysis (16kHz)
        audio_16k = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)
        pred_orig, conf_orig = run_inference(audio_16k)
        
        # Downsampled analysis
        audio_down = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
        audio_up = librosa.resample(audio_down, orig_sr=target_sr, target_sr=16000)
        pred_down, conf_down = run_inference(audio_up)
        
        # Save files
        orig_file = save_audio(audio_16k, 16000, 'original')
        down_file = save_audio(audio_down, target_sr, 'downsampled')
        
        return jsonify({
            "original": {
                "predicted_class": pred_orig,
                "confidence": round(conf_orig, 4),
                "audio_url": f"/static/temp/{orig_file}"
            },
            "downsampled": {
                "predicted_class": pred_down,
                "confidence": round(conf_down, 4),
                "sample_rate": target_sr,
                "audio_url": f"/static/temp/{down_file}"
            },
            "comparison": {
                "confidence_drop": round(conf_orig - conf_down, 4),
                "classification_changed": pred_orig != pred_down
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/cleanup_temp', methods=['POST'])
def cleanup_temp():
    try:
        if os.path.exists(TEMP_DIR):
            current_time = time.time()
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 3600:
                    os.remove(filepath)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500