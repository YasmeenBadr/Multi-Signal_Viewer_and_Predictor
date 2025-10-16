from flask import Blueprint, render_template, request, jsonify
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import os
import uuid
from scipy.io import wavfile
import numpy as np

bp = Blueprint('radar', __name__, template_folder='templates')

# Load the model and processor globally
model_id = "preszzz/drone-audio-detection-05-17-trial-0"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)
labels = model.config.id2label

@bp.route('/')
def index():
    return render_template('radar.html')

@bp.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load audio from stream
        audio_data, sr = librosa.load(file.stream, sr=None)
        
        # Resample to 16kHz
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        
        # Preprocess
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted index and class
        pred_idx = torch.argmax(probs, dim=-1).item()
        predicted_class = labels[pred_idx]
        
        # Get confidence
        confidence = probs[0][pred_idx].item()
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# ADDING ROUTE FOR DOWNSAMPLING ANALYSIS
@bp.route('/analyze_downsampled', methods=['POST'])
def analyze_downsampled():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get target sample rate from form
    try:
        target_sr = int(request.form.get('target_sr', 8000))
        
        # Validate sample rate (must be between 1000 and 16000 Hz)
        if target_sr < 1000 or target_sr > 16000:
            return jsonify({'error': 'Sample rate must be between 1000 and 16000 Hz'}), 400
            
    except ValueError:
        return jsonify({'error': 'Invalid sample rate value'}), 400
    
    try:
        # Load audio from stream at original sample rate
        audio_data, original_sr = librosa.load(file.stream, sr=None)
        
        # === ORIGINAL AUDIO ANALYSIS (16kHz) ===
        audio_16k = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)
        inputs_original = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            outputs_original = model(**inputs_original)
        
        probs_original = torch.nn.functional.softmax(outputs_original.logits, dim=-1)
        pred_idx_original = torch.argmax(probs_original, dim=-1).item()
        predicted_class_original = labels[pred_idx_original]
        confidence_original = probs_original[0][pred_idx_original].item()
        
        # === DOWNSAMPLED AUDIO ANALYSIS ===
        # First downsample to target rate, then upsample back to 16kHz for model
        audio_downsampled = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
        audio_upsampled = librosa.resample(audio_downsampled, orig_sr=target_sr, target_sr=16000)
        
        inputs_downsampled = processor(audio_upsampled, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            outputs_downsampled = model(**inputs_downsampled)
        
        probs_downsampled = torch.nn.functional.softmax(outputs_downsampled.logits, dim=-1)
        pred_idx_downsampled = torch.argmax(probs_downsampled, dim=-1).item()
        predicted_class_downsampled = labels[pred_idx_downsampled]
        confidence_downsampled = probs_downsampled[0][pred_idx_downsampled].item()
        
        # Calculate changes
        confidence_drop = confidence_original - confidence_downsampled
        classification_changed = predicted_class_original != predicted_class_downsampled
        
        # === SAVE AUDIO FILES FOR PLAYBACK ===
        # Create a static/temp directory if it doesn't exist
        temp_dir = os.path.join('static', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())[:8]
        original_filename = f'original_{unique_id}.wav'
        downsampled_filename = f'downsampled_{unique_id}.wav'
        
        original_path = os.path.join(temp_dir, original_filename)
        downsampled_path = os.path.join(temp_dir, downsampled_filename)
        
        # Save original audio (16kHz)
        audio_16k_int = (audio_16k * 32767).astype(np.int16)
        wavfile.write(original_path, 16000, audio_16k_int)
        
        # Save downsampled audio (at target rate for authentic playback)
        audio_downsampled_int = (audio_downsampled * 32767).astype(np.int16)
        wavfile.write(downsampled_path, target_sr, audio_downsampled_int)
        
        return jsonify({
            "original": {
                "predicted_class": predicted_class_original,
                "confidence": round(confidence_original, 4),
                "audio_url": f"/static/temp/{original_filename}"
            },
            "downsampled": {
                "predicted_class": predicted_class_downsampled,
                "confidence": round(confidence_downsampled, 4),
                "sample_rate": target_sr,
                "audio_url": f"/static/temp/{downsampled_filename}"
            },
            "comparison": {
                "confidence_drop": round(confidence_drop, 4),
                "classification_changed": classification_changed
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


#Add cleanup route to delete old temp files

@bp.route('/cleanup_temp', methods=['POST'])
def cleanup_temp():
    """Delete temporary audio files older than 1 hour"""
    try:
        temp_dir = os.path.join('static', 'temp')
        if os.path.exists(temp_dir):
            import time
            current_time = time.time()
            for filename in os.listdir(temp_dir):
                filepath = os.path.join(temp_dir, filename)
                # Delete files older than 1 hour (3600 seconds)
                if os.path.isfile(filepath):
                    if current_time - os.path.getmtime(filepath) > 3600:
                        os.remove(filepath)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500