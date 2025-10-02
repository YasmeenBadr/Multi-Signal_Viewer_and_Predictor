from flask import Blueprint, render_template, request, jsonify
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

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
