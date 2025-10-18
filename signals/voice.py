from flask import Blueprint, render_template, request, jsonify, send_file
import librosa
import numpy as np
import io
from scipy.io import wavfile
import torch
import os
import sys
import tempfile

# Add the voice-gender-classifier directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_PATH = os.path.join(project_root, 'voice-gender-classifier')

print(f"Looking for classifier at: {CLASSIFIER_PATH}")
print(f"Exists? {os.path.exists(CLASSIFIER_PATH)}")

if os.path.exists(CLASSIFIER_PATH):
    sys.path.insert(0, CLASSIFIER_PATH)
    print(f"✓ Added to path")
else:
    print(f"✗ Path not found!")

# Define the blueprint for the Voice Processing Suite
bp = Blueprint('voice', __name__, template_folder='templates')

# Global model variable
model = None
device = None

def load_model():
    """Load the ECAPA gender classification model from HuggingFace"""
    global model, device
    try:
        # Import the model class
        from model import ECAPA_gender
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model from HuggingFace hub
        print("Loading gender classification model from HuggingFace...")
        model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
        model.eval()
        model.to(device)
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"Warning: Could not load model - {e}")
        print("Will use fallback pitch-based classification")
        model = None

# Try to load model on blueprint initialization
try:
    load_model()
except Exception as e:
    print(f"Error during model initialization: {e}")

def classify_with_model(audio_path):
    """
    Classify gender using the ECAPA model
    Returns: ('male' or 'female', confidence)
    """
    if model is None:
        return None, None
    
    try:
        with torch.no_grad():
            output = model.predict(audio_path, device=device)
        
        # The output is a string: 'male' or 'female'
        gender = output.lower()
        # Return None for confidence to trigger pitch-based confidence calculation
        return gender, None
    
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None, None

def classify_with_pitch(audio_data, sr):
    """
    IMPROVED: Pitch-based classification with realistic confidence
    Male voices typically < 165 Hz, female > 165 Hz
    """
    try:
        # Extract pitch with better parameters
        pitches, magnitudes = librosa.piptrack(
            y=audio_data, 
            sr=sr, 
            fmin=50,  # Minimum frequency for human voice
            fmax=400,  # Maximum frequency for human voice
            threshold=0.1  # Reduced threshold for better detection
        )
        
        # Get pitch values with significant magnitude
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            mag = magnitudes[index, t]
            # Only include pitches with significant magnitude
            if pitch > 0 and mag > np.mean(magnitudes) * 0.5:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 10:  # Need enough samples for reliable estimate
            mean_pitch = np.mean(pitch_values)
            std_pitch = np.std(pitch_values)
            
            # Calculate confidence based on distance from threshold (165 Hz)
            threshold = 165
            distance = abs(mean_pitch - threshold)
            
            # Confidence increases with distance from threshold
            # Max confidence at ±50 Hz or more from threshold
            confidence = min(0.95, 0.55 + (distance / 100.0))
            
            # Reduce confidence if pitch varies a lot (indicates uncertainty)
            if std_pitch > 30:
                confidence *= 0.85
            
            # Determine gender
            if mean_pitch < threshold:
                return 'male', confidence, mean_pitch
            else:
                return 'female', confidence, mean_pitch
        else:
            # Not enough pitch data
            return 'unknown', 0.5, 0
    
    except Exception as e:
        print(f"Error in pitch classification: {e}")
        return 'unknown', 0.5, 0

def apply_aliasing(audio_data, original_sr, target_sr):
    """
    Apply SEVERE aliasing effect by downsampling WITHOUT any filter
    This intentionally creates maximum distortion
    """
    if target_sr >= original_sr:
        return audio_data
    
    # Calculate downsample factor
    downsample_factor = int(original_sr / target_sr)
    
    # Simple decimation (just skip samples - causes aliasing)
    aliased_audio = audio_data[::downsample_factor]
    
    # For even MORE dramatic effect, add spectral inversion
    if target_sr < 8000:
        # Invert every other sample to simulate frequency folding
        aliased_audio = aliased_audio.copy()  # Make a copy to avoid modifying original
        aliased_audio[1::2] = -aliased_audio[1::2]
    
    return aliased_audio

def apply_anti_aliasing(audio_data, original_sr, target_sr):
    """
    Apply proper anti-aliasing using librosa's high-quality resampling
    MUCH FASTER than scipy filtfilt
    """
    if target_sr >= original_sr:
        return audio_data
    
    # Use librosa's resample which has built-in anti-aliasing
    resampled = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    
    return resampled

def save_temp_audio(audio_data, sr, prefix='temp'):
    """Save audio to temporary WAV file for model prediction"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix=prefix)
    temp_path = temp_file.name
    temp_file.close()
    
    # Normalize and convert to int16
    if np.max(np.abs(audio_data)) > 0:
        audio_normalized = audio_data / np.max(np.abs(audio_data))
    else:
        audio_normalized = audio_data
    audio_int16 = np.int16(audio_normalized * 32767)
    
    wavfile.write(temp_path, sr, audio_int16)
    return temp_path

@bp.route("/")
def voice_dashboard():
    """Renders the voice processing template"""
    return render_template("voice.html")

@bp.route("/process_audio", methods=['POST'])
def process_audio():
    """
    Process uploaded audio file and classify gender with different sampling methods
    OPTIMIZED for speed
    """
    temp_original = None
    temp_aliased = None
    temp_anti = None
    temp_input = None
    
    try:
        # Get the uploaded file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        target_sr = int(request.form.get('target_sr', 8000))
        
        print(f"\n{'='*50}")
        print(f"Processing audio: {audio_file.filename}")
        print(f"Target sample rate: {target_sr} Hz")
        
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_input.close()
        audio_file.save(temp_input.name)
        
        # FIX 1: Load at 16kHz max for MUCH faster processing
        print("Loading audio at 16kHz...")
        audio_data, original_sr = librosa.load(temp_input.name, sr=16000)
        
        print(f"Loaded: {len(audio_data)} samples at {original_sr} Hz")
        
        # === 1. CLASSIFY ORIGINAL AUDIO ===
        if model is not None:
            # Use the ECAPA model
            temp_original = save_temp_audio(audio_data, original_sr, 'original')
            original_gender, original_confidence = classify_with_model(temp_original)
            
            # If model didn't return confidence, use pitch-based
            if original_confidence is None:
                _, original_confidence, original_pitch = classify_with_pitch(audio_data, original_sr)
            else:
                _, _, original_pitch = classify_with_pitch(audio_data, original_sr)
        else:
            # Use pitch fallback
            original_gender, original_confidence, original_pitch = classify_with_pitch(audio_data, original_sr)
        
        print(f"Original: {original_gender} (conf: {original_confidence:.2f}, pitch: {original_pitch:.1f} Hz)")
        
        # === 2. APPLY ALIASING (NO FILTER) ===
        aliased_audio = apply_aliasing(audio_data, original_sr, target_sr)
        print(f"Aliased: {len(aliased_audio)} samples at {target_sr} Hz")
        
        if model is not None:
            temp_aliased = save_temp_audio(aliased_audio, target_sr, 'aliased')
            aliased_gender, aliased_confidence = classify_with_model(temp_aliased)
            
            if aliased_confidence is None:
                _, aliased_confidence, aliased_pitch = classify_with_pitch(aliased_audio, target_sr)
            else:
                _, _, aliased_pitch = classify_with_pitch(aliased_audio, target_sr)
        else:
            aliased_gender, aliased_confidence, aliased_pitch = classify_with_pitch(aliased_audio, target_sr)
        
        print(f"Aliased: {aliased_gender} (conf: {aliased_confidence:.2f}, pitch: {aliased_pitch:.1f} Hz)")
        
        # === 3. APPLY ANTI-ALIASING (WITH FILTER) ===
        anti_aliased_audio = apply_anti_aliasing(audio_data, original_sr, target_sr)
        print(f"Anti-aliased: {len(anti_aliased_audio)} samples at {target_sr} Hz")
        
        if model is not None:
            temp_anti = save_temp_audio(anti_aliased_audio, target_sr, 'anti_aliased')
            anti_aliased_gender, anti_aliased_confidence = classify_with_model(temp_anti)
            
            if anti_aliased_confidence is None:
                _, anti_aliased_confidence, anti_aliased_pitch = classify_with_pitch(anti_aliased_audio, target_sr)
            else:
                _, _, anti_aliased_pitch = classify_with_pitch(anti_aliased_audio, target_sr)
        else:
            anti_aliased_gender, anti_aliased_confidence, anti_aliased_pitch = classify_with_pitch(anti_aliased_audio, target_sr)
        
        print(f"Anti-aliased: {anti_aliased_gender} (conf: {anti_aliased_confidence:.2f}, pitch: {anti_aliased_pitch:.1f} Hz)")
        print(f"{'='*50}\n")
        
        # Prepare response
        response = {
            'original': {
                'gender': original_gender,
                'confidence': float(original_confidence),
                'sample_rate': int(original_sr),
                'pitch': float(original_pitch)
            },
            'aliased': {
                'gender': aliased_gender,
                'confidence': float(aliased_confidence),
                'sample_rate': int(target_sr),
                'pitch': float(aliased_pitch)
            },
            'anti_aliased': {
                'gender': anti_aliased_gender,
                'confidence': float(anti_aliased_confidence),
                'sample_rate': int(target_sr),
                'pitch': float(anti_aliased_pitch)
            },
            'model_used': 'ECAPA-TDNN' if model is not None else 'Pitch-based (fallback)'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR processing audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up all temporary files
        for temp_file in [temp_original, temp_aliased, temp_anti, temp_input]:
            if temp_file:
                try:
                    if isinstance(temp_file, str):
                        os.remove(temp_file)
                    else:
                        os.remove(temp_file.name)
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

@bp.route("/resample_audio", methods=['POST'])
def resample_audio():
    """
    Resample audio and return the processed audio file for playback
    FIXED: Proper normalization and faster processing
    """
    temp_input = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        target_sr = int(request.form.get('target_sr', 8000))
        mode = request.form.get('mode', 'aliased')
        
        print(f"Resampling for playback: mode={mode}, target_sr={target_sr}")
        
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_input.close()
        audio_file.save(temp_input.name)
        
        # Load at 16kHz for faster processing
        audio_data, sr = librosa.load(temp_input.name, sr=16000)
        
        # Process based on mode
        if mode == 'anti_aliased':
            processed_audio = apply_anti_aliasing(audio_data, sr, target_sr)
        else:  # aliased
            processed_audio = apply_aliasing(audio_data, sr, target_sr)
        
        # FIX 2: Better normalization to prevent clipping/silence
        if np.max(np.abs(processed_audio)) > 0:
            processed_audio = processed_audio / np.max(np.abs(processed_audio)) * 0.95
        else:
            processed_audio = np.zeros_like(processed_audio)
        
        # Convert to int16
        processed_audio_int = np.int16(processed_audio * 32767)
        
        # Create WAV file in memory
        wav_io = io.BytesIO()
        wavfile.write(wav_io, target_sr, processed_audio_int)
        wav_io.seek(0)
        
        print(f"✓ Resampled successfully: {len(processed_audio_int)} samples at {target_sr} Hz")
        
        return send_file(
            wav_io,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'{mode}_audio.wav'
        )
    
    except Exception as e:
        print(f"Error resampling audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary file
        if temp_input:
            try:
                os.remove(temp_input.name)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")