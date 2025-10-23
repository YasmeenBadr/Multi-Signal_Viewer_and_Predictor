# Voice Processing Suite - Quick Start Guide

## Overview
Complete voice processing and gender classification system with anti-aliasing filters, signal reconstruction, and ML-based analysis.

📚 **For detailed documentation, see [VOICE_PROCESSING_DOCUMENTATION.md](VOICE_PROCESSING_DOCUMENTATION.md)**

## Quick Start

### Install Dependencies
Run this command to install all required packages:
```bash
pip install -r voice_requirements.txt
```

Or install individually:
```bash
pip install torch torchaudio librosa numpy huggingface_hub safetensors soundfile
```

### Run the Application
```bash
python app.py
```

Then navigate to: `http://localhost:5000/voice`

## Implementation Details

### Backend (`signals/voice.py`)
- **Model**: ECAPA-TDNN pre-trained model from HuggingFace (`JaesungHuh/voice-gender-classifier`)
- **Endpoint**: `/voice/classify` (POST)
- **Features Extracted**:
  - Gender classification (male/female)
  - Confidence score
  - Average pitch (fundamental frequency)

### Frontend (`templates/voice.html`)
- Added "Classify Gender" button
- Classification results display showing:
  - Gender (color-coded: blue for male, pink for female)
  - Confidence percentage
  - Average pitch in Hz

## Dependencies Required
The following packages are needed (from `voice-gender-classifier/requirements.txt`):
- torch
- torchaudio
- librosa
- numpy
- huggingface_hub
- safetensors

## Features

### 🎤 Voice Input
- **Microphone Recording**: Record directly from browser
- **File Upload**: Drag & drop MP3/WAV files

### 🔊 Audio Processing
- **Sample Rate Adjustment**: 4kHz - 44.1kHz
- **Anti-Aliasing Filter**: Toggle ON/OFF
- **Signal Reconstruction**: Upsample filtered signals
- **Three Playback Modes**: Original, Resampled, Reconstructed

### 🤖 Gender Classification
- **Original Audio**: Classify at full quality
- **Resampled Audio**: Test with aliasing effects
- **Reconstructed Audio**: Classify recovered signal
- **Comparative Analysis**: Side-by-side results

### 📊 Analysis Features
- **Classification Match**: Compare results across versions
- **Confidence Tracking**: Monitor accuracy changes
- **Pitch Analysis**: Fundamental frequency detection
- **Nyquist Frequency**: Display theoretical limits
- **Aliasing Effects**: Detailed impact assessment

## How to Use

### Basic Workflow
1. **Input**: Record voice or upload audio file
2. **Configure**: Adjust sample rate slider (try 6000 Hz for dramatic effects)
3. **Toggle**: Enable/disable anti-aliasing filter
4. **Playback**: Listen to all three versions
5. **Classify**: Click all three classification buttons
6. **Analyze**: Compare results and observe aliasing effects

### Demonstrating Aliasing
1. Upload clear voice sample
2. Set sample rate to 6000 Hz
3. **With filter OFF**: Classify resampled → observe poor results
4. **With filter ON**: Classify reconstructed → observe improvement
5. Compare confidence scores and pitch accuracy

## Model Details
- **Architecture**: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
- **Input**: 16kHz mono audio
- **Output**: Binary classification (male/female) + confidence + pitch
- **Features**: 80-dimensional log mel-filterbank
- **Source**: HuggingFace (`JaesungHuh/voice-gender-classifier`)

## Technical Highlights

### Signal Processing

- **Low-Pass Filter**: Moving average (adaptive order)
- **Downsampling**: Nearest neighbor
- **Upsampling**: Linear interpolation
- **WAV Generation**: Custom encoder with proper headers

### Anti-Aliasing
- **Cutoff**: Nyquist frequency (targetRate/2)
- **Purpose**: Prevent frequency folding
- **Effect**: Preserves classification accuracy

## Notes
- Model downloads automatically from HuggingFace on first use
- All audio resampled to 16kHz for classification
- Temporary files cleaned up automatically
- Anti-aliasing enabled by default (recommended)
- Works best with clear voice recordings (2-5 seconds)

## Troubleshooting
- **No microphone**: Check browser permissions
- **Classification fails**: Ensure backend server is running
- **No reconstructed audio**: Enable anti-aliasing toggle
- **Poor accuracy**: Use sample rates ≥8kHz

---

📚 **Full Documentation**: [VOICE_PROCESSING_DOCUMENTATION.md](VOICE_PROCESSING_DOCUMENTATION.md)
