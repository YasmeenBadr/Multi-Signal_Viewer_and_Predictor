# Voice Gender Classification Feature

## Overview
Added gender classification functionality to the Voice Processing Suite using the ECAPA-TDNN model.

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

## How to Use
1. Upload an audio file (MP3/WAV)
2. Play the original or resampled audio
3. Click "Classify Gender" button
4. View the classification results

## Model Details
- **Architecture**: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
- **Input**: 16kHz audio samples
- **Output**: Binary classification (male/female)
- **Features**: 80-dimensional log mel-filterbank features

## Notes
- The model automatically downloads from HuggingFace on first use
- Audio is resampled to 16kHz for classification
- Pitch analysis uses librosa's piptrack algorithm
- Temporary files are cleaned up after classification
