# Signal Viewer & Disease Prediction System

A unified **Medical and Physical Signal Viewer** with intelligent AI-based abnormality detection.  
This repository combines **real-time visualization**, **interactive multi-mode analysis**, and **deep-learning-based disease prediction** for biomedical and physical signals such as **EEG**, **ECG**, **Doppler**, and **Radar**.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [EEG Signal Viewer & Disease Predictor](#eeg-signal-viewer--disease-predictor)
   - [Overview](#overview)
   - [Viewer Modes](#viewer-modes)
   - [EEGPT Model Integration](#eegpt-model-integration)
5. [ECG Real-Time Viewer](#ecg-real-time-viewer)
   - [Highlights](#highlights)
   - [Setup and Installation](#setup-and-installation)
   - [Files of Interest](#files-of-interest)
   - [How It Works](#how-it-works)
   - [Developer Notes & Tuning](#developer-notes--tuning)
   - [Application Interface](#application-interface)
6. [Doppler](#acoustic-signal-viewer-coming-soon)
7. [Radar](#rf-signal-viewer-coming-soon)
8. [Installation](#installation)
9. [Contributors](#contributors)
10. [License](#license)

---

## Introduction

The **Signal Viewer Project** is designed to visualize, analyze, and classify multiple signal types across biomedical and physical domains.  
It provides **multi-channel interactive visualization tools** combined with **AI-driven classification** that detects abnormalities in real-time.

![Intro animation](docs/images/intro_anim.gif)
*Animation: quick demo of the Signal Viewer interface.*

Each module (EEG, ECG, Radar, Doppler) includes:
- A **real-time multi-signal viewer** with multiple visualization modes.
- A **deep-learning model** trained on domain-specific data for abnormality detection.
- A **modular design** allowing users to plug in new models or visual modes.

---

## Features

 Multi-signal visualization (EEG, ECG)
 Real-time smooth plotting using optimized Python backends  
 Multiple visualization modes (Time, XOR, Polar, Recurrence)  
 Channel selection & color map customization  
 Integration with pretrained AI models for automatic abnormality detection  
 Interactive user interface with smooth playback, zoom, and pan controls  

---
## System Architecture

### **Backend Framework**
- **Flask** - Web framework with blueprint architecture
- **PyTorch** - Deep learning model inference
- **MNE-Python** - EEG signal processing
- **NumPy/SciPy** - Numerical computing and signal processing
- **Transformers** - Hugging Face models for audio classification

### **Frontend Technologies**
- **HTML5/CSS3** - Modern web interface
- **Tailwind CSS** - Responsive design framework
- **JavaScript** - Interactive functionality
- **Plotly.js** - Real-time data visualization
- **Chart.js** - Statistical plotting

## EEG Signal Viewer & Disease Predictor

### Overview
The **EEG Viewer** is a professional real-time visualization tool that supports multiple analysis modes for EEG data.  
It enables users to upload EEG recordings, visualize them in different modes, and automatically predict the neurological condition using a pretrained **EEGPT-based AI model**.

Upon uploading an EEG file:
- The system can detect **whether the signal is normal or abnormal**.
- If abnormal, it classifies it into one of four neurological conditions:
  - **Alzheimer**
  - **Epilepsy**
  - **Parkinson**
  - **Sleep Disorder**

---

### Viewer Modes

| Mode | Description | Key Features |
|------|--------------|---------------|
| **Time Domain (Default)** | Standard continuous-time plot with fixed viewport. | Play, pause, zoom, pan, speed control. |
| **Polar Mode** | Plots signal magnitude (r) against time (θ). | Can run as **fixed time window** or **cumulative** view. |
| **XOR Mode** | Divides signal into time chunks and overlays them using XOR logic. | Highlights differences between repeated patterns. |
| **Recurrence Plot** | Plots two channels (chX, chY) as a cumulative heat map. | Useful for visualizing synchrony and correlation. |


![EEG Viewer Demo](docs/images/eeg2.gif)
*Animated demo: EEG Time graph Viewer in action.*


![EEG Viewer Demo 2](docs/images/eeg3.gif)
*Animated demo: EEG polar graph Viewer in action.*


![EEG Viewer Demo 3](docs/images/eeg4.gif)
*Animated demo: EEG recurrence graph Viewer in action.*



Additional controls:
- Select one or more channels for display.
- For better visualization, hence the EEG signals of different channels look so different we decided to plot them seperately each one in its own graph.
- Band Power graph to illustrate the power of each frequency range (alpha/beta/delta/theta/gamma).
- Adjust time chunk width.
- Choose custom color maps for 2D representations.
- Polar graph can be cumulative plot to retain full history.
- Control speed as preferable.
- Zooming in and out for clearer visualization of signal details.



---

### EEGPT Model Integration

**Deep Learning Model: Custom EEGPT Fine-Tuning Implementation**

We developed and implemented a comprehensive **PyTorch Lightning** setup to fine-tune the powerful **EEGPT** (Electroencephalography Generative Pre-trained Transformer) for downstream neurological disease classification. This custom, production-ready implementation features:

* **Targeted Classification:** Successfully fine-tuned the model for four distinct diseases: **Alzheimer's**, **Epilepsy**,**Parkinson** and **Sleep Disorder**.
* **Custom PyTorch Data Pipeline:** Engineered a dedicated `DiseaseClassificationDataset` with automatic stratified data splitting (Train/Val/Test), mean centering, and normalization to ensure robust and reproducible training.
* **Model Adaptation:** Implemented a new **Classification Head** and a **Channel Adaptation Layer** to seamlessly connect the pretrained EEGPT encoder to our disease-specific tasks, supporting various channel configurations.
* **Optimized Training:** Leveraged **PyTorch Lightning** for advanced features, including **AdamW** optimization with **Cosine Annealing** scheduling, **Mixed Precision (16-bit)** for efficiency, and comprehensive logging (TensorBoard and CSV).
* **Robust Evaluation:** The system generates detailed evaluation metrics, including **Accuracy, Precision, Recall, F1-scores**, and **Confusion Matrices**, monitored in real-time.

**Supported Diseases (Fine-Tuned):**
* Alzheimer
* Epilepsy
* Sleep Disorder
* Parkinson

### Datasets Used

The following datasets were used for training and evaluation of the disease classifiers:

- Epilepsy: https://data.mendeley.com/datasets/5pc2j46cbc/1
- Alzheimer: https://data.mendeley.com/datasets/ch87yswbz4/1
- Sleep disorder: https://www.physionet.org/content/sleep-edfx/1.0.0/
- Parkinson: https://www.kaggle.com/datasets/s3programmer/parkison-diseases-eeg-dataset

Full source code and implementation details are available in its dedicated repository:

🔗 **[EEGPT Disease Classification Repository](https://github.com/YasmeenBadr/EEG-Model_for_disease_classification)**

---


## ECG Real-Time Viewer

This repository contains a Flask-based ECG real-time viewer and lightweight model prototypes for detecting abnormalities using both 1D time-domain signals and 2D recurrence-image representations.

## Highlights

- Live streaming ECG visualization (time domain, XOR diff, polar, recurrence colormap) using Plotly.
- Drag & drop upload of WFDB records (.hea/.dat/.xyz) to visualize and evaluate signals.
- Lightweight 1D CNN classifier (SimpleECG) for time-domain prediction.
- Lightweight 2D CNN classifier (Simple2DCNN) trained on recurrence-style 2D histograms generated from two-channel pairs.
- Background training of the 2D model on uploaded WFDB records (if .hea contains labels); recurrence data is saved to `results/recurrence_data/`.

## Setup and Installation


1. Clone the Repository:

   ```bash
   git clone https://github.com/YasmeenBadr/Task_1_DSP.git


2. Navigate to the Project Directory:

   ```bash
   cd Task_1_DSP


3. Install Required Dependencies:

   ```bash
   pip install -r requirements.txt


4. Run the Application:

   ```bash
   python app.py


After running, open your browser and go to:
👉 http://127.0.0.1:5000/ecg

## Files of interest

- `app.py` — Flask app bootstrap (registers the `ecg` blueprint).
- `signals/ecg.py` — Core streaming logic, prediction wrappers, recurrence image builder, and 2D training hooks.
- `templates/ecg.html` — Frontend UI, Plotly plots, controls (channel selection, XOR threshold, polar mode), drag & drop upload.
- `models/` — training artifacts and model weights (if present).
- `results/recurrence_data/` — CSV exports of the two-channel recurrence data saved prior to 2D training.

## How it works

- The browser polls `/ecg/update` with selected channels and visualization options. The server returns downsampled time series, XOR diffs (for single-channel), polar data, recurrence colormap data (for 2 channels), and predictions.
- 1D predictions: a rolling per-channel buffer is accumulated and passed to a small 1D CNN to predict Normal/Abnormal. Predictions are smoothed over a short window.
- 2D predictions: recurrence images are generated from two-channel pairs and used for a separate 2D CNN. Training runs in a background thread when a WFDB record with labels is loaded.

## Developer notes & tuning

- Smoothing window: `SMOOTH_WINDOW` in `signals/ecg.py` controls temporal averaging of probabilities.
- Minimum samples: `MIN_PRED_LEN` controls when the 1D model will run (helps avoid padding bias).
- Recurrence CSVs are written to `results/recurrence_data/` before training; useful for reproducibility.
------------------------
### Dataset used

https://www.physionet.org/content/ptbdb/1.0.0/

------------------------

## Videos & Screenshoots


![ECG Viewer Demo](ECG/ecg.gif)



![ECG Viewer Demo 2](ECG/ecg1.gif)
*Animated demo: ECG polar graph Viewer in action.*


![ECG Viewer Demo 3](ECG/ecg3.gif)
*Animated demo: ECG recurrence graph Viewer in action.*


![ECG Viewer Demo 3](ECG/xor.png)
*ECG XOR graph Viewer.*

