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



## ❤️ ECG Real-Time Viewer

Highlights

- Live streaming ECG visualization (time domain, XOR diff, polar, recurrence colormap) using Plotly.

- Drag & drop upload of WFDB records (.hea, .dat, .xyz) to visualize and evaluate signals.

- Lightweight 1D CNN classifier (SimpleECG) for time-domain prediction.

- Lightweight 2D CNN classifier (Simple2DCNN) trained on recurrence-style 2D histograms from two-channel pairs.

- Background training of the 2D model when labeled data is uploaded; recurrence data saved in results/recurrence_data/.

Setup and Installation

Clone the Repository

Bash

```bash
git clone https://github.com/YasmeenBadr/Task_1_DSP.git
```

Navigate to the Project Directory

Bash

```bash
cd Task_1_DSP
```

Install Required Dependencies

Bash

```bash
pip install -r requirements.txt
```

Run the Application

Bash

```bash
python app.py
```

After running, open your browser and go to:

👉 http://127.0.0.1:5000/ecg

Files of Interest

File	Description
app.py	Flask app bootstrap (registers the ECG blueprint).
signals/ecg.py	Core streaming logic, prediction wrappers, recurrence image builder, and 2D training hooks.
templates/ecg.html	Frontend UI, Plotly plots, controls (channel selection, XOR threshold, polar mode), drag & drop upload.
models/	Training artifacts and model weights (if present).
results/recurrence_data/	CSV exports of recurrence data saved before 2D training.

Export to Sheets

How It Works

The browser polls /ecg/update with selected channels and visualization options.

The server returns:

Downsampled time series

XOR diffs (for single-channel)

Polar data

Recurrence colormap data (for two channels)

Predictions from 1D and 2D models

Prediction Flow:

1D Predictions:
A rolling per-channel buffer is passed to a small 1D CNN to predict Normal/Abnormal. Predictions are smoothed over a short window.

2D Predictions:
Recurrence images are generated from two-channel pairs and used for a separate 2D CNN.

Training runs in a background thread when labeled WFDB records are loaded.

Developer Notes & Tuning

Parameter	Description
SMOOTH_WINDOW	Controls temporal averaging of probabilities.
MIN_PRED_LEN	Minimum samples before 1D model runs (prevents padding bias).
results/recurrence_data/	Stores CSV recurrence data for reproducibility.

Export to Sheets

Application Interface

Preview	Description
ECG dashboard with multiple visualization options	Recurrence plot and real-time prediction

Export to Sheets

### Videos & Screenshots

Below are demo videos and example screenshots for the ECG Real-Time Viewer. Replace the placeholders with your actual URLs or local image paths.

- Demo Video 1: https://github.com/user-attachments/assets/3347061f-21d7-40a4-aa6c-54c7df7f7570
- Demo Video 2: https://github.com/user-attachments/assets/fa16a8ff-3014-48c1-a356-1e56812666d5

Screenshots (place image files in `docs/images/` or `uploads/` and reference them with the relative path):

<img width="1335" height="562" alt="Screenshot 2025-10-11 001541" src="https://github.com/user-attachments/assets/af41933f-1758-4421-ab8a-1e4141efc931" />
*Figure: ECG Viewer displaying the time-domain plot.*

<img width="866" height="654" alt="Screenshot 2025-10-11 001508" src="https://github.com/user-attachments/assets/c1a4578d-51c0-4b03-8e02-a2f6ca763c0b" />
*Figure: ECG Viewer showing a recurrence colormap used for 2D model input.*

🌊 Acoustic Signal Viewer (Coming Soon)
Vehicle-passing Doppler effect simulator and detector

Drone/submarine sound identification models

📡 RF Signal Viewer (Coming Soon)
SAR/Cosmic signal visualization and analysis tools

🛠️ Installation
Ensure you have Python 3.9+ and install all dependencies:

Bash

```bash
pip install -r requirements.txt
```

Supports both CPU and CUDA-enabled GPU environments.

👥 Contributors

Yasmeen Badr — EEG & ECG Viewer Development, Model Integration

Team Members — Signal Processing, Model Fine-tuning, Visualization

📄 License

This project is released under the MIT License.

See LICENSE for more details.
