ECG Real-Time Viewer
=====================

This repository contains a Flask-based ECG real-time viewer and lightweight model prototypes for detecting abnormalities using both 1D time-domain signals and 2D recurrence-image representations.

Highlights
----------
- Live streaming ECG visualization (time domain, XOR diff, polar, recurrence colormap) using Plotly.
- Drag & drop upload of WFDB records (.hea/.dat/.xyz) to visualize and evaluate signals.
- Lightweight 1D CNN classifier (SimpleECG) for time-domain prediction.
- Lightweight 2D CNN classifier (Simple2DCNN) trained on recurrence-style 2D histograms generated from two-channel pairs.
- Background training of the 2D model on uploaded WFDB records (if .hea contains labels); recurrence data is saved to `results/recurrence_data/`.

Setup and Installation
----------

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

Files of interest
-----------------
- `app.py` — Flask app bootstrap (registers the `ecg` blueprint).
- `signals/ecg.py` — Core streaming logic, prediction wrappers, recurrence image builder, and 2D training hooks.
- `templates/ecg.html` — Frontend UI, Plotly plots, controls (channel selection, XOR threshold, polar mode), drag & drop upload.
- `models/` — training artifacts and model weights (if present).
- `results/recurrence_data/` — CSV exports of the two-channel recurrence data saved prior to 2D training.

How it works
------------
- The browser polls `/ecg/update` with selected channels and visualization options. The server returns downsampled time series, XOR diffs (for single-channel), polar data, recurrence colormap data (for 2 channels), and predictions.
- 1D predictions: a rolling per-channel buffer is accumulated and passed to a small 1D CNN to predict Normal/Abnormal. Predictions are smoothed over a short window.
- 2D predictions: recurrence images are generated from two-channel pairs and used for a separate 2D CNN. Training runs in a background thread when a WFDB record with labels is loaded.

Developer notes & tuning
------------------------
- Smoothing window: `SMOOTH_WINDOW` in `signals/ecg.py` controls temporal averaging of probabilities.
- Minimum samples: `MIN_PRED_LEN` controls when the 1D model will run (helps avoid padding bias).
- Recurrence CSVs are written to `results/recurrence_data/` before training; useful for reproducibility.

Application Interface
------------------------

https://github.com/user-attachments/assets/3347061f-21d7-40a4-aa6c-54c7df7f7570



https://github.com/user-attachments/assets/fa16a8ff-3014-48c1-a356-1e56812666d5

<img width="1335" height="562" alt="Screenshot 2025-10-11 001541" src="https://github.com/user-attachments/assets/af41933f-1758-4421-ab8a-1e4141efc931" />
<img width="866" height="654" alt="Screenshot 2025-10-11 001508" src="https://github.com/user-attachments/assets/c1a4578d-51c0-4b03-8e02-a2f6ca763c0b" />



