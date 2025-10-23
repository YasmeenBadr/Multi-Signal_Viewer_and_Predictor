# EEG Module Overview (`signals/eeg.py`)

This document explains the structure and logic of the EEG Flask blueprint and related processing/prediction utilities.

## Contents
- **[Blueprint and Globals](#blueprint-and-globals)**
- **[DSP Helpers](#dsp-helpers)**
- **[Prediction Models](#prediction-models)**
- **[Prediction Orchestration](#prediction-orchestration)**
- **[HTTP Routes](#http-routes)**
- **[Frontend Interaction](#frontend-interaction)**

---

## Blueprint and Globals
- **Blueprint**: `bp = Blueprint("eeg", __name__, template_folder="../templates")`
  - Registers EEG routes under `/eeg`.

- **Global EEG state**: `class EEGData` stores:
  - `raw`: loaded MNE Raw object
  - `fs`: native sampling frequency
  - `n_times`: total samples
  - `ch_names`: channel names
  - `current_index`: stream position

- **Streaming params**:
  - `INITIAL_OFFSET_SAMPLES = fs * 10` (start ~10s into the record)
  - `CHUNK_SAMPLES = fs / 10` (~100ms per update)

- **XOR buffers** (server-side assistance for XOR visualization):
  - `_XOR_BUFFERS: Dict[int, List[float]]`
  - `_XOR_PREV_WINDOWS: Dict[int, List[float]]`

- **EEG Bands**: `BANDS = {Delta, Theta, Alpha, Beta, Gamma}` with their frequency ranges.

---

## DSP Helpers
- **`butter_bandpass(lowcut, highcut, fs, order=2)`**
  - Designs IIR filter coefficients for a band given the native `fs`.
  - Special-case: Delta band treated as lowpass. Returns `(b, a)` or `(None, None)` on invalid ranges.

- **`calculate_band_power(data, fs)`**
  - For each band in `BANDS`:
    - Designs filter via `butter_bandpass()`.
    - Applies zero-phase `filtfilt`.
    - Computes mean-squared value (power) and scales by a large factor for display.
  - Returns `{band_name: power}`.

---

## Prediction Models
Each model either loads a simple Torch MLP (if a checkpoint is found) or falls back to handcrafted feature analysis.

Common flow in `predict(...)`:
- Flatten to 1D, size/pad to 1024 samples.
- Z-score normalize.
- If model present: softmax → class, confidence, class name.
- Else: call `_analyze_*_patterns(...)` heuristic and threshold.

- **`EpilepsyPredictor`**
  - Heuristics focus on spikes/sharp waves, PSD (20–40 Hz), and amplitude asymmetry.

- **`AlzheimerPredictor`**
  - Heuristics emphasize spectral ratios: reduced alpha, increased theta, plus entropy.

- **`SleepDisorderPredictor`**
  - Uses PSD ratios, spindle power (11–15 Hz), K-complex proxy (0.5–2 Hz), conservative scoring.

- **`ParkinsonPredictor`**
  - Uses beta suppression (13–30 Hz), tremor band (4–6 Hz), theta ratio, entropy.

---

## Prediction Orchestration
- **`run_all_predictions(eeg_data)`**
  - Calls all four predictors and collects positive predictions.
  - Ranks positives by confidence; returns only the highest if above a threshold; otherwise reports all as Normal/Healthy.
  - Output schema:
```
{
  'epilepsy': {'predicted_class', 'confidence', 'class_name'},
  'alzheimer': {...},
  'sleep_disorder': {...},
  'parkinson': {...}
}
```

---

## HTTP Routes
- **`GET /eeg/` → `eeg_home()`**
  - Renders `templates/eeg.html` EEG viewer.

- **`POST /eeg/upload` → `upload_file()`**
  - Accepts `.edf` or `.fif/.fif.gz`.
  - Loads via MNE, sets `eeg_data.fs`, `n_times`, `ch_names`.
  - Initializes streaming (`INITIAL_OFFSET_SAMPLES`, `CHUNK_SAMPLES`).
  - Returns `{success, message, channels: {idx: name}, fs}`.

- **`POST /eeg/update` → `update()`**
  - Request JSON: `{channels: [int], mode: str, width: float}`.
  - Returns a chunk for the selected channels:
    - `signals`: `{channel_index_str: [samples...]}`
    - `n_samples`: number of samples in this chunk
    - `band_power`: averaged over the selected channels
    - `xor` (optional): server-side XOR series for single-channel XOR mode
  - Maintains `eeg_data.current_index` with wrap-around.

- **`POST /eeg/predict` → `predict_diseases()`**
  - Request JSON: `{channels: [int], downsample_factor: int}`
  - Takes the current chunk from the first selected channel.
  - Applies naive downsampling (`[::factor]`) to simulate aliasing if requested.
  - Calls `run_all_predictions(...)` and returns prediction results.

---

## Frontend Interaction
- The UI (`templates/eeg.html`) uploads a file, receives `fs` and channels, and polls `/eeg/update` ~every 100ms.
- Plots the signals over a sliding time window `width`.
- Shows averaged band power from the backend.
- XOR mode can use the optional `xor` array.
- Prediction panel calls `/eeg/predict` with the current downsample factor.

---

## Notes
- `CHUNK_SAMPLES = fs/10` ties update cadence to the file’s native sampling rate.
- Band power uses the native `fs` to keep spectral calculations correct.
- Client visualization may resample for display, but backend data stays at native resolution per chunk.
