# ECG Module Overview (`signals/ecg.py`)

This document explains the structure, concepts, and logic of the ECG Flask blueprint and its visualization/prediction utilities. It mirrors the EEG documentation flow for consistency.

## Contents
- **[Blueprint and Globals](#blueprint-and-globals)**
- **[Downsampling and Aliasing](#downsampling-and-aliasing)**
- **[Prediction Models](#prediction-models)**
- **[Prediction Orchestration](#prediction-orchestration)**
- **[HTTP Routes](#http-routes)**
- **[Frontend Interaction](#frontend-interaction)**
- **[Key Parameters and Limits](#key-parameters-and-limits)**
- **[Troubleshooting](#troubleshooting)**

---

## Blueprint and Globals
- **Blueprint**: `ECG_BP = Blueprint("ecg", __name__, url_prefix="/ecg", template_folder="templates")`
  - Registers ECG routes under `/ecg`.

- **Global stream state `_stream`** stores:
  - `signals_raw`: the native (original) ECG sample matrix (samples × channels).
  - `signals`: the current working signal at the operating sampling rate.
  - `fs`: current operating sampling frequency (may be changed by user).
  - `fs_native`: original native sampling frequency of the record.
  - `pos_native`: current index for native stream cursor (circular).
  - `channels`: channel labels list.
  - `alias_phase`: persistent per-target-fs map for decimation phase continuity.
  - Rolling buffers for predictions and state for derived plots.

- **Defaults/caps**:
  - `FREQ_DEFAULT = 500`, `FREQ_MIN = 10`, `MAX_FREQ_LIMIT = 500` (cap used by endpoints).
  - `STREAMING_CHUNK_DURATION = 1.0` s (per request window size).

---

## Downsampling and Aliasing
Downsampling is intentionally performed without anti-aliasing to make aliasing artifacts visible and educational.

- **Shared implementation**: `signals/resampling.py`
  - `decimate_with_aliasing(sig, native_fs, target_fs, pos_native, phase_state)`
  - Integer factor k → take every k-th sample; non-integer → phase accumulator.
  - Uses `phase_state` to keep a persistent phase per `target_fs` so artifacts remain consistent across chunks.

- **Wrapper in ECG**: `resample_with_aliasing(sig, native_fs, target_fs, pos_native)`
  - Delegates to `decimate_with_aliasing` while passing `_stream["alias_phase"]` for persistence.

- **Where used**:
  - `/ecg/update`: per-request native chunk → decimate to current operating `fs` → plot and predict.
  - `/ecg/set_sampling` and `/ecg/set_freq`: rebuilds `_stream['signals']` at new operating `fs` from `signals_raw`.
  - `/ecg/reset_sampling`: restores `_stream['signals']` and `fs` to native.

Effects:
- Lower `fs` → reduced detail, visible waveform changes (e.g., R-peak shape), potential prediction shifts.
- Frontend always shows server-reported `used_sampling_freq` to avoid UI/logic drift.

---

## Prediction Models
Two demo models are bundled:

- **1D model (SimpleECG)**
  - A tiny CNN that classifies single-channel sequences as Normal/Abnormal.
  - Input length `_model_seq_len` (e.g., 5000 samples). Buffers maintain the most recent samples at the current `fs`.

- **2D model (Simple2DCNN)**
  - A small CNN that classifies recurrence-density images built from two channels.
  - Can be trained in the background using `.hea`-extracted diagnosis text as labels.

Both models are for demonstration. The pipeline normalizes inputs, handles short buffers ("Waiting" state), and smooths predictions across a short history window.

---

## Prediction Orchestration
- `predict_signal(sig_chunk)` runs the 1D model for selected channels, returning labels, probabilities, and confidence.
- Recurrence helpers build 2D density images and optionally run the 2D model.
- In `/ecg/update`, predictions are produced from the decimated (displayed) signal to faithfully reflect aliasing effects seen by the user.

---

## HTTP Routes
- **`GET /ecg/` → `index()`**
  - Renders `templates/ecg.html` ECG viewer.

- **`GET /ecg/config` → `config()`**
  - Returns UI configuration:
    ```json
    {
      "fs": <current_fs>,
      "fs_native": <native_fs>,
      "display_fs": <display_fs>,
      "channels": ["Lead 1", ...],
      "default_time_window_s": 15.0,
      "freq_default": 500,
      "freq_min": 10,
      "hea_diagnosis": "..."
    }
    ```

- **`POST /ecg/update` → `update()`** (main streaming endpoint)
  - Request JSON:
    ```json
    {
      "channels": [0,1],
      "width": 5.0,
      "polar_mode": "fixed" | "cumulative",
      "xor_threshold": 0.05,
      "sampling_freq": 200
    }
    ```
  - Processing:
    - Extract native chunk using circular buffer at `pos_native`.
    - Decimate to current operating `fs` with phase-aware aliasing.
    - Update prediction buffers, compute Time/XOR/Polar/Recurrence data, run predictions.
  - Response (abridged):
    ```json
    {
      "signals": {"0": [..], "1": [..]},
      "display_fs": 200,
      "used_sampling_freq": 200,
      "native_fs": 500,
      "prediction": {"label": "Normal", "confidence": 0.83, ...},
      "recurrence_prediction": {"label": "...", "confidence": 0.72},
      "xor": {"0": [..]},
      "polar": {"0": {"r": [..], "theta": [..]}},
      "recurrence_scatter": {"x_vals": [..], "y_vals": [..]},
      "recurrence_colormap": [[..]]
    }
    ```

- **`POST /ecg/set_sampling`** (alias: **`POST /ecg/set_freq`**)
  - Accepts `{ "sampling_freq": <Hz> }` or `{ "frequency": <Hz> }`.
  - Clamps to `[FREQ_MIN, fs_native, 500]`, rebuilds `_stream['signals']` via aliasing decimation, updates `_stream['fs']`.

- **`POST /ecg/reset_sampling`**
  - Restores `_stream['signals']` and `_stream['fs']` to native; resets alias phase map.

---

## Frontend Interaction
- The UI (`templates/ecg.html`) workflow:
  1. `GET /ecg/config` to initialize native/fs and channels; render channel checkboxes and set slider caps.
  2. On file upload, `POST /ecg/upload` (handled in the page JS) and re-run config.
  3. While streaming, `fetchData()` posts `{channels, width, polar_mode, xor_threshold, sampling_freq}` to `/ecg/update`.
  4. The server responds with plot-ready arrays and metadata; the UI renders Time, XOR (single-channel), Polar, and Recurrence.
  5. The UI mirrors `used_sampling_freq` (server truth) into the label to keep the slider and plots consistent.

- Notes:
  - XOR mode requires exactly one channel; Recurrence requires exactly two.
  - The UI no longer updates XOR in 'all' mode (fallback removed) — XOR only in its dedicated mode.

---

## Key Parameters and Limits
- `FREQ_MIN = 10` Hz minimum display/operating frequency.
- `MAX_FREQ_LIMIT = 500` Hz maximum (UI cap; also applied server-side).
- `STREAMING_CHUNK_DURATION = 1.0` s per update.
- `_model_seq_len = 5000` for the 1D model input length.

---

## Troubleshooting
- "No signals loaded": Upload `.hea` and `.dat` (same basename), or use simulated mode.
- Recurrence blank: Select exactly two channels.
- XOR blank: Select exactly one channel and wait for the second chunk (needs previous window).
- Prediction stuck on "Waiting": Continue streaming until buffers fill to `_model_seq_len`; very low sampling slows this.
- Sampling label mismatch: The UI shows `used_sampling_freq` from the server; if different from the slider, the server clamped it ≤ native ≤ 500.
