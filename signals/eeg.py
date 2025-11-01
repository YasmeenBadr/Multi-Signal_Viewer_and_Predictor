# eeg_refactored.py
"""
EEG Signal Processing and Disease Prediction Module

Provides real-time EEG visualization, multi-disease prediction (Epilepsy, 
Alzheimer's, Sleep Disorders, Parkinson's), and signal analysis.
"""

import os
import logging
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, welch

from flask import Blueprint, request, jsonify, render_template

# Import MNE for EEG file loading
try:
    import mne
except ImportError:
    mne = None

# Import shared utilities
from shared_utils import (
    RollingBuffer, normalize_signal,
    format_prediction_response, format_streaming_response,
    validate_channels
)
from .resampling import decimate_with_aliasing, resample_signal


# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eeg")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Streaming parameters
BASE_CHUNK_SAMPLES = 16
INITIAL_OFFSET_SAMPLES = 0  # Will be set after file load


# ============================================================================
# GLOBAL STATE
# ============================================================================

class EEGState:
    """Centralized state management for EEG streaming."""
    
    def __init__(self):
        # MNE Raw handle (set after a successful file load)
        self.raw = None
        
        # Signal properties discovered from the file
        self.fs = 160              # Display/native sampling frequency (Hz)
        self.n_times = 0           # Total number of samples available
        self.ch_names = []         # Ordered list of channel labels
        
        # Streaming cursor state for incremental chunking
        self.current_index = 0     # Current read head position in samples
        self.initial_offset = 0    # Skip an initial segment to avoid headers/artifacts
        
        # Server-side XOR rolling context per-channel
        self.xor_buffers = {}      # Channel -> rolling buffer of recent samples
        self.xor_prev_windows = {} # Channel -> last full window used for XOR diff
        
        # Indicates whether a valid file is loaded and ready
        self.loaded = False
    
    def reset_streaming_state(self):
        """Reset streaming-related state."""
        # Rewind to the initial offset for a clean restart
        self.current_index = self.initial_offset
        # Clear XOR buffers and previous windows to avoid stale state
        self.xor_buffers = {}
        self.xor_prev_windows = {}
    
    def reset_all(self):
        """Reset all state."""
        # Reinitialize to default values (equivalent to creating a new instance)
        self.__init__()


# Global state instance
state = EEGState()


# ============================================================================
# SIGNAL PROCESSING UTILITIES
# ============================================================================


def calculate_xor_difference_eeg(current_buffer: List[float], 
                                 previous_window: List[float],
                                 chunk_size: int) -> List[float]:
    """
    Calculate thresholded XOR difference for EEG signals.
    
    Args:
        current_buffer: Current signal buffer
        previous_window: Previous window for comparison
        chunk_size: Size of analysis window
    
    Returns:
        XOR difference signal
    """
<<<<<<< Updated upstream
    #Not enough data yet → Return as-is (no comparison possible) 
    if len(current_buffer) < chunk_size:
        return current_buffer
    #First iteration OR size mismatch
    # Nothing to compare against → Return current buffer
=======
    # Require at least one full window of data in the current buffer
    if len(current_buffer) < chunk_size:
        return current_buffer
    
    # If no valid previous window, echo the current window (no diff yet)
>>>>>>> Stashed changes
    if not previous_window or len(previous_window) != chunk_size:
        return current_buffer
    
    # Take last chunk_size samples from buffer
    current_window = current_buffer[-chunk_size:]
    
    # Calculate statistics for dynamic threshold
    mean = np.mean(current_window)         # Unused but illustrative for extensions
    std = np.std(current_window)
    threshold = std * 0.1                  # 10% of standard deviation as sensitivity
    
    # Compute thresholded difference
    xor_result = []
    for i in range(chunk_size):
<<<<<<< Updated upstream
        curr_val = current_window[i]  # Sample from current window
        prev_val = previous_window[i] # Same position in previous window
        distance = abs(curr_val - prev_val)
=======
        curr_val = current_window[i]       # Current sample value
        prev_val = previous_window[i]      # Previous window's corresponding sample
        distance = abs(curr_val - prev_val) # Absolute difference (magnitude change)
>>>>>>> Stashed changes
        
        # Keep only salient differences: values over threshold
        xor_result.append(distance if distance > threshold else 0)
    
    return xor_result


# ============================================================================
# DISEASE PREDICTION MODELS
# ============================================================================

def _prepare_input_1d(eeg_data: np.ndarray, target_len: int = 1024) -> np.ndarray:
    # Accept raw EEG data as numpy array (1D or 2D) and ensure a fixed-length 1D vector
    arr = eeg_data  # Local reference to avoid mutating caller's object
    # If the array is 2D (e.g., [channels, samples]), flatten to 1D sequence
    if arr.ndim == 2:
        arr = arr.flatten()
    # If longer than the model input length, truncate to target_len
    if len(arr) > target_len:
        arr = arr[:target_len]
    # If shorter, pad with zeros at the tail to reach target_len
    elif len(arr) < target_len:
        arr = np.pad(arr, (0, target_len - len(arr)))
    # Normalize to zero-mean, unit-variance to stabilize model inputs
    arr = normalize_signal(arr, method='zscore')
    # Return float32 view to match torch default tensor dtype expectations
    return arr.astype(np.float32, copy=False)

def _predict_softmax2(model: nn.Module, x1d: np.ndarray, device: torch.device, class_names: List[str]) -> Tuple[int, float, str]:
    # Convert numpy input to a torch batch tensor of shape [1, D]
    tensor = torch.from_numpy(x1d).unsqueeze(0).to(device)
    # Inference-only context: disable gradients for speed/memory
    with torch.no_grad():
        # Forward pass through the model to obtain raw logits
        logits = model(tensor)
        # Convert logits to probabilities across 2 classes with softmax
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        # Argmax to get predicted class index (0 or 1)
        pred_idx = int(np.argmax(probs))
        # Confidence is the probability of the predicted class
        conf = float(probs[pred_idx])
    # Map index to human-readable class name and return
    return pred_idx, conf, class_names[pred_idx]

class SimpleDiseasePredictor(nn.Module):
    """Simple neural network for disease prediction."""
    
    def __init__(self, input_size: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.net(x)


class EpilepsyPredictor:
    """Epilepsy detection from EEG patterns."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        # Select computation device (CPU/GPU) based on availability or explicit choice
        self.device = self._get_device(device)
        # Optional path to a serialized model checkpoint
        self.model_path = model_path
        # Lazy-loaded model handle (initialized on first use)
        self.model = None
        # Two-class names aligned with model output indices [0,1]
        self.class_names = ['Normal', 'Epilepsy']
    
    def _get_device(self, device: str) -> torch.device:
        # Resolve 'auto' to CUDA if available, otherwise CPU
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Return a torch.device instance for downstream use
        return torch.device(device)
    
    def _load_model(self):
        """Load model if available."""
        # If already loaded, no action needed
        if self.model is not None:
            return
        
        # Load a lightweight demo model and hydrate weights from checkpoint if provided
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Create the model architecture on the chosen device
                self.model = SimpleDiseasePredictor().to(self.device)
                # Load checkpoint from disk (supports raw state_dict or wrapped dict)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                # Tolerate minor key mismatches by setting strict=False
                self.model.load_state_dict(state_dict, strict=False)
                # Switch to eval mode for deterministic layers like Dropout
                self.model.eval()
                logger.info(f"Loaded epilepsy model from {self.model_path}")
            except Exception as e:
                # On any failure, fall back to heuristic path
                logger.warning(f"Failed to load epilepsy model: {e}")
                self.model = None
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """
        Predict epilepsy from EEG data.
        
        Args:
            eeg_data: EEG signal data
        
        Returns:
            Tuple of (predicted_class, confidence, class_name)
        """
        # Ensure model is loaded once before attempting inference
        self._load_model()
        
        # Use pattern analysis if model not available
        if self.model is None:
            # Fall back to deterministic heuristic scoring when no model is present
            epilepsy_score = self._analyze_epilepsy_patterns(eeg_data)
            
            if epilepsy_score > 0.7:
                return 1, epilepsy_score, "Epilepsy"
            else:
                return 0, 1 - epilepsy_score, "Normal"
        
        # Use model prediction
        try:
<<<<<<< Updated upstream
            # Preprocess
            #Converts 2D EEG data (multiple channels × time) to 1D 
            #because  Neural network expects flat input vector
            if eeg_data.ndim == 2:
                eeg_data = eeg_data.flatten()


            # Resize to model input to keep exactly 1024 input features for the neural network
            if len(eeg_data) > 1024:
                eeg_data = eeg_data[:1024]
            elif len(eeg_data) < 1024:
                eeg_data = np.pad(eeg_data, (0, 1024 - len(eeg_data)))
            
            # Normalize
            eeg_data = normalize_signal(eeg_data, method='zscore')
            
            # Predict
            tensor = torch.from_numpy(eeg_data.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
                predicted_class = 1 if probs[1] > 0.5 else 0
                confidence = max(probs[0], probs[1])
                class_name = self.class_names[predicted_class]
                
                return predicted_class, confidence, class_name
=======
            # Preprocess raw EEG into a fixed-length normalized 1D vector
            x1d = _prepare_input_1d(eeg_data, target_len=1024)
            # Run a two-class softmax prediction and decode to labels
            predicted_class, confidence, class_name = _predict_softmax2(self.model, x1d, self.device, self.class_names)
            return predicted_class, confidence, class_name
>>>>>>> Stashed changes
        
        except Exception as e:
            # Log and return a conservative default in case of model errors
            logger.error(f"Epilepsy prediction error: {e}")
            return 0, 0.7, "Normal"
    
    def _analyze_epilepsy_patterns(self, eeg_data: np.ndarray) -> float:
        """Analyze EEG for epilepsy-specific patterns."""
        try:
            # Statistical features
            #Calculate standard deviation of EEG signal
            std_amp = np.std(eeg_data)
            
            # Spike detection ->amplitude jumps
            diff = np.diff(eeg_data)
            spike_threshold = std_amp * 2
            spikes = np.sum(np.abs(diff) > spike_threshold) 
            spike_ratio = spikes / len(diff) if len(diff) > 0 else 0
            
            # Sharp waves
            sharp_threshold = std_amp * 1.5
            sharp_waves = np.sum(np.abs(diff) > sharp_threshold)
            sharp_ratio = sharp_waves / len(diff) if len(diff) > 0 else 0
            
            # High-frequency activity (seizure-like)
            freqs, psd = welch(eeg_data, fs=160, nperseg=min(256, len(eeg_data)//4))
            seizure_power = np.sum(psd[(freqs >= 20) & (freqs <= 40)])
            total_power = np.sum(psd)
            seizure_ratio = seizure_power / total_power if total_power > 0 else 0
            
            # Amplitude asymmetry
            mean_amp = np.mean(np.abs(eeg_data))
            amplitude_asymmetry = np.std(np.abs(eeg_data)) / (mean_amp + 1e-6)
            
            # Calculate score
            score = min(1.0, (
                spike_ratio * 1.5 +
                sharp_ratio * 1.2 +
                seizure_ratio * 1.2 +
                min(amplitude_asymmetry * 0.1, 0.2)
            ))
            
            # Boost if multiple indicators
            if spike_ratio > 0.05 and sharp_ratio > 0.05:
                score = min(1.0, score * 1.3)
            
            return score
        
        except Exception:
            return 0.1


class AlzheimerPredictor:
    """Alzheimer's disease detection from EEG patterns."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        # Choose device automatically if requested; otherwise respect explicit device
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        # Optional checkpoint path (if absent -> heuristic only)
        self.model_path = model_path
        # Lazy-initialized model handle
        self.model = None
        # Output label names ordered by model logit indices
        self.class_names = ['Normal', 'Alzheimer']
    
    def _load_model(self):
        """Load model if available."""
        # Skip if already loaded
        if self.model is not None:
            return
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Instantiate demo classifier and hydrate from checkpoint
                self.model = SimpleDiseasePredictor().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                # Switch to eval for deterministic behavior
                self.model.eval()
                logger.info(f"Loaded Alzheimer model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Alzheimer model: {e}")
                self.model = None
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """Predict Alzheimer's from EEG data."""
        self._load_model()
        
        if self.model is None:
            # Heuristic path: compute score from PSD/entropy patterns
            alzheimer_score = self._analyze_alzheimer_patterns(eeg_data)
            
            if alzheimer_score > 0.2:
                return 1, alzheimer_score, "Alzheimer"
            else:
                return 0, alzheimer_score, "Normal"
        
        # Model-based prediction (similar structure as epilepsy)
        try:
            # Normalize and shape to fixed-length vector
            x1d = _prepare_input_1d(eeg_data, target_len=1024)
            # Softmax-based 2-class inference
            predicted_class, confidence, class_name = _predict_softmax2(self.model, x1d, self.device, self.class_names)
            return predicted_class, confidence, class_name
        
        except Exception as e:
            logger.error(f"Alzheimer prediction error: {e}")
            return 0, 0.75, "Normal"
    
    def _analyze_alzheimer_patterns(self, eeg_data: np.ndarray) -> float:
        """Analyze EEG for Alzheimer's patterns."""
        try:
            # Check for flat data
            if np.std(eeg_data) < 1e-6:
                return 0.05
            
            # Frequency analysis
            freqs, psd = welch(eeg_data, fs=160, nperseg=min(256, len(eeg_data)//4))
            
            delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
            theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
            beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
            
            total_power = delta_power + theta_power + alpha_power + beta_power
            
            if total_power > 0:
                alpha_ratio = alpha_power / total_power
                theta_ratio = theta_power / total_power
                delta_ratio = delta_power / total_power
                
                # Entropy
                psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                
                # Check for epilepsy-like patterns
                diff = np.diff(eeg_data)
                std_amp = np.std(eeg_data)
                spikes = np.sum(np.abs(diff) > std_amp * 2)
                spike_ratio = spikes / len(diff) if len(diff) > 0 else 0
                
                # Alzheimer score
                score = min(1.0, (
                    (1 - alpha_ratio) * 3.0 +
                    theta_ratio * 2.0 +
                    delta_ratio * 1.0 +
                    (entropy / 8) * 0.5
                ))
                
                # Reduce if epilepsy patterns present
                if spike_ratio > 0.1:
                    score *= 0.3
                
                # Boost for cognitive decline patterns
                if alpha_ratio < 0.3 and theta_ratio > 0.2:
                    score = min(1.0, score * 1.5)
                
                return score
            
            return 0.1
        
        except Exception:
            return 0.1


class SleepDisorderPredictor:
    """Sleep disorder detection from EEG patterns."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        # Auto-select device when requested
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        # Optional on-disk model path
        self.model_path = model_path
        # Lazy-loaded model
        self.model = None
        # Two-class naming scheme
        self.class_names = ['Normal', 'Sleep Disorder']
    
    def _load_model(self):
        # Avoid reloading repeatedly
        if self.model is not None:
            return
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Create simple classifier and load checkpoint
                self.model = SimpleDiseasePredictor().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                # Ensure eval mode for inference
                self.model.eval()
                logger.info(f"Loaded sleep disorder model")
            except Exception as e:
                logger.warning(f"Failed to load sleep disorder model: {e}")
                self.model = None
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """Predict sleep disorder from EEG data."""
        self._load_model()
        
        if self.model is None:
            # Heuristic estimation based on band ratios and spindles
            score = self._analyze_sleep_patterns(eeg_data)
            
            if score > 0.95:
                return 1, score, "Sleep Disorder"
            else:
                return 0, 1 - score, "Normal"
        
        # Model-based prediction
        try:
            # Standardized preprocessing + softmax inference
            x1d = _prepare_input_1d(eeg_data, target_len=1024)
            predicted_class, confidence, class_name = _predict_softmax2(self.model, x1d, self.device, self.class_names)
            return predicted_class, confidence, class_name
        
        except Exception as e:
            logger.error(f"Sleep disorder prediction error: {e}")
            return 1, 0.65, "Sleep Disorder"
    
    def _analyze_sleep_patterns(self, eeg_data: np.ndarray) -> float:
        """Analyze EEG for sleep disorder patterns."""
        try:
            if np.std(eeg_data) < 1e-6:
                return 0.05
            
            freqs, psd = welch(eeg_data, fs=160, nperseg=min(256, len(eeg_data)//4))
            
            delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
            theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
            beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
            
            total_power = delta_power + theta_power + alpha_power + beta_power
            
            if total_power > 0:
                delta_ratio = delta_power / total_power
                
                # Sleep spindles (reduced in sleep disorders)
                spindle_power = np.sum(psd[(freqs >= 11) & (freqs <= 15)])
                spindle_ratio = spindle_power / total_power
                
                # K-complexes
                k_complex_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 2)])
                k_complex_ratio = k_complex_power / total_power
                
                # Entropy
                psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                
                # Check for spikes (epilepsy-like)
                diff = np.diff(eeg_data)
                std_amp = np.std(eeg_data)
                spikes = np.sum(np.abs(diff) > std_amp * 2)
                spike_ratio = spikes / len(diff) if len(diff) > 0 else 0
                
                # Score (very conservative)
                score = min(1.0, (
                    (1 - spindle_ratio) * 0.5 +
                    delta_ratio * 0.3 +
                    k_complex_ratio * 0.4 +
                    (entropy / 25) * 0.05
                ))
                
                # Reduce if epilepsy patterns
                if spike_ratio > 0.01:
                    score *= 0.1
                
                # Boost for strong indicators
                if spindle_ratio < 0.005 and delta_ratio > 0.8:
                    score = min(1.0, score * 1.1)
                
                return score
            
            return 0.1
        
        except Exception:
            return 0.1


class ParkinsonPredictor:
    """Parkinson's disease detection from EEG patterns."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        # Auto device or explicit device
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        # Optional checkpoint path
        self.model_path = model_path
        # Lazy-loaded model reference
        self.model = None
        # Order-locked label list
        self.class_names = ['Healthy', 'Parkinson']
    
    def _load_model(self):
        # Do nothing if model already resident
        if self.model is not None:
            return
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Construct classifier and load weights
                self.model = SimpleDiseasePredictor().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                # Switch to evaluation mode
                self.model.eval()
                logger.info(f"Loaded Parkinson model")
            except Exception as e:
                logger.warning(f"Failed to load Parkinson model: {e}")
                self.model = None
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """Predict Parkinson's from EEG data."""
        self._load_model()
        
        if self.model is None:
            # Heuristic path using band/entropy/tremor proxies
            score = self._analyze_parkinson_patterns(eeg_data)
            
            if score > 0.95:
                return 1, score, "Parkinson"
            else:
                return 0, 1 - score, "Healthy"
        
        # Model-based prediction
        try:
            # Use the same standardized inference path for consistency
            x1d = _prepare_input_1d(eeg_data, target_len=1024)
            predicted_class, confidence, class_name = _predict_softmax2(self.model, x1d, self.device, self.class_names)
            return predicted_class, confidence, class_name
        
        except Exception as e:
            logger.error(f"Parkinson prediction error: {e}")
            return 0, 0.70, "Healthy"
    
    def _analyze_parkinson_patterns(self, eeg_data: np.ndarray) -> float:
        """Analyze EEG for Parkinson's patterns."""
        try:
            freqs, psd = welch(eeg_data, fs=160, nperseg=min(256, len(eeg_data)//4))
            
            delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
            theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
            beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
            
            total_power = delta_power + theta_power + alpha_power + beta_power
            
            if total_power > 0:
                beta_ratio = beta_power / total_power
                theta_ratio = theta_power / total_power
                
                # Tremor-related activity (4-6 Hz)
                tremor_power = np.sum(psd[(freqs >= 4) & (freqs <= 6)])
                tremor_ratio = tremor_power / total_power
                
                # Beta suppression
                beta_suppression = 1 - beta_ratio
                
                # Entropy
                psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                
                # Check for spikes
                diff = np.diff(eeg_data)
                std_amp = np.std(eeg_data)
                spikes = np.sum(np.abs(diff) > std_amp * 2)
                spike_ratio = spikes / len(diff) if len(diff) > 0 else 0
                
                # Score (very conservative)
                score = min(1.0, (
                    beta_suppression * 0.5 +
                    tremor_ratio * 0.3 +
                    theta_ratio * 0.2 +
                    (entropy / 20) * 0.1
                ))
                
                # Reduce if epilepsy patterns
                if spike_ratio > 0.01:
                    score *= 0.1
                
                # Boost for strong motor indicators
                if beta_ratio < 0.15 and tremor_ratio > 0.2:
                    score = min(1.0, score * 1.4)
                
                return score
            
            return 0.1
        
        except Exception:
            return 0.1


# Initialize predictors
epilepsy_predictor = EpilepsyPredictor()
alzheimer_predictor = AlzheimerPredictor()
sleep_disorder_predictor = SleepDisorderPredictor()
parkinson_predictor = ParkinsonPredictor()


def run_all_predictions(eeg_data: np.ndarray) -> Dict[str, Dict]:
    """
    Run all disease predictions and return results.
    
    Args:
        eeg_data: EEG signal data
    
    Returns:
        Dictionary of prediction results for each disease
    """
    results = {}
    
    try:
        # Get predictions
        ep_class, ep_conf, ep_name = epilepsy_predictor.predict(eeg_data)
        alz_class, alz_conf, alz_name = alzheimer_predictor.predict(eeg_data)
        sleep_class, sleep_conf, sleep_name = sleep_disorder_predictor.predict(eeg_data)
        park_class, park_conf, park_name = parkinson_predictor.predict(eeg_data)
        
        # Rank predictions (only consider positive detections)
        scores = {}
        if ep_class == 1:
            scores['epilepsy'] = ep_conf
        if alz_class == 1:
            scores['alzheimer'] = alz_conf
        if sleep_class == 1:
            scores['sleep_disorder'] = sleep_conf
        if park_class == 1:
            scores['parkinson'] = park_conf
        
        # Find highest scoring condition
        confidence_threshold = 0.6
        
        if scores:
            max_condition = max(scores, key=scores.get)
            max_score = scores[max_condition]
        else:
            max_condition = None
            max_score = 0
        
        # Show only highest scoring condition above threshold
        if max_condition and max_score > confidence_threshold:
            # Set only the detected condition as positive
            results['epilepsy'] = {
                'predicted_class': 1 if max_condition == 'epilepsy' else 0,
                'confidence': ep_conf if max_condition == 'epilepsy' else 1 - ep_conf,
                'class_name': ep_name if max_condition == 'epilepsy' else 'Normal'
            }
            results['alzheimer'] = {
                'predicted_class': 1 if max_condition == 'alzheimer' else 0,
                'confidence': alz_conf if max_condition == 'alzheimer' else 1 - alz_conf,
                'class_name': alz_name if max_condition == 'alzheimer' else 'Normal'
            }
            results['sleep_disorder'] = {
                'predicted_class': 1 if max_condition == 'sleep_disorder' else 0,
                'confidence': sleep_conf if max_condition == 'sleep_disorder' else 1 - sleep_conf,
                'class_name': sleep_name if max_condition == 'sleep_disorder' else 'Normal'
            }
            results['parkinson'] = {
                'predicted_class': 1 if max_condition == 'parkinson' else 0,
                'confidence': park_conf if max_condition == 'parkinson' else 1 - park_conf,
                'class_name': park_name if max_condition == 'parkinson' else 'Healthy'
            }
        else:
            # All normal
            results['epilepsy'] = {
                'predicted_class': 0,
                'confidence': 1 - ep_conf,
                'class_name': 'Normal'
            }
            results['alzheimer'] = {
                'predicted_class': 0,
                'confidence': 1 - alz_conf,
                'class_name': 'Normal'
            }
            results['sleep_disorder'] = {
                'predicted_class': 0,
                'confidence': 1 - sleep_conf,
                'class_name': 'Normal'
            }
            results['parkinson'] = {
                'predicted_class': 0,
                'confidence': 1 - park_conf,
                'class_name': 'Healthy'
            }
    
    except Exception as e:
        logger.error(f"Prediction pipeline error: {e}")
        # Return defaults
        results = {
            'epilepsy': {'predicted_class': 0, 'confidence': 0.5, 'class_name': 'Normal'},
            'alzheimer': {'predicted_class': 0, 'confidence': 0.5, 'class_name': 'Normal'},
            'sleep_disorder': {'predicted_class': 0, 'confidence': 0.5, 'class_name': 'Normal'},
            'parkinson': {'predicted_class': 0, 'confidence': 0.5, 'class_name': 'Healthy'}
        }
    
    return results


# ============================================================================
# FILE LOADING
# ============================================================================

def load_eeg_file(filepath: str) -> bool:
    """
    Load EEG file (EDF or FIF format).
    
    Args:
        filepath: Path to EEG file
    
    Returns:
        True if successful, False otherwise
    """
    # Ensure MNE is available for EEG file I/O
    if mne is None:
        logger.error("MNE not available for EEG loading")
        return False
    
    try:
        # Determine file type
        ext = filepath.lower().split('.')[-1]
        
<<<<<<< Updated upstream
        #File Loading (Format-Specific) & storing the loaded data 

=======
        # Read the file with the appropriate MNE reader (preload into memory)
>>>>>>> Stashed changes
        if ext == 'edf':
            state.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        elif ext in ['fif', 'gz']:
            state.raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        else:
            logger.error(f"Unsupported file format: {ext}")
            return False
        

        #extract sampling rate , channel names, total samples 
        # Update state
        state.fs = int(state.raw.info["sfreq"])  # Native sampling frequency
        state.n_times = state.raw.n_times         # Total number of samples
        state.ch_names = state.raw.ch_names       # Channel labels
        
        # Calculate initial offset
<<<<<<< Updated upstream
        state.initial_offset = state.fs * 10  # Skip first 10 seconds for artifact removal
=======
        state.initial_offset = state.fs * 10      # Skip first 10 seconds as warm-up
>>>>>>> Stashed changes
        state.current_index = min(state.initial_offset, state.n_times)
        
        state.loaded = True                        # Mark as ready
        
        # Log a concise summary for debugging/telemetry
        logger.info(f"Loaded EEG file: {filepath}")
        logger.info(f"Channels: {len(state.ch_names)}, FS: {state.fs} Hz, Duration: {state.n_times / state.fs:.2f}s")
        
        return True
    
    except Exception as e:
        # On I/O or parsing failures, log stack and signal failure to the caller
        logger.exception(f"Failed to load EEG file: {e}")
        return False


# ============================================================================
# FLASK BLUEPRINT
# ============================================================================

bp = Blueprint("eeg", __name__, template_folder="../templates")


@bp.route("/", methods=["GET"])
def eeg_home():
    """Render EEG viewer page."""
    return render_template("eeg.html")


@bp.route("/sampling", methods=["GET"])
def eeg_sampling_analysis():
    """Render EEG sampling analysis page."""
    return render_template("sampling_analysis.html")


def _get_channel_segment(channel_index: int, seconds: float = 5.0):
    """Helper: return a short recent segment for a channel as numpy array."""
    # Safety: ensure a file is loaded before slicing from raw stream
    # The global 'state' holds the active MNE Raw object and indices
    if state.raw is None:
        return None, 0
    # Sampling rate (Hz) pulled from the MNE info dict, fallback to state.fs
    fs = int(state.raw.info.get("sfreq", state.fs))
    # Compute window length in samples for the requested seconds (>=1 sample)
    win = max(1, int(seconds * fs))
    # End index is the current playhead or the end of the file, whichever is smaller
    end = min(state.current_index if state.current_index > 0 else state.n_times, state.n_times)
    # Start index backs off by 'win' samples but never before the recording start
    start = max(0, end - win)
    try:
        # Read the segment for the single channel into a numpy array
        picked = state.raw.get_data(picks=[channel_index], start=start, stop=end)
        # MNE returns shape (1, N); flatten to 1D
        seg = picked[0] if picked.ndim == 2 else picked
        return seg.astype(float), fs
    except Exception:
        return None, 0

#default route call when opening the page 
#show confusion matrix for different sampling rates as an example
@bp.route("/analyze-sampling", methods=["POST"])
def analyze_sampling():
    """Batch analyze multiple target sampling rates for a channel. Returns metrics per fs."""
    if state.raw is None:
        return jsonify({"success": False, "message": "No file loaded."}), 400
    try:
        # Parse JSON body: channels array is expected; use first channel
        data = request.get_json() or {}
        
        channels = data.get("channels", [])
        if not channels:
            return jsonify({"success": False, "message": "No channel provided."}), 400
        ch = int(channels[0])
        # Extract an ~8s recent window for analysis
        seg, fs = _get_channel_segment(ch, seconds=8.0)
        if seg is None or fs == 0:
            return jsonify({"success": False, "message": "Failed to extract signal segment."}), 500
        # Define candidate target sampling rates down to 10 Hz (inclusive)
        targets = [fs, max(10, fs//2), max(10, fs//4), max(10, fs//8)]
        results = {}
        # Iterate unique target Fs in descending order for display
        for tfs in sorted(set(int(x) for x in targets if x >= 10), reverse=True):
            # Apply aliasing decimation intentionally to visualize information loss
            dec = decimate_with_aliasing(seg, native_fs=fs, target_fs=tfs)
            # Produce deterministic demo metrics for visualization
            res_len = int(len(seg) * (tfs / fs)) if fs > 0 else 0
            ratio = (fs / tfs) if tfs > 0 else 0
            # Simple signal quality: SNR approximation and amplitude range
            snr = float(np.mean(seg**2) / (np.var(seg - np.mean(seg)) + 1e-8)) if len(seg) else 0.0
            results[str(tfs)] = {
                "sampling_ratio": float(ratio),
                "resampled_length": int(res_len),
                "metrics": {
                    # Synthetic classification metrics to illustrate trade-offs
                    "accuracy": max(0.0, min(1.0, 1.0 - (ratio-1)*0.1)),
                    "precision": max(0.0, min(1.0, 1.0 - (ratio-1)*0.12)),
                    "recall": max(0.0, min(1.0, 1.0 - (ratio-1)*0.08)),
                    "f1_score": max(0.0, min(1.0, 1.0 - (ratio-1)*0.1)),
                    "true_negative": int(90 * max(0.0, min(1.0, 1.0 - (ratio-1)*0.2))),
                    "false_positive": int(10 * max(0.0, min(1.0, (ratio-1)*0.2))),
                    "false_negative": int(10 * max(0.0, min(1.0, (ratio-1)*0.15))),
                    "true_positive": int(90 * max(0.0, min(1.0, 1.0 - (ratio-1)*0.15)))
                },
                "signal_quality": {
                    "snr": float(snr),
                    "variance": float(np.var(seg)),
                    "range": float(np.max(seg) - np.min(seg))
                }
            }
        return jsonify({"success": True, "results": results})
    except Exception as e:
        logger.exception("analyze-sampling failed")
        return jsonify({"success": False, "message": str(e)}), 500

#when upload a file and analyze sampling this route is called
#the actual sampling analysis
@bp.route("/get-sampling-signal", methods=["POST"])
def get_sampling_signal():
    """Return original and resampled signal/time arrays for plotting."""
    if state.raw is None:
        return jsonify({"success": False, "message": "No file loaded."}), 400
    try:
        # Read channel selection and desired target Fs from the client
        data = request.get_json() or {}
        channels = data.get("channels", [])
        target_fs = int(data.get("target_fs", state.fs))
        if not channels:
            return jsonify({"success": False, "message": "No channel provided."}), 400
        ch = int(channels[0])
        # Use a ~5s window for side-by-side time comparison
        seg, fs = _get_channel_segment(ch, seconds=5.0)
        if seg is None or fs == 0:
            return jsonify({"success": False, "message": "Failed to extract signal segment."}), 500
        # Build time axes for original and resampled signals over equal duration
        duration = len(seg) / fs
        t_orig = np.linspace(0, duration, len(seg)).tolist()
        # Apply aliasing decimation to visualize how samples shrink with lower Fs
        res = decimate_with_aliasing(seg, native_fs=fs, target_fs=max(1, target_fs))
        t_res = np.linspace(0, duration, len(res)).tolist()
        return jsonify({
            "success": True,
            "original_signal": {"time": t_orig, "data": seg.tolist(), "fs": int(fs)},
            "resampled_signal": {"time": t_res, "data": res.tolist(), "fs": int(target_fs)}
        })
    except Exception as e:
        logger.exception("get-sampling-signal failed")
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/analyze-single-sampling", methods=["POST"])
def analyze_single_sampling():
    """Analyze a single target sampling rate and return metrics for the UI."""
    if state.raw is None:
        return jsonify({"success": False, "message": "No file loaded."}), 400
    try:
        # Expect one channel and a target Fs; return metrics for that single choice
        data = request.get_json() or {}
        channels = data.get("channels", [])
        target_fs = int(data.get("target_fs", state.fs))
        if not channels:
            return jsonify({"success": False, "message": "No channel provided."}), 400
        ch = int(channels[0])
        # Slightly longer window (~8s) for robust metric estimates
        seg, fs = _get_channel_segment(ch, seconds=8.0)
        if seg is None or fs == 0:
            return jsonify({"success": False, "message": "Failed to extract signal segment."}), 500
        # Downsample with aliasing and compute sampling ratio vs native
        res = decimate_with_aliasing(seg, native_fs=fs, target_fs=max(1, target_fs))
        ratio = (fs / target_fs) if target_fs > 0 else 0
        result = {
            "sampling_ratio": float(ratio),
            "resampled_length": int(len(res)),
            "metrics": {
                "accuracy": max(0.0, min(1.0, 1.0 - (ratio-1)*0.1)),
                "precision": max(0.0, min(1.0, 1.0 - (ratio-1)*0.12)),
                "recall": max(0.0, min(1.0, 1.0 - (ratio-1)*0.08)),
                "f1_score": max(0.0, min(1.0, 1.0 - (ratio-1)*0.1)),
                "true_negative": int(90 * max(0.0, min(1.0, 1.0 - (ratio-1)*0.2))),
                "false_positive": int(10 * max(0.0, min(1.0, (ratio-1)*0.2))),
                "false_negative": int(10 * max(0.0, min(1.0, (ratio-1)*0.15))),
                "true_positive": int(90 * max(0.0, min(1.0, 1.0 - (ratio-1)*0.15)))
            },
            "signal_quality": {
                # Signal quality snapshot from the original segment
                "snr": float(np.mean(seg**2) / (np.var(seg - np.mean(seg)) + 1e-8)) if len(seg) else 0.0,
                "variance": float(np.var(seg)),
                "range": float(np.max(seg) - np.min(seg))
            }
        }
        return jsonify({"success": True, "result": result})
    except Exception as e:
        logger.exception("analyze-single-sampling failed")
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/upload", methods=["POST"])
def upload_file():
    """Upload EEG file."""
    # Validate multi-part form has a 'file' field
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    # Extract the werkzeug FileStorage object
    file = request.files['file']
    # Ensure a filename was provided (not an empty selection)
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    # Prepare disk destination for persistent storage
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)  # Create folder if missing
    filepath = os.path.join(upload_dir, file.filename)
    
    try:
        # Save the uploaded file bytes to disk
        file.save(filepath)
        
        # Attempt to parse and load EEG using MNE (supports EDF/FIF)
        if load_eeg_file(filepath):
            # Build channel info mapping index->name for the frontend
            ch_info = {i: name for i, name in enumerate(state.ch_names)}
            
            # Return success response with discovered metadata
            return jsonify({
                "success": True,
                "message": f"File {file.filename} loaded successfully.",
                "channels": ch_info,
                "fs": state.fs
            })
        else:
            # Loading failed (unsupported format or parsing error)
            return jsonify({
                "success": False,
                "message": "Failed to load EEG file. Check format (EDF or FIF)."
            }), 500
    
    except Exception as e:
        # Log and surface a structured error if saving/loading throws
        logger.exception(f"Upload error: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


@bp.route("/update", methods=["POST"])
def update():
    """Stream EEG data update."""
    #Validation
    if state.raw is None:
        return jsonify({
            "n_samples": 0,
            "signals": {},
           
            "message": "No file loaded."
        })
    
    try:
        data = request.get_json()
        
        # Get requested channel indices from frontend
        #and Validate they exist in the loaded file
        channels = validate_channels(
            data.get("channels", []),
            max_channels=len(state.ch_names)
        )
        
        if not channels:
            return jsonify({
                "n_samples": 0,
                "signals": {},
                
            })
        
        mode = data.get("mode", "time")   
        width = float(data.get("width", 5))   # Window size in seconds
        downsample_factor = data.get("downsample_factor", 1)
        
        # Calculate chunk size
        chunk_samples = BASE_CHUNK_SAMPLES  # Fixed: 16 samples per call
        
        # Read Data Chunk from File
        start = state.current_index # Current reading position 
        stop = start + chunk_samples   # Read next 16 samples
        
        # Handle  End-of-File   wrap-around
        if stop > state.n_times:
            stop = state.n_times
            chunk_samples = stop - start
            state.current_index = state.initial_offset
        else:
            state.current_index = stop
        
        if chunk_samples <= 0:
            state.current_index = state.initial_offset
            return jsonify({
                "n_samples": 0,
                "signals": {},
               
            })
        
        # Get data for selected channels
        picked = state.raw.get_data(picks=channels, start=start, stop=stop)
        
        # Apply downsampling if requested
        if downsample_factor > 1:
            picked_downsampled = [] #Create empty list to store downsampled channels
            #Loop through each channel
            for ch_idx in range(picked.shape[0]): #picked.shape[0] = number of channels
                downsampled = decimate_with_aliasing(
                    picked[ch_idx],
                    native_fs=state.fs,
                    target_fs=max(1, state.fs // downsample_factor)
                )
                picked_downsampled.append(downsampled)
            #Convert list back to numpy array    
            picked = np.array(picked_downsampled)
        
        
        # Build Response
        signals = {
            # Convert numpy array to JSON-friendly list
            str(ch): picked[i].tolist() 
            for i, ch in enumerate(channels)
        }
        
        response = {
            "n_samples": picked.shape[1] if picked.ndim == 2 else len(picked),
            "signals": signals,
            
        }
        
        # Server-side XOR computation

        if mode == "xor" and len(channels) == 1 and picked.shape[0] == 1:
             # Get the selected channel number (only one channel is supported for XOR)
            ch = int(channels[0])
            new_samples = signals[str(ch)]
            
            # Initialize buffers if they don't exist yet
            #xor_buffers[ch] → holds the rolling buffer of recent samples
            # xor_prev_windows[ch] → stores the last full window for comparison
            if ch not in state.xor_buffers:
                state.xor_buffers[ch] = []
            if ch not in state.xor_prev_windows:
                state.xor_prev_windows[ch] = []
            
           # Define window size in samples (width in seconds × sampling frequency)
            chunk_size = max(1, int(width * state.fs))
            
            buf = state.xor_buffers[ch]
            buf.extend(new_samples)

             # Keep only the most recent samples 
             # so the buffer length equals chunk_size
            if len(buf) > chunk_size:
                del buf[0:len(buf) - chunk_size]
            
            # Calculate XOR
            xor_result = calculate_xor_difference_eeg(
                buf,
                state.xor_prev_windows.get(ch, []),
                chunk_size
            )
            
            # Update previous window 
            # (used in the next iteration for comparison)
            if len(buf) == chunk_size:
                state.xor_prev_windows[ch] = buf[-chunk_size:].copy()
            
            response["xor"] = xor_result
        
        return jsonify(response)
    except Exception as e:
        logger.exception("Update failed")
        return jsonify({
            "n_samples": 0,
            "signals": {},
           
            "error": str(e)
        }), 500

@bp.route("/predict", methods=["POST"])
def predict_diseases():
    """Run disease predictions on current EEG data."""
    if state.raw is None:
        return jsonify({
            "success": False,
            "message": "No file loaded."
        }), 400
    
    try:
        data = request.get_json() or {}
        channels = validate_channels(
            data.get("channels", []),
            max_channels=len(state.ch_names)
        )
        target_fs = int(data.get("target_fs", state.fs))
        
        if not channels:
            return jsonify({
                "success": False,
                "message": "No channels selected."
            }), 400
        
        # Get current data
        start = state.current_index
        stop = start + BASE_CHUNK_SAMPLES
        
        if stop > state.n_times:
            stop = state.n_times
        
        if stop <= start:
            return jsonify({
                "success": False,
                "message": "No data available."
            }), 400
        
        # Get data for first selected channel
        picked = state.raw.get_data(picks=[channels[0]], start=start, stop=stop)
        
        if picked.shape[0] == 0 or picked.shape[1] == 0:
            return jsonify({
                "success": False,
                "message": "No valid data."
            }), 400
        
        eeg_data_for_prediction = picked[0]
        
        # Resample to requested target sampling rate if different from native
        if target_fs != state.fs:
            eeg_data_for_prediction = decimate_with_aliasing(
                eeg_data_for_prediction,
                native_fs=state.fs,
                target_fs=max(1, int(target_fs)),
                pos_native=start,
                phase_state=None
            )
            logger.info(f"Resampled for prediction: {state.fs} -> {target_fs} Hz")
        
        # Run predictions
        prediction_results = run_all_predictions(eeg_data_for_prediction)
        
        return jsonify({
            "success": True,
            "predictions": prediction_results,
            "channel_used": channels[0],
            "data_length": len(eeg_data_for_prediction),
            "target_fs": int(target_fs)
        })
    
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500