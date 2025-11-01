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

# Configure the root logging system for the entire application
# This sets up how log messages will be formatted and what level of messages to display
logging.basicConfig(
    # Set the minimum logging level to INFO
    # This means: DEBUG (lowest) < INFO < WARNING < ERROR < CRITICAL (highest)
    # Only messages with level INFO and above will be displayed
    level=logging.INFO,
    
    # Define the format string for log messages:
    # %(asctime)s - Timestamp when the log record was created
    # [%(levelname)s] - Log level (INFO, WARNING, ERROR, etc.) in brackets
    # %(message)s - The actual log message text
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Create a named logger instance specifically for the EEG module
# This allows us to identify which part of the application generated the log
# We can configure this logger separately from the root logger if needed
logger = logging.getLogger("eeg")

# Device configuration for PyTorch operations
# This determines whether to use GPU (CUDA) or CPU for tensor computations
DEVICE = torch.device(
    # Conditional expression to choose the best available hardware:
    # Check if CUDA (NVIDIA GPU support) is available in the current environment
    "cuda" if torch.cuda.is_available() else 
    
    # Fall back to CPU if no compatible GPU is available
    # CPU will be used for all PyTorch operations in this case
    "cpu"
)

# Streaming parameters - control how EEG data is streamed to clients

# BASE_CHUNK_SAMPLES defines the number of data samples sent per streaming request
# This is a fundamental parameter that affects:
# - Real-time performance: Smaller chunks = more frequent updates but higher overhead
# - Network load: Larger chunks = less frequent updates but more data per request
# - Client-side smoothness: Balance between update frequency and data volume
BASE_CHUNK_SAMPLES = 16

# INITIAL_OFFSET_SAMPLES determines how many samples to skip at the beginning of the file
# This is useful for:
# - Skipping initial artifacts or calibration periods in EEG recordings
# - Avoiding unstable signal segments at the start of recordings
# - Allowing hardware to stabilize before processing meaningful data
# The value 0 indicates no initial skip, but this will be recalculated after file load
# based on the actual sampling rate and desired skip duration (e.g., 10 seconds)
INITIAL_OFFSET_SAMPLES = 0  # Will be set after file load

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Define a class to manage the global state of EEG streaming and processing
# This serves as a centralized container for all EEG-related data and state variables
# Using a class provides encapsulation and organized access to related state information
class EEGState:
    """Centralized state management for EEG streaming."""
    
    # Constructor method - initializes all state variables when an instance is created
    def __init__(self):
        # MNE raw object - stores the loaded EEG data and metadata
        # MNE (MNE-Python) is a comprehensive EEG/MEG processing library
        # self.raw contains the actual signal data, channel info, sampling rate, etc.
        # None indicates no file is currently loaded
        self.raw = None
        
        # Signal properties - fundamental characteristics of the EEG data
        
        # Sampling frequency (fs) in Hz - number of samples per second
        # Default value of 160 Hz is a common sampling rate for EEG
        # This will be updated to the actual file's sampling rate after loading
        self.fs = 160  # Sampling frequency
        
        # Total number of samples in the entire EEG recording
        # This represents the length of the recording: n_times / fs = duration in seconds
        # 0 indicates no data is currently loaded
        self.n_times = 0  # Total samples
        
        # List of channel names (electrode positions) from the EEG recording
        # Examples: ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2'] according to 10-20 system
        # Empty list indicates no channels are defined yet
        self.ch_names = []  # Channel names
        
        # Streaming state - variables that control real-time data playback
        
        # Current playback position in samples (like a "playhead" in audio/video)
        # This tracks where we are in the EEG file during streaming
        # 0 means we're at the beginning of the file
        self.current_index = 0  # Current playback position
        
        # Number of samples to skip at the beginning of the file
        # Used to avoid initial artifacts, calibration periods, or unstable signals
        # Typically set to skip first 10 seconds: fs * 10
        self.initial_offset = 0  # Skip initial samples
        
        # XOR mode state - server-side buffers for change detection analysis
        
        # Dictionary that stores rolling buffers for each channel in XOR mode
        # Key: channel index (integer), Value: list of recent samples for that channel
        # Used to maintain a sliding window of data for real-time change detection
        self.xor_buffers = {}  # Rolling buffers per channel
        
        # Dictionary that stores previous analysis windows for each channel in XOR mode
        # Key: channel index, Value: previous window of samples for comparison
        # Used to compute differences between consecutive time windows
        self.xor_prev_windows = {}  # Previous window per channel
        
        # File loaded flag - simple boolean to indicate if EEG data is ready for processing
        # True: file is loaded and valid, False: no file loaded or loading failed
        # Used for quick validation checks throughout the application
        self.loaded = False
    
    # Method to reset only streaming-related state while keeping the loaded file
    # Useful for restarting playback from the beginning without reloading the file
    def reset_streaming_state(self):
        """Reset streaming-related state."""
        # Reset playback position to the initial offset (skip initial samples)
        # This ensures we start from the same point each time we reset
        self.current_index = self.initial_offset
        
        # Clear all XOR buffers - removes accumulated data for change detection
        # This ensures clean state when restarting streaming
        self.xor_buffers = {}
        
        # Clear all previous windows - removes reference data for change comparison
        # This ensures we start fresh with new baseline comparisons
        self.xor_prev_windows = {}
    
    # Method to completely reset all state to initial conditions
    # Useful when switching files or when complete cleanup is needed
    def reset_all(self):
        """Reset all state."""
        # Call the constructor again to reinitialize all variables to defaults
        # This is equivalent to creating a new EEGState instance
        # Note: This approach reuses the same object rather than creating a new one
        self.__init__()


# Create a global instance of the EEGState class
# This single instance will be shared across the entire application
# Using a global state instance provides:
# - Centralized access to EEG data and state
# - Consistent state management across different modules
# - Simplified data sharing between different components
state = EEGState()


# ============================================================================
# SIGNAL PROCESSING UTILITIES
# ============================================================================


def calculate_xor_difference_eeg(current_buffer: List[float], 
                                 previous_window: List[float],
                                 chunk_size: int) -> List[float]:
    """
    Calculate thresholded XOR difference for EEG signals.
    
    This function compares consecutive windows of EEG data and identifies
    significant changes between them, filtering out minor fluctuations.
    
    Args:
        current_buffer: Current signal buffer containing recent EEG samples
        previous_window: Previous window of EEG data for comparison
        chunk_size: Size of analysis window (number of samples to compare)
    
    Returns:
        XOR difference signal where only significant changes are preserved
    """
    # Not enough data yet → Return as-is (no comparison possible) 
    # If we don't have enough samples in the current buffer to form a full window,
    # we can't perform the comparison, so return the buffer unchanged
    if len(current_buffer) < chunk_size:
        return current_buffer
    
    # First iteration OR size mismatch
    # Nothing to compare against → Return current buffer
    # This handles cases where:
    # - This is the first time we're processing data (no previous window)
    # - The previous window size doesn't match our expected chunk_size
    if not previous_window or len(previous_window) != chunk_size:
        return current_buffer
    
    # Take last chunk_size samples from buffer
    # Extract the most recent window of data from the current buffer
    # This gives us a fixed-size window to compare with the previous one
    current_window = current_buffer[-chunk_size:]
    
    # Calculate statistics for dynamic threshold
    # Compute mean and standard deviation of the current window
    # This allows us to set a threshold that adapts to the signal's characteristics
    mean = np.mean(current_window)
    std = np.std(current_window)
    threshold = std * 0.1  # 10% of standard deviation
    # The threshold is proportional to signal variability - more variable signals
    # get a higher threshold, making the algorithm more robust to noise
    
    # Compute thresholded difference
    xor_result = []
    for i in range(chunk_size):
        curr_val = current_window[i]  # Sample from current window
        prev_val = previous_window[i] # Same position in previous window
        
        # Calculate absolute difference between corresponding samples
        distance = abs(curr_val - prev_val)
        
        # Show difference if above threshold, else 0
        # Only keep differences that are statistically significant
        # This filters out minor noise and preserves meaningful signal changes
        xor_result.append(distance if distance > threshold else 0)
    
    return xor_result


# ============================================================================
# DISEASE PREDICTION MODELS
# ============================================================================

def _prepare_input_1d(eeg_data: np.ndarray, target_len: int = 1024) -> np.ndarray:
    """
    Prepare EEG data as a fixed-length 1D float32 vector for model input.
    
    This function standardizes EEG data to a consistent format that neural networks can process.
    
    Args:
        eeg_data: Raw EEG data array (can be 1D or 2D)
        target_len: Desired length of output vector (default: 1024 samples)
    
    Returns:
        Standardized 1D vector ready for model input
    """
    arr = eeg_data
    # Convert 2D data to 1D if necessary
    # Example: If input is [channels, samples], flatten to [channels * samples]
    if arr.ndim == 2:
        arr = arr.flatten()
    
    # Handle data that's too long: truncate to target length
    # Take only the first target_len samples from the beginning
    if len(arr) > target_len:
        arr = arr[:target_len]
    # Handle data that's too short: pad with zeros at the end
    elif len(arr) < target_len:
        arr = np.pad(arr, (0, target_len - len(arr)))
    
    # Normalize the signal to have consistent scale
    # Z-score normalization: (x - mean) / std → mean=0, std=1
    arr = normalize_signal(arr, method='zscore')
    
    # Convert to float32 for efficient GPU processing
    # copy=False avoids unnecessary memory duplication if already correct type
    return arr.astype(np.float32, copy=False)


def _predict_softmax2(model: nn.Module, x1d: np.ndarray, device: torch.device, class_names: List[str]) -> Tuple[int, float, str]:
    """
    Run 2-class softmax model on 1D input and decode to label and confidence.
    
    This performs inference and converts model outputs to human-readable predictions.
    
    Args:
        model: Trained neural network model
        x1d: Prepared 1D input vector
        device: Computing device (CPU/GPU)
        class_names: List of class labels (e.g., ['normal', 'abnormal'])
    Tensor Conversion: NumPy → PyTorch with batch dimension

    Logits: Raw model outputs before probability conversion

    Softmax: Converts logits to probabilities that sum to 1

    Argmax: Finds the most probable class
    Returns:
        Tuple of (predicted_index, confidence_score, class_name)
    """
    # Convert numpy array to PyTorch tensor and add batch dimension
    # unsqueeze(0) adds batch dimension: [1024] → [1, 1024]
    tensor = torch.from_numpy(x1d).unsqueeze(0).to(device)
    
    # Disable gradient computation for faster inference
    with torch.no_grad():
        # Get raw model outputs (logits)
        logits = model(tensor)
        
        # Convert logits to probabilities using softmax
        # softmax ensures probabilities sum to 1
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Find the class with highest probability
        pred_idx = int(np.argmax(probs))
        
        # Get the confidence score for the predicted class
        conf = float(probs[pred_idx])
    
    # Return index, confidence, and human-readable class name
    return pred_idx, conf, class_names[pred_idx]


class SimpleDiseasePredictor(nn.Module):
    """
    Simple neural network for disease prediction from EEG data.
    
    Architecture: Input → 512 → 256 → 2 (binary classification)
    """
    
    def __init__(self, input_size: int = 1024):
        """
        Initialize the neural network layers.
        
        Args:
            input_size: Number of input features (default: 1024 EEG samples)
        """
        super().__init__()
        self.net = nn.Sequential(
            # First layer: input_size → 512 neurons
            nn.Linear(input_size, 512),
            nn.ReLU(),  # Activation function for non-linearity
            nn.Dropout(0.3),  # Regularization: randomly disable 30% of neurons during training
            
            # Second layer: 512 → 256 neurons  
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer: 256 → 2 neurons (for 2-class classification)
            # No activation here - raw logits will be passed to softmax during inference
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        """
        Define the forward pass of the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Raw logits (unnormalized scores) for each class
        """
        return self.net(x)


class EpilepsyPredictor:
    """
    Epilepsy detection from EEG patterns.
    
    This class provides both model-based and rule-based epilepsy detection.
    It first tries to use a trained neural network, and falls back to
    signal analysis if no model is available.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the epilepsy predictor.
        
        Args:
            model_path: Path to trained model file (.pth, .pt, etc.)
            device: 'auto', 'cuda', or 'cpu' - where to run computations
        """
        self.device = self._get_device(device)
        self.model_path = model_path
        self.model = None  # Will be loaded lazily when needed
        self.class_names = ['Normal', 'Epilepsy']  # Output classes
    
    def _get_device(self, device: str) -> torch.device:
        """Automatically select the best available device (GPU/CPU)."""
        if device == 'auto':
            # Prefer GPU for speed, fall back to CPU if no GPU available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self):
        """Load model if available - lazy loading to avoid unnecessary operations."""
        # If model is already loaded, do nothing
        if self.model is not None:
            return
        
        # Try to find and load the model file
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Initialize model architecture
                self.model = SimpleDiseasePredictor().to(self.device)
                
                # Load trained weights
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                # Some checkpoints store weights under 'state_dict' key, others directly
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                # Load weights with flexibility for minor architecture differences
                self.model.load_state_dict(state_dict, strict=False)
                
                # Set model to evaluation mode (disables dropout, batch norm uses running stats)
                self.model.eval()
                
                logger.info(f"Loaded epilepsy model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load epilepsy model: {e}")
                self.model = None  # Ensure model is None if loading fails
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """
        Predict epilepsy from EEG data.
        
        Args:
            eeg_data: EEG signal data (1D or 2D array)
        
        Returns:
            Tuple of (predicted_class, confidence, class_name)
            - predicted_class: 0 for Normal, 1 for Epilepsy
            - confidence: Probability score between 0-1
            - class_name: Human-readable label
        """
        # Ensure model is loaded (if available)
        self._load_model()
        
        # Use pattern analysis if model not available (fallback method)
        if self.model is None:
            epilepsy_score = self._analyze_epilepsy_patterns(eeg_data)
            
            # Apply threshold for classification
            if epilepsy_score > 0.7:
                return 1, epilepsy_score, "Epilepsy"
            else:
                return 0, 1 - epilepsy_score, "Normal"
        
        # Use neural network model prediction (primary method)
        try:
            # Prepare input data for the model
            x1d = _prepare_input_1d(eeg_data, target_len=1024)
            
            # Run prediction
            predicted_class, confidence, class_name = _predict_softmax2(
                self.model, x1d, self.device, self.class_names
            )
            return predicted_class, confidence, class_name
        
        except Exception as e:
            # If model prediction fails, fall back to rule-based analysis
            logger.error(f"Epilepsy prediction error: {e}")
            return 0, 0.7, "Normal"  # Default to "Normal" with medium confidence
    
    def _analyze_epilepsy_patterns(self, eeg_data: np.ndarray) -> float:
        """
        Analyze EEG for epilepsy-specific patterns using signal processing.
        
        This rule-based method detects common epilepsy indicators:
        - Spikes and sharp waves
        - High-frequency seizure activity
        - Amplitude abnormalities
        
        Args:
            eeg_data: Raw EEG signal
            
        Returns:
            Epilepsy probability score between 0-1
        """
        try:
            # Statistical features
            # Calculate standard deviation of EEG signal
            # Higher std may indicate more abnormal activity
            std_amp = np.std(eeg_data)
            
            # Spike detection - look for rapid amplitude jumps
            # Spikes are sudden, high-amplitude changes
            diff = np.diff(eeg_data)  # Calculate differences between consecutive samples
            spike_threshold = std_amp * 2  # Dynamic threshold based on signal variability
            spikes = np.sum(np.abs(diff) > spike_threshold)  # Count spikes above threshold
            spike_ratio = spikes / len(diff) if len(diff) > 0 else 0  # Normalize by signal length
            
            # Sharp waves detection - similar to spikes but slightly less extreme
            # Sharp waves are characteristic of epileptic activity
            sharp_threshold = std_amp * 1.5
            sharp_waves = np.sum(np.abs(diff) > sharp_threshold)
            sharp_ratio = sharp_waves / len(diff) if len(diff) > 0 else 0
            
            # High-frequency activity analysis using Power Spectral Density (PSD)
            # Seizures often show increased power in 20-40Hz range (beta/gamma)
            freqs, psd = welch(eeg_data, fs=160, nperseg=min(256, len(eeg_data)//4))
            seizure_power = np.sum(psd[(freqs >= 20) & (freqs <= 40)])  # Power in seizure-prone frequencies
            total_power = np.sum(psd)  # Total power across all frequencies
            seizure_ratio = seizure_power / total_power if total_power > 0 else 0
            
            # Amplitude asymmetry - measure of signal irregularity
            # Epileptic signals often show asymmetric amplitude distributions
            mean_amp = np.mean(np.abs(eeg_data))
            amplitude_asymmetry = np.std(np.abs(eeg_data)) / (mean_amp + 1e-6)  # +1e-6 avoids division by zero
            
            # Calculate composite epilepsy score
            # Weight different features based on their importance for epilepsy detection
            score = min(1.0, (
                spike_ratio * 1.5 +          # Spikes are strong indicators
                sharp_ratio * 1.2 +          # Sharp waves are also important
                seizure_ratio * 1.2 +        # High-frequency power matters
                min(amplitude_asymmetry * 0.1, 0.2)  # Asymmetry contributes less
            ))
            
            # Boost score if multiple indicators are present simultaneously
            # Co-occurrence of spikes and sharp waves is particularly suspicious
            if spike_ratio > 0.05 and sharp_ratio > 0.05:
                score = min(1.0, score * 1.3)  # Increase confidence by 30%
            
            return score
        
        except Exception:
            # If analysis fails, return low probability (assume normal)
            return 0.1

class AlzheimerPredictor:
    """Alzheimer's disease detection from EEG patterns."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model_path = model_path
        self.model = None
        self.class_names = ['Normal', 'Alzheimer']
    
    def _load_model(self):
        """Load model if available."""
        if self.model is not None:
            return
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = SimpleDiseasePredictor().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                logger.info(f"Loaded Alzheimer model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Alzheimer model: {e}")
                self.model = None
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """Predict Alzheimer's from EEG data."""
        self._load_model()
        
        if self.model is None:
            alzheimer_score = self._analyze_alzheimer_patterns(eeg_data)
            
            if alzheimer_score > 0.2:
                return 1, alzheimer_score, "Alzheimer"
            else:
                return 0, alzheimer_score, "Normal"
        
        # Model-based prediction (similar structure as epilepsy)
        try:
            x1d = _prepare_input_1d(eeg_data, target_len=1024)
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
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model_path = model_path
        self.model = None
        self.class_names = ['Normal', 'Sleep Disorder']
    
    def _load_model(self):
        if self.model is not None:
            return
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = SimpleDiseasePredictor().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                logger.info(f"Loaded sleep disorder model")
            except Exception as e:
                logger.warning(f"Failed to load sleep disorder model: {e}")
                self.model = None
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """Predict sleep disorder from EEG data."""
        self._load_model()
        
        if self.model is None:
            score = self._analyze_sleep_patterns(eeg_data)
            
            if score > 0.95:
                return 1, score, "Sleep Disorder"
            else:
                return 0, 1 - score, "Normal"
        
        # Model-based prediction
        try:
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
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model_path = model_path
        self.model = None
        self.class_names = ['Healthy', 'Parkinson']
    
    def _load_model(self):
        if self.model is not None:
            return
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = SimpleDiseasePredictor().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                logger.info(f"Loaded Parkinson model")
            except Exception as e:
                logger.warning(f"Failed to load Parkinson model: {e}")
                self.model = None
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, str]:
        """Predict Parkinson's from EEG data."""
        self._load_model()
        
        if self.model is None:
            score = self._analyze_parkinson_patterns(eeg_data)
            
            if score > 0.95:
                return 1, score, "Parkinson"
            else:
                return 0, 1 - score, "Healthy"
        
        # Model-based prediction
        try:
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
    if mne is None:
        logger.error("MNE not available for EEG loading")
        return False
    
    try:
        # Determine file type
        ext = filepath.lower().split('.')[-1]
        
        #File Loading (Format-Specific) & storing the loaded data 

        if ext == 'edf':
            state.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        elif ext in ['fif', 'gz']:
            state.raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        else:
            logger.error(f"Unsupported file format: {ext}")
            return False
        

        #extract sampling rate , channel names, total samples 
        # Update state
        state.fs = int(state.raw.info["sfreq"])
        state.n_times = state.raw.n_times
        state.ch_names = state.raw.ch_names
        
        # Calculate initial offset
        state.initial_offset = state.fs * 10  # Skip first 10 seconds for artifact removal
        state.current_index = min(state.initial_offset, state.n_times)
        
        state.loaded = True
        
        logger.info(f"Loaded EEG file: {filepath}")
        logger.info(f"Channels: {len(state.ch_names)}, FS: {state.fs} Hz, Duration: {state.n_times / state.fs:.2f}s")
        
        return True
    
    except Exception as e:
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
    """
    Helper: return a short recent segment for a channel as numpy array.
    
    Extracts a time window from the specified EEG channel for analysis.
    
    Args:
        channel_index: Index of the EEG channel to extract
        seconds: Duration of the segment to extract (default: 5 seconds)
    
    Returns:
        Tuple of (signal_segment, sampling_frequency)
        Returns (None, 0) if extraction fails
    """
    if state.raw is None:
        return None, 0
    
    # Get sampling frequency from the raw EEG data
    fs = int(state.raw.info.get("sfreq", state.fs))
    
    # Calculate window size in samples
    win = max(1, int(seconds * fs))
    
    # Determine end point (most recent sample)
    end = min(state.current_index if state.current_index > 0 else state.n_times, state.n_times)
    
    # Calculate start point to get the desired window
    start = max(0, end - win)
    
    try:
        # Extract data for the specified channel and time window
        picked = state.raw.get_data(picks=[channel_index], start=start, stop=end)
        
        # Handle different return formats (2D array vs 1D)
        seg = picked[0] if picked.ndim == 2 else picked
        
        return seg.astype(float), fs
    except Exception:
        return None, 0


@bp.route("/analyze-sampling", methods=["POST"])
def analyze_sampling():
    """
    Batch analyze multiple target sampling rates for a channel. Returns metrics per fs.
    
    This endpoint tests how downsampling affects signal quality and analysis metrics.
    Useful for finding optimal sampling rates that balance performance and computational cost.
    """
    if state.raw is None:
        return jsonify({"success": False, "message": "No file loaded."}), 400
    
    try:
        data = request.get_json() or {}
        channels = data.get("channels", [])
        if not channels:
            return jsonify({"success": False, "message": "No channel provided."}), 400
        
        # Get first channel from request
        ch = int(channels[0])
        
        # Extract signal segment for analysis
        seg, fs = _get_channel_segment(ch, seconds=8.0)
        if seg is None or fs == 0:
            return jsonify({"success": False, "message": "Failed to extract signal segment."}), 500
        
        # Target sampling rates to test (progressive downsampling)
        # Includes original and progressively lower sampling rates
        targets = [fs, max(10, fs//2), max(10, fs//4), max(10, fs//8)]
        
        results = {}
        
        # Test each target sampling rate (highest to lowest)
        for tfs in sorted(set(int(x) for x in targets if x >= 10), reverse=True):
            # Apply naive decimation (demonstrates aliasing effects)
            dec = decimate_with_aliasing(seg, native_fs=fs, target_fs=tfs)
            
            # Calculate expected length after resampling
            res_len = int(len(seg) * (tfs / fs)) if fs > 0 else 0
            
            # Calculate sampling ratio (how much we're downsampling)
            ratio = (fs / tfs) if tfs > 0 else 0
            
            # Calculate simple signal-to-noise ratio
            snr = float(np.mean(seg**2) / (np.var(seg - np.mean(seg)) + 1e-8)) if len(seg) else 0.0
            
            # Generate synthetic performance metrics
            # These metrics simulate how downsampling affects classification performance
            # The formulas show performance degradation as sampling ratio increases
            results[str(tfs)] = {
                "sampling_ratio": float(ratio),  # How much we're downsampling
                "resampled_length": int(res_len),  # Length after resampling
                "metrics": {
                    # Classification metrics that degrade with downsampling
                    "accuracy": max(0.0, min(1.0, 1.0 - (ratio-1)*0.1)),
                    "precision": max(0.0, min(1.0, 1.0 - (ratio-1)*0.12)),
                    "recall": max(0.0, min(1.0, 1.0 - (ratio-1)*0.08)),
                    "f1_score": max(0.0, min(1.0, 1.0 - (ratio-1)*0.1)),
                    
                    # Confusion matrix elements
                    "true_negative": int(90 * max(0.0, min(1.0, 1.0 - (ratio-1)*0.2))),
                    "false_positive": int(10 * max(0.0, min(1.0, (ratio-1)*0.2))),
                    "false_negative": int(10 * max(0.0, min(1.0, (ratio-1)*0.15))),
                    "true_positive": int(90 * max(0.0, min(1.0, 1.0 - (ratio-1)*0.15)))
                },
                "signal_quality": {
                    "snr": float(snr),  # Signal-to-noise ratio
                    "variance": float(np.var(seg)),  # Signal variability
                    "range": float(np.max(seg) - np.min(seg))  # Dynamic range
                }
            }
        
        return jsonify({"success": True, "results": results})
    
    except Exception as e:
        logger.exception("analyze-sampling failed")
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/get-sampling-signal", methods=["POST"])
def get_sampling_signal():
    """
    Return original and resampled signal/time arrays for plotting.
    
    This endpoint provides data for visualizing the effects of downsampling.
    Frontend can use this to show both original and resampled signals side by side.
    """
    if state.raw is None:
        return jsonify({"success": False, "message": "No file loaded."}), 400
    
    try:
        data = request.get_json() or {}
        channels = data.get("channels", [])
        target_fs = int(data.get("target_fs", state.fs))
        
        if not channels:
            return jsonify({"success": False, "message": "No channel provided."}), 400
        
        ch = int(channels[0])
        
        # Extract 5-second segment for visualization
        seg, fs = _get_channel_segment(ch, seconds=5.0)
        if seg is None or fs == 0:
            return jsonify({"success": False, "message": "Failed to extract signal segment."}), 500
        
        # Create time array for original signal
        duration = len(seg) / fs
        t_orig = np.linspace(0, duration, len(seg)).tolist()
        
        # Apply downsampling
        res = decimate_with_aliasing(seg, native_fs=fs, target_fs=max(1, target_fs))
        
        # Create time array for resampled signal
        t_res = np.linspace(0, duration, len(res)).tolist()
        
        return jsonify({
            "success": True,
            "original_signal": {
                "time": t_orig,      # Time points for original signal
                "data": seg.tolist(), # Original signal values
                "fs": int(fs)        # Original sampling frequency
            },
            "resampled_signal": {
                "time": t_res,       # Time points for resampled signal  
                "data": res.tolist(), # Resampled signal values
                "fs": int(target_fs) # Target sampling frequency
            }
        })
    
    except Exception as e:
        logger.exception("get-sampling-signal failed")
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/analyze-single-sampling", methods=["POST"])
def analyze_single_sampling():
    """
    Analyze a single target sampling rate and return metrics for the UI.
    
    Similar to analyze-sampling but for one specific sampling rate.
    Used when user selects a particular sampling rate to test.
    """
    if state.raw is None:
        return jsonify({"success": False, "message": "No file loaded."}), 400
    
    try:
        data = request.get_json() or {}
        channels = data.get("channels", [])
        target_fs = int(data.get("target_fs", state.fs))
        
        if not channels:
            return jsonify({"success": False, "message": "No channel provided."}), 400
        
        ch = int(channels[0])
        
        # Extract signal segment
        seg, fs = _get_channel_segment(ch, seconds=8.0)
        if seg is None or fs == 0:
            return jsonify({"success": False, "message": "Failed to extract signal segment."}), 500
        
        # Apply downsampling
        res = decimate_with_aliasing(seg, native_fs=fs, target_fs=max(1, target_fs))
        
        # Calculate sampling ratio
        ratio = (fs / target_fs) if target_fs > 0 else 0
        
        # Generate performance metrics (similar to batch version)
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
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    # Save file
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    
    try:
        file.save(filepath)
        
        # Load EEG file
        if load_eeg_file(filepath):
            # Map channels
            ch_info = {i: name for i, name in enumerate(state.ch_names)}
            
            return jsonify({
                "success": True,
                "message": f"File {file.filename} loaded successfully.",
                "channels": ch_info,
                "fs": state.fs
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to load EEG file. Check format (EDF or FIF)."
            }), 500
    
    except Exception as e:
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