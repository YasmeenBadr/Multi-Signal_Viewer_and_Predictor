# shared_utils.py
"""
Shared utilities for ECG and EEG signal processing and visualization.
This module contains common functions used by both ECG and EEG modules.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# SIGNAL PROCESSING UTILITIES
# ============================================================================

def normalize_signal(signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signal data.
    
    Args:
        signal: Input signal array
        method: Normalization method ('zscore', 'minmax')
    
    Returns:
        Normalized signal
    """
    if method == 'zscore':
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    else:
        return signal


def resample_signal_linear(signal: np.ndarray, original_fs: float, 
                           target_fs: float) -> np.ndarray:
    """
    Resample signal using linear interpolation (anti-aliased).
    
    Args:
        signal: Input signal
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency
    
    Returns:
        Resampled signal
    """
    if original_fs == target_fs:
        return signal
    
    n_samples_original = len(signal)
    duration = n_samples_original / original_fs
    n_samples_target = int(duration * target_fs)
    
    # Create time arrays
    time_original = np.linspace(0, duration, n_samples_original)
    time_target = np.linspace(0, duration, n_samples_target)
    
    # Linear interpolation
    resampled = np.interp(time_target, time_original, signal)
    return resampled


def decimate_with_aliasing(signal: np.ndarray, original_fs: float, 
                           target_fs: float, pos_native: int = 0,
                           phase_state: Optional[Dict] = None) -> np.ndarray:
    """
    Decimate signal by simple downsampling (introduces aliasing).
    This simulates lower sampling rate hardware.
    
    Args:
        signal: Input signal (samples x channels) or (samples,)
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency
        pos_native: Current position in native signal (for phase tracking)
        phase_state: Dictionary to store/retrieve phase between calls
    
    Returns:
        Decimated signal with aliasing artifacts
    """
    if original_fs <= target_fs:
        return signal
    
    # Calculate decimation factor
    decimation_factor = original_fs / target_fs
    
    # Handle phase state for continuous streaming
    if phase_state is not None:
        key = str(target_fs)
        phase = phase_state.get(key, 0.0)
    else:
        phase = 0.0
    
    # Simple decimation - just pick samples at intervals
    n_samples = len(signal) if signal.ndim == 1 else signal.shape[0]
    indices = []
    
    current_pos = phase
    while current_pos < n_samples:
        indices.append(int(current_pos))
        current_pos += decimation_factor
    
    # Update phase for next call
    if phase_state is not None:
        phase_state[key] = current_pos - n_samples
    
    if len(indices) == 0:
        return np.array([]) if signal.ndim == 1 else np.empty((0, signal.shape[1]))
    
    # Extract samples
    if signal.ndim == 1:
        return signal[indices]
    else:
        return signal[indices, :]


# ============================================================================
# BUFFER MANAGEMENT
# ============================================================================

class RollingBuffer:
    """
    A rolling buffer for signal data with automatic size management.
    """
    
    def __init__(self, max_samples: int, n_channels: int = 1):
        """
        Initialize rolling buffer.
        
        Args:
            max_samples: Maximum number of samples to store
            n_channels: Number of channels
        """
        self.max_samples = max_samples
        self.n_channels = n_channels
        self.data = [] if n_channels == 1 else [[] for _ in range(n_channels)]
        
    def append(self, new_data: np.ndarray):
        """
        Append new data to buffer and trim if necessary.
        
        Args:
            new_data: New samples to append (samples,) or (samples, channels)
        """
        if self.n_channels == 1:
            self.data.extend(new_data.tolist())
            if len(self.data) > self.max_samples:
                excess = len(self.data) - self.max_samples
                self.data = self.data[excess:]
        else:
            for ch in range(self.n_channels):
                self.data[ch].extend(new_data[:, ch].tolist())
                if len(self.data[ch]) > self.max_samples:
                    excess = len(self.data[ch]) - self.max_samples
                    self.data[ch] = self.data[ch][excess:]
    
    def get_data(self, channel: Optional[int] = None) -> np.ndarray:
        """
        Get buffer data.
        
        Args:
            channel: Channel index (None for all channels)
        
        Returns:
            Buffer data as numpy array
        """
        if self.n_channels == 1:
            return np.array(self.data)
        elif channel is not None:
            return np.array(self.data[channel])
        else:
            return np.array(self.data).T
    
    def clear(self):
        """Clear all buffer data."""
        if self.n_channels == 1:
            self.data = []
        else:
            self.data = [[] for _ in range(self.n_channels)]
    
    def __len__(self):
        """Get buffer length."""
        if self.n_channels == 1:
            return len(self.data)
        else:
            return len(self.data[0]) if self.data else 0


# ============================================================================
# PREDICTION SMOOTHING
# ============================================================================

class PredictionSmoother:
    """
    Smooth predictions over time using a moving average.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize smoother.
        
        Args:
            window_size: Number of predictions to average
        """
        self.window_size = window_size
        self.history = []
    
    def add_prediction(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Add new prediction and return smoothed result.
        
        Args:
            probabilities: Prediction probabilities array
        
        Returns:
            Smoothed probabilities
        """
        self.history.append(probabilities)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Average over history
        smoothed = np.mean(np.stack(self.history, axis=0), axis=0)
        return smoothed
    
    def clear(self):
        """Clear prediction history."""
        self.history = []


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def extract_diagnosis_from_header(file_path: str, 
                                  healthy_keywords: List[str],
                                  disease_keywords: List[str]) -> Optional[str]:
    """
    Extract diagnosis information from a header file.
    
    Args:
        file_path: Path to header file (.hea)
        healthy_keywords: Keywords indicating healthy/normal state
        disease_keywords: Keywords indicating disease/abnormality
    
    Returns:
        Diagnosis string or None
    """
    if not file_path or not Path(file_path).exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            text = f.read()
    except Exception as e:
        logger.warning(f"Failed to read header file: {e}")
        return None
    
    text_lower = text.lower()
    
    # Check for healthy indicators
    if any(keyword in text_lower for keyword in healthy_keywords):
        return "healthy"
    
    # Extract diagnosis line
    for line in text.splitlines():
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in disease_keywords):
            parts = line.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()
            return parts[0].strip()
    
    return None


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def build_recurrence_matrix(x: np.ndarray, y: np.ndarray, 
                            size: int = 128) -> np.ndarray:
    """
    Build a 2D recurrence/density matrix from two signals.
    
    Args:
        x: First signal
        y: Second signal
        size: Output matrix size (size x size)
    
    Returns:
        Recurrence matrix
    """
    try:
        x = np.asarray(x, dtype=np.float32).flatten()
        y = np.asarray(y, dtype=np.float32).flatten()
        
        if len(x) == 0 or len(y) == 0:
            return np.zeros((size, size), dtype=np.float32)
        
        # Calculate value ranges
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        
        # Add small epsilon to avoid division by zero
        if xmin == xmax:
            xmin -= 1e-3
            xmax += 1e-3
        if ymin == ymax:
            ymin -= 1e-3
            ymax += 1e-3
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(
            x, y, bins=size, 
            range=[[xmin, xmax], [ymin, ymax]]
        )
        
        # Log transform and normalize
        H = np.log1p(H)
        H = (H - H.mean()) / (H.std() + 1e-6)
        
        return H.astype(np.float32)
        
    except Exception as e:
        logger.debug(f"Failed to build recurrence matrix: {e}")
        return np.zeros((size, size), dtype=np.float32)


def calculate_xor_difference(current: np.ndarray, previous: np.ndarray,
                             threshold: float = 0.05) -> np.ndarray:
    """
    Calculate XOR-like difference between current and previous signals.
    
    Args:
        current: Current signal window
        previous: Previous signal window
        threshold: Minimum difference to show
    
    Returns:
        XOR difference signal
    """
    if current.shape != previous.shape:
        return current
    
    diff = current - previous
    mask = np.abs(diff) > threshold
    xor_signal = np.where(mask, diff, 0.0)
    
    return xor_signal


# ============================================================================
# RESPONSE FORMATTING
# ============================================================================

def format_prediction_response(label: str, probabilities: List[float],
                               descriptions: Dict[str, str],
                               disease_name: str = "",
                               confidence: Optional[float] = None) -> Dict[str, Any]:
    """
    Format a prediction into a standardized response dictionary.
    
    Args:
        label: Predicted class label
        probabilities: Class probabilities
        descriptions: Dictionary of label descriptions
        disease_name: Specific disease name (if applicable)
        confidence: Confidence score (if None, uses max probability)
    
    Returns:
        Formatted prediction dictionary
    """
    if confidence is None:
        confidence = float(max(probabilities))
    
    return {
        "label": label,
        "probabilities": probabilities,
        "confidence": confidence,
        "description": descriptions.get(label, ""),
        "disease_name": disease_name
    }


def format_streaming_response(time: List[float], 
                              signals: Dict[str, List[float]],
                              prediction: Optional[Dict] = None,
                              metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Format streaming data into a standardized response.
    
    Args:
        time: Time axis values
        signals: Dictionary of channel signals
        prediction: Prediction results
        metadata: Additional metadata
    
    Returns:
        Formatted response dictionary
    """
    response = {
        "time": time,
        "signals": signals
    }
    
    if prediction is not None:
        response["prediction"] = prediction
    
    if metadata is not None:
        response.update(metadata)
    
    return response


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_channels(channels: Any, max_channels: int) -> List[int]:
    """
    Validate and normalize channel selection.
    
    Args:
        channels: Channel specification (int, str, or list)
        max_channels: Maximum number of available channels
    
    Returns:
        List of valid channel indices
    """
    validated = []
    
    # Convert to list
    if isinstance(channels, int):
        validated = [channels]
    elif isinstance(channels, str):
        try:
            validated = [int(x) for x in channels.split(",") if x.strip()]
        except Exception:
            validated = []
    elif isinstance(channels, (list, tuple)):
        for x in channels:
            try:
                validated.append(int(x))
            except Exception:
                continue
    
    # Remove duplicates and filter valid range
    seen = set()
    result = []
    for ch in validated:
        if 0 <= ch < max_channels and ch not in seen:
            result.append(ch)
            seen.add(ch)
    
    # Default to first channels if none valid
    if not result:
        result = list(range(min(12, max_channels)))
    
    return result


def validate_sampling_frequency(requested_fs: float, native_fs: float,
                                min_fs: float = 10, max_fs: float = 500) -> int:
    """
    Validate and clamp sampling frequency to valid range.
    
    Args:
        requested_fs: Requested sampling frequency
        native_fs: Native/original sampling frequency
        min_fs: Minimum allowed frequency
        max_fs: Maximum allowed frequency
    
    Returns:
        Valid sampling frequency as integer
    """
    # Clamp to valid range
    fs = max(min_fs, min(requested_fs, native_fs, max_fs))
    return int(fs)