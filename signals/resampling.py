# resampling.py
"""
Signal resampling utilities for ECG, EEG, and other signal processing modules.
Place this file in the signals/ directory.
"""

import numpy as np
from typing import Dict, Optional


def resample_signal(signal: np.ndarray, original_fs: float, 
                   target_fs: float, method: str = 'linear') -> np.ndarray:
    """
    Resample signal with anti-aliasing (proper resampling).
    
    Args:
        signal: Input signal (samples,) or (samples, channels)
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency
        method: Resampling method ('linear', 'cubic')
    
    Returns:
        Resampled signal
    """
    if original_fs == target_fs:
        return signal
    
    is_1d = signal.ndim == 1
    if is_1d:
        signal = signal[:, np.newaxis]
    
    n_samples, n_channels = signal.shape
    duration = n_samples / original_fs
    n_samples_target = int(duration * target_fs)
    
    # Create time arrays
    time_original = np.linspace(0, duration, n_samples)
    time_target = np.linspace(0, duration, n_samples_target)
    
    # Resample each channel
    resampled = np.zeros((n_samples_target, n_channels))
    for ch in range(n_channels):
        resampled[:, ch] = np.interp(time_target, time_original, signal[:, ch])
    
    if is_1d:
        return resampled[:, 0]
    return resampled


def decimate_with_aliasing(signal: np.ndarray, native_fs: float, 
                           target_fs: float, pos_native: int = 0,
                           phase_state: Optional[Dict] = None) -> np.ndarray:
    """
    Decimate signal by simple downsampling (introduces aliasing).
    This simulates lower sampling rate hardware and demonstrates aliasing effects.
    
    Args:
        signal: Input signal (samples,) or (samples, channels)
        native_fs: Native/original sampling frequency
        target_fs: Target sampling frequency
        pos_native: Current position in native signal (for phase continuity)
        phase_state: Dictionary to store phase between calls for streaming
    
    Returns:
        Decimated signal with aliasing artifacts
    """
    # No downsampling needed
    if native_fs <= target_fs:
        return signal
    
    # Handle 1D vs 2D signals
    is_1d = signal.ndim == 1
    if is_1d:
        signal = signal[:, np.newaxis]
    
    n_samples, n_channels = signal.shape
    decimation_factor = native_fs / target_fs
    
    # Retrieve or initialize phase for continuous streaming
    if phase_state is not None:
        phase_key = f"{target_fs}"
        phase = phase_state.get(phase_key, 0.0)
    else:
        phase = 0.0
    
    # Calculate sample indices to keep (simple decimation)
    indices = []
    current_pos = phase
    while current_pos < n_samples:
        indices.append(int(current_pos))
        current_pos += decimation_factor
    
    # Update phase for next call (continuous streaming)
    if phase_state is not None:
        phase_state[phase_key] = current_pos - n_samples
    
    # Return empty if no samples
    if len(indices) == 0:
        if is_1d:
            return np.array([])
        return np.empty((0, n_channels))
    
    # Extract decimated samples
    decimated = signal[indices, :]
    
    if is_1d:
        return decimated[:, 0]
    return decimated


def calculate_decimation_factor(native_fs: float, target_fs: float) -> int:
    """
    Calculate integer decimation factor.
    
    Args:
        native_fs: Native sampling frequency
        target_fs: Target sampling frequency
    
    Returns:
        Integer decimation factor
    """
    if target_fs >= native_fs:
        return 1
    return max(1, int(round(native_fs / target_fs)))


def estimate_aliasing_level(native_fs: float, target_fs: float) -> str:
    """
    Estimate the level of aliasing based on sampling frequencies.
    
    Args:
        native_fs: Native sampling frequency
        target_fs: Target sampling frequency
    
    Returns:
        Aliasing level description ('none', 'light', 'moderate', 'heavy', 'severe')
    """
    factor = calculate_decimation_factor(native_fs, target_fs)
    
    if factor == 1:
        return "none"
    elif factor <= 2:
        return "light"
    elif factor <= 4:
        return "moderate"
    elif factor <= 8:
        return "heavy"
    else:
        return "severe"


def get_nyquist_limit(sampling_fs: float) -> float:
    """
    Calculate Nyquist frequency limit.
    
    Args:
        sampling_fs: Sampling frequency
    
    Returns:
        Nyquist frequency (fs / 2)
    """
    return sampling_fs / 2.0