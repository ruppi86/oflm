import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, List

__all__ = [
    "detect_spikes",
]

def detect_spikes(signal: np.ndarray, threshold_sigma: float = 4.0) -> np.ndarray:
    """Return boolean array where spikes occur.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of (normalized) voltage readings.
    threshold_sigma : float
        Peaks above `mean + threshold_sigma * std` are considered spikes.

    Returns
    -------
    np.ndarray
        Boolean mask of same length as `signal`.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")

    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma == 0:
        return np.zeros_like(signal, dtype=bool)

    thresh = mu + threshold_sigma * sigma
    peaks, _ = find_peaks(signal, height=thresh)
    mask = np.zeros_like(signal, dtype=bool)
    mask[peaks] = True
    return mask 