import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, List

__all__ = [
    "detect_spikes",
]

def detect_spikes(signal: np.ndarray, threshold_sigma: float | None = 4.0) -> np.ndarray:
    """Return boolean array where spikes occur.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of (normalized) voltage readings.
    threshold_sigma : float | None
        If a float, peaks above `mean + threshold_sigma * std` are considered spikes.
        If None, an adaptive threshold based on the *median absolute deviation*
        is used (≈ 4σ for Gaussian noise, but robust to outliers).

    Returns
    -------
    np.ndarray
        Boolean mask of same length as `signal`.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")

    # ------------------------------------------------------------------
    # Robust threshold estimate
    # ------------------------------------------------------------------
    if threshold_sigma is None:
        # Median Absolute Deviation → consistent with σ under Gaussian noise
        median = np.median(signal)
        mad = np.median(np.abs(signal - median)) + 1e-9
        sigma_est = mad * 1.4826  # MAD→σ conversion factor
        thresh = median + 4.0 * sigma_est  # keep historical ~4σ default
    else:
        mu = np.mean(signal)
        sigma = np.std(signal)
        if sigma == 0:
            return np.zeros_like(signal, dtype=bool)
        thresh = mu + threshold_sigma * sigma

    peaks, _ = find_peaks(signal, height=thresh)
    mask = np.zeros_like(signal, dtype=bool)
    mask[peaks] = True
    return mask 