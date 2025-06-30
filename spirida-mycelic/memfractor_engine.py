"""Memfractor Engine – Curvature-based memory modulation
========================================================

A placeholder for the *mem-fractive* behaviour described in Letter IX.  It
keeps a history of `SpikeEvent`s and produces a simple modulation factor that
other components (e.g. CapacitanceFade) can consume.
"""

from collections import deque
from typing import Deque, List

import numpy as np

# Local import guarded to avoid circular dependency during skeleton phase
try:
    from .bio_interface import SpikeEvent  # type: ignore
except ImportError:  # pragma: no cover
    class SpikeEvent:  # minimal stand-in
        timestamp: float
        amplitude_pattern: List[float]


class MemfractorEngine:
    """Track recent spike curvature and output modulation factor (0.8–1.2)."""

    def __init__(self, window_size: int = 256):
        self.window_size = window_size
        self.history: Deque[SpikeEvent] = deque(maxlen=window_size)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def record(self, spike: SpikeEvent) -> None:
        self.history.append(spike)

    def modulation_factor(self) -> float:
        """Very crude placeholder: returns factor based on amplitude variance."""
        if len(self.history) < 2:
            return 1.0
        amps = np.array([np.max(s.amplitude_pattern) for s in self.history])
        variability = np.std(amps) / (np.mean(amps) + 1e-6)
        return float(np.clip(1.0 + variability * 0.2, 0.8, 1.2)) 