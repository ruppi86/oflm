#!/usr/bin/env python3
"""
Enhanced Bio-Mood System for Spirida-Mycelic
===========================================

Merged into core module (replaces earlier stub). Provides:
• BioMood enum (unchanged names)
• MoodScore & PhysiologicalSignals dataclasses
• EnhancedBioMoodEngine with numeric mood scores, glyph probability modifiers,
  physiological signal integration, and demo utilities.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "BioMood",
    "MoodScore",
    "PhysiologicalSignals",
    "EnhancedBioMoodEngine",
]


class BioMood(Enum):
    """High-level bio-mood states for a fungal field."""

    CALM = "calm"
    TIRED = "tired"
    ALERT = "alert"
    SUSPICIOUS = "suspicious"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


@dataclass
class MoodScore:
    """Continuous mood vector (0–1 for each axis)."""

    energy: float = 0.5
    trust: float = 0.5
    attention: float = 0.5
    coherence: float = 0.5

    def to_discrete(self) -> BioMood:
        """Map vector to discrete mood using simple heuristics."""
        if self.trust < 0.3:
            return BioMood.SUSPICIOUS
        if self.energy < 0.3:
            return BioMood.TIRED
        if self.attention > 0.7:
            return BioMood.ALERT
        return BioMood.CALM


@dataclass
class PhysiologicalSignals:
    """Measured physiological inputs used for mood inference."""

    spike_entropy: float = 0.5
    impedance_drift: float = 0.0
    frequency_stability: float = 1.0
    channel_correlation: float = 0.5
    temperature_gradient: float = 0.0
    ph_stability: float = 1.0


class EnhancedBioMoodEngine:
    """Compute mood from physiological context and provide glyph modifiers."""

    def __init__(self) -> None:
        self.mood: BioMood = BioMood.CALM
        self.scores = MoodScore()
        self.signals = PhysiologicalSignals()

        self.history: deque[Tuple[float, BioMood, MoodScore]] = deque(maxlen=120)
        self.decay = 0.95  # persistence of scores

        self.glyph_modifiers: Dict[str, Dict[BioMood, float]] = {
            "🌌": {BioMood.CALM: 1.2, BioMood.TIRED: 0.1, BioMood.ALERT: 0.7, BioMood.SUSPICIOUS: 0.3},
            "🌊": {BioMood.CALM: 1.0, BioMood.TIRED: 0.8, BioMood.ALERT: 1.3, BioMood.SUSPICIOUS: 0.6},
            "🌪️": {BioMood.CALM: 0.5, BioMood.TIRED: 0.2, BioMood.ALERT: 1.8, BioMood.SUSPICIOUS: 1.5},
            "⭕": {BioMood.CALM: 1.0, BioMood.TIRED: 1.5, BioMood.ALERT: 0.8, BioMood.SUSPICIOUS: 1.2},
            "🌱": {BioMood.CALM: 1.1, BioMood.TIRED: 1.3, BioMood.ALERT: 0.9, BioMood.SUSPICIOUS: 0.4},
        }

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------

    def record_signals(
        self,
        *,
        spike_entropy: Optional[float] = None,
        impedance_drift: Optional[float] = None,
        frequency_stability: Optional[float] = None,
        channel_correlation: Optional[float] = None,
        temperature_gradient: Optional[float] = None,
        ph_stability: Optional[float] = None,
        frequency_intrusion: bool = False,
        care_pause: bool = False,
    ) -> None:
        """Update physiological readings and recompute mood."""
        if spike_entropy is not None:
            self.signals.spike_entropy = spike_entropy
        if impedance_drift is not None:
            self.signals.impedance_drift = impedance_drift
        if frequency_stability is not None:
            self.signals.frequency_stability = frequency_stability
        if channel_correlation is not None:
            self.signals.channel_correlation = channel_correlation
        if temperature_gradient is not None:
            self.signals.temperature_gradient = temperature_gradient
        if ph_stability is not None:
            self.signals.ph_stability = ph_stability

        self._update_scores(frequency_intrusion, care_pause)
        new_mood = self.scores.to_discrete()
        if new_mood != self.mood:
            self.mood = new_mood
        self.history.append((time.time(), self.mood, MoodScore(**vars(self.scores))))

    def glyph_modifier(self, glyph: str) -> float:
        """Return probability multiplier for *glyph* based on current mood."""
        return self.glyph_modifiers.get(glyph, {}).get(self.mood, 1.0)

    def status(self) -> Dict[str, Any]:
        return {
            "mood": self.mood.value,
            "scores": vars(self.scores),
            "modifiers": {g: self.glyph_modifier(g) for g in self.glyph_modifiers},
        }

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _update_scores(self, intrusion: bool, pause: bool) -> None:
        # Energy
        energy_delta = -0.1 * float(abs(self.signals.impedance_drift) > 5) - 0.05 * float(abs(self.signals.temperature_gradient) > 1) - 0.2 * float(pause)
        self._apply_delta("energy", energy_delta)

        # Trust
        trust_delta = -0.3 * float(intrusion) - 0.1 * float(self.signals.channel_correlation < 0.3) - 0.05 * float(self.signals.frequency_stability < 0.5)
        self._apply_delta("trust", trust_delta)

        # Attention
        att_delta = 0.1 * float(self.signals.spike_entropy > 0.8) - 0.1 * float(self.signals.spike_entropy < 0.3) + 0.05 * float(self.signals.frequency_stability > 0.8)
        self._apply_delta("attention", att_delta)

        # Coherence
        coh_delta = 0.05 * float(self.signals.ph_stability > 0.8) + 0.05 * float(self.signals.channel_correlation > 0.7) + 0.05 * float(self.signals.frequency_stability > 0.7) - 0.1 * float(intrusion or pause)
        self._apply_delta("coherence", coh_delta)

    def _apply_delta(self, attr: str, delta: float) -> None:
        val = getattr(self.scores, attr)
        val = val * self.decay + delta
        setattr(self.scores, attr, max(0.0, min(1.0, val))) 