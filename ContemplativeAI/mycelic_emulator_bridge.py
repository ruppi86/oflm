from __future__ import annotations

"""mycelic_emulator_bridge.py – ContemplativeAI ↔ Neural-Mycelic Emulator

A lightweight analogue to *oflm_bridge.py* that ferries single breath-
fragments to the **Neural-Mycelic Emulator**.  It is intentionally
minimal – the emulator is unconditional and produces glyph sequences
without needing network conditions – but we respect the same
contemplative vows:

1. **One fragment at a time** – no long logs.
2. **One-way forgetting** – responses may be immediately composted.
3. **EXHALE-only traffic** – integration point for Pulmonos phases.

The bridge is *optional*: if ``neural_mycelic_emulator`` (and PyTorch)
are unavailable we fall back to a deterministic mock-response so that
ContemplativeAI scripts remain runnable.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import random

# ---------------------------------------------------------------------------
# Pulmonos phase fallback (identical to oflm_bridge)
# ---------------------------------------------------------------------------
try:
    from pulmonos_daemon import Phase
except ImportError:
    class Phase(Enum):
        INHALE = auto()
        HOLD = auto()
        EXHALE = auto()
        REST = auto()

# ---------------------------------------------------------------------------
# Optional emulator import
# ---------------------------------------------------------------------------
EMULATOR_AVAILABLE = False
TORCH_AVAILABLE = False
adapter = None
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    # Ensure workspace root on PYTHONPATH so relative import works when
    # ContemplativeAI is executed as a package.
    root_dir = Path(__file__).resolve().parents[1]
    emulator_path = root_dir / "neural_mycelic_emulator"
    if emulator_path.exists() and emulator_path.as_posix() not in sys.path:
        sys.path.insert(0, emulator_path.as_posix())

    from neural_mycelic_emulator.spiramycel_bridge import (
        load_emulator_for_spiramycel as _load_adapter,
        format_glyph_sequence as _fmt,
    )

    EMULATOR_AVAILABLE = True
except ModuleNotFoundError:
    # Silent failure – we will simulate
    def _fmt(seq):
        return " ".join(f"g{g:02X}" for g in seq)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class EmulatorResponseType(Enum):
    SILENCE = "silence"
    GLYPH_SEQUENCE = "glyph_sequence"

@dataclass
class EmulatorBreath:
    fragment: str
    response_type: EmulatorResponseType
    content: str
    glyph_sequence: Optional[List[int]]
    silence_ratio: float
    timestamp: float

    def is_audible(self) -> bool:
        return self.response_type == EmulatorResponseType.GLYPH_SEQUENCE

# ---------------------------------------------------------------------------
# Bridge logic
# ---------------------------------------------------------------------------
class NeuralMycelicBridge:
    def __init__(
        self,
        model_tag: str = "cordyceps_small",
        checkpoint: Optional[Path | str] = None,
        max_generate_len: int = 48,
    ) -> None:
        self.max_generate_len = max_generate_len
        self.current_phase: Phase = Phase.REST

        # Load emulator if possible
        self.adapter = None
        if EMULATOR_AVAILABLE and TORCH_AVAILABLE:
            if checkpoint is None:
                # Derive default checkpoint path relative to ContemplativeAI
                ckpt_default = (
                    emulator_path / "models" / model_tag / f"{model_tag}_best.pt"
                )
                checkpoint = ckpt_default if ckpt_default.exists() else None
            if checkpoint is not None and Path(checkpoint).exists():
                try:
                    self.adapter = _load_adapter(model_tag, checkpoint)
                    print(f"🧩 Neural-Mycelic emulator loaded ({model_tag}).")
                except Exception as e:  # pragma: no cover – avoid fatal
                    print(f"⚠️  Failed loading emulator: {e}. Using mock mode.")
            else:
                print("⚠️  Checkpoint not found – using mock mode.")
        else:
            print("⚠️  Emulator not available – using mock mode.")

    # ------------------------------------------------------------------
    # Public API – single exhale exchange
    # ------------------------------------------------------------------
    async def exhale_exchange(
        self,
        fragment: str,
        phase: Phase = Phase.EXHALE,
        community_pressure: float = 0.5,
    ) -> EmulatorBreath:
        now = time.time()

        if phase != Phase.EXHALE or community_pressure > 0.6:
            return EmulatorBreath(
                fragment=fragment,
                response_type=EmulatorResponseType.SILENCE,
                content="",
                glyph_sequence=None,
                silence_ratio=1.0,
                timestamp=now,
            )

        if self.adapter is None:
            # Mock response – deterministic for reproducibility
            random.seed(hash(fragment) % (2**32))
            if random.random() < 0.7:
                return EmulatorBreath(
                    fragment=fragment,
                    response_type=EmulatorResponseType.SILENCE,
                    content="",
                    glyph_sequence=None,
                    silence_ratio=1.0,
                    timestamp=now,
                )
            seq = [8] + [random.randint(0, 15) for _ in range(5)]
            return EmulatorBreath(
                fragment=fragment,
                response_type=EmulatorResponseType.GLYPH_SEQUENCE,
                content=_fmt(seq),
                glyph_sequence=seq,
                silence_ratio=seq.count(0) / len(seq),
                timestamp=now,
            )

        # Real generation
        import torch

        with torch.no_grad():
            seq: List[int] = self.adapter.generate(
                start_token=8, max_len=self.max_generate_len, device="cpu"
            )
        silence_ratio = seq.count(0) / len(seq)
        if silence_ratio > 0.6:
            r_type = EmulatorResponseType.SILENCE
            content = ""
        else:
            r_type = EmulatorResponseType.GLYPH_SEQUENCE
            content = _fmt(seq)
        return EmulatorBreath(
            fragment=fragment,
            response_type=r_type,
            content=content,
            glyph_sequence=None if r_type == EmulatorResponseType.SILENCE else seq,
            silence_ratio=silence_ratio,
            timestamp=now,
        ) 