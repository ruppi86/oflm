from __future__ import annotations

"""spiramycel_bridge.py – Optional integration layer between the stand-alone
Neural-Mycelic Emulator package and the *Spiramycel* ecosystem.

Goals
-----
1. **Zero hard dependency** – The emulator remains perfectly functional even if
   *Spiramycel* is not installed.  All imports are guarded and a graceful
   fallback is provided.
2. **Adapter shim** – Expose the LSTMEmulator using the API surface that
   `spiramycel.cross_validation_evaluation.evaluate_model_on_ood` expects, i.e.
   a `forward()` returning ``(glyph_logits, eff_logits, silence_logits, ...)``.
3. **Convenience helpers** – Utilities to load a trained emulator checkpoint
   and generate glyph sequences given a start token.

Example
-------
>>> from neural_mycelic_emulator.spiramycel_bridge import load_emulator_for_spiramycel
>>> model = load_emulator_for_spiramycel(
...     tag="cordyceps_small",
...     checkpoint="neural_mycelic_emulator/models/cordyceps_small/cordyceps_small_best.pt",
... )
>>> # Now `model` can be passed into Spiramycel OOD utilities.

Author: o3 automated bridge generator • 2025-07-02
"""

from pathlib import Path
from typing import Any, Tuple, Optional, Sequence

# ---- Optional Torch --------------------------------------------------------
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover – keep runtime tiny if no torch
    nn = object  # type: ignore
    TORCH_AVAILABLE = False

# ---- Local imports ---------------------------------------------------------
from neural_mycelic_emulator.models.lstm_emulator import LSTMEmulator
from neural_mycelic_emulator.models.trainer import load_config, DEVICE  # re-use device logic

# ---- Optional Spiramycel bits ---------------------------------------------
SPIRAMYCEL_AVAILABLE = False
try:
    from spiramycel.glyph_codec import SpiramycelGlyphCodec  # type: ignore
    SPIRAMYCEL_AVAILABLE = True
except ModuleNotFoundError:
    SpiramycelGlyphCodec = None  # type: ignore


# ---------------------------------------------------------------------------
# Adapter shim
# ---------------------------------------------------------------------------
class _EmulatorAdapter(nn.Module if TORCH_AVAILABLE else object):
    """Wrap an :class:`~neural_mycelic_emulator.models.lstm_emulator.LSTMEmulator`
    so that it *looks* like a :class:`spiramycel.neural_trainer.SpiramycelNeuralModel`.

    The Spiramycel evaluation code only inspects three outputs:

    1. ``glyph_logits`` – shape (B, T, V)
    2. ``eff_logits``  – shape (B, T, 1)  (effectiveness head)
    3. ``silence_logits`` – shape (B, T, 1)  (silence probability)

    Any additional return values are ignored.  Consequently we can supply
    *dummy* tensors for (2) and (3) filled with zeros while still allowing the
    OOD utilities to operate.
    """

    def __init__(self, base: LSTMEmulator):
        if TORCH_AVAILABLE:
            super().__init__()
        self._base = base
        self.vocab_size = base.vocab_size

        # Freeze base model – this is inference-only
        if TORCH_AVAILABLE:
            for p in self._base.parameters():
                p.requires_grad_(False)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def generate(self, *args, **kwargs):  # passthrough for convenience
        return self._base.generate(*args, **kwargs)

    # ---------------------------------------------------------------------
    # nn.Module interface expected by Spiramycel
    # ---------------------------------------------------------------------
    def forward(
        self,
        glyph_tokens: "torch.Tensor",  # (B, T)
        conditions: Optional["torch.Tensor"] = None,  # ignored – emulator is unconditional
        *_,
        **__,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", None, None, None]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available – cannot forward through adapter")

        logits, _ = self._base(glyph_tokens)
        # Fake heads: zeros with proper shape so downstream code is happy
        eff = torch.zeros((*logits.shape[:-1], 1), device=logits.device)
        sil = torch.zeros_like(eff)
        return logits, eff, sil, None, None, None


# ---------------------------------------------------------------------------
# Helper – load checkpoint & wrap
# ---------------------------------------------------------------------------

def load_emulator_for_spiramycel(
    tag: str,
    checkpoint: Path | str,
    cfg_path: Path | str | None = None,
    device: "torch.device" | str = DEVICE,
) -> _EmulatorAdapter:
    """Load a trained emulator checkpoint and return an adapter instance.

    Parameters
    ----------
    tag : str
        The *model tag* as defined in ``emulator_parameters.yml``.
    checkpoint : Path | str
        Path to the ``.pt`` file saved by ``trainer.py``.
    cfg_path : Path | str | None, optional
        Path to the YAML config.  Uses the default next to trainer.py if None.
    device : torch.device | str, default = trainer.DEVICE
        Device to materialise the model on.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load an emulator checkpoint.")

    checkpoint = Path(checkpoint)
    if cfg_path is None:
        cfg_path = Path(__file__).parent / "models" / "emulator_parameters.yml"

    cfg = load_config(tag, Path(cfg_path))
    base = LSTMEmulator(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    base.load_state_dict(state)
    base.eval()

    return _EmulatorAdapter(base)


# ---------------------------------------------------------------------------
# Convenience – glyph formatting
# ---------------------------------------------------------------------------

def format_glyph_sequence(seq: Sequence[int]) -> str:
    """Return a human-readable string representation of *seq*.

    • If *Spiramycel* is available we delegate to its codec for consistency.
    • Otherwise we fall back to a simple hex-style formatting.
    """
    if SPIRAMYCEL_AVAILABLE and SpiramycelGlyphCodec is not None:
        return SpiramycelGlyphCodec().format_glyph_sequence(seq)
    return " ".join(f"g{g:02X}" for g in seq)


# ---------------------------------------------------------------------------
# CLI entry point – minimal demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover – manual smoke test
    import argparse, sys

    p = argparse.ArgumentParser("Neural-Mycelic Emulator ⇢ Spiramycel bridge demo")
    p.add_argument("tag", help="model tag in emulator_parameters.yml")
    p.add_argument("checkpoint", type=Path, help="Path to trained .pt file")
    p.add_argument("--generate", type=int, default=64, help="Length of sequence to sample")
    args = p.parse_args()

    adapter = load_emulator_for_spiramycel(args.tag, args.checkpoint)

    start = 8  # channel-0 prefix under new 0–7 activity / 8–15 channel scheme
    seq = adapter.generate(start_token=start, max_len=args.generate, device=DEVICE)
    print(format_glyph_sequence(seq)) 