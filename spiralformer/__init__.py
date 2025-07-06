"""Spiralformer – contemplative transformer experiments."""

from .core.model import SpiralFormer
from .utils.breath_clock import BreathClock
from .utils.positional import SinusoidalPositionalEmbedding
from .core.spiral_attention import build_spiral_attention_mask
from .core.dynamic_mask import build_glyph_conditioned_mask

__all__ = [
    "SpiralFormer",
    "BreathClock",
    "SinusoidalPositionalEmbedding",
    "build_spiral_attention_mask",
    "build_glyph_conditioned_mask",
] 