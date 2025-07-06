import torch
import torch.nn as nn

from ..utils.breath_clock import BreathClock
from ..utils.positional import SinusoidalPositionalEmbedding
from .spiral_attention import build_spiral_attention_mask
from .dynamic_mask import build_glyph_conditioned_mask


class SpiralFormer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, seq_len=256, num_layers=4, vocab_size=128):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=seq_len)

        self.layers = nn.ModuleList([
            _SpiralBlock(d_model, n_heads) for _ in range(num_layers)
        ])

        self.breath = BreathClock()
        base_mask = build_spiral_attention_mask(seq_len)
        self.register_buffer("base_mask", base_mask, persistent=False)

    def forward(self, tokens: torch.Tensor, t: float):
        x = self.embed(tokens)
        x = self.pos(x)
        attn_mask_batch = build_glyph_conditioned_mask(tokens, self.base_mask)
        for layer in self.layers:
            x = layer(x, t, attn_mask_batch, self.breath)
        return x


class _SpiralBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, t: float, attn_mask_batch: torch.Tensor, breath: BreathClock):
        phase = breath.phase_at(t)
        weight = breath.weight_for_phase(phase)
        if weight == 0.0:
            y = x
        else:
            ignore = ~attn_mask_batch
            y, _ = self.attn(x, x, x, attn_mask=ignore)
            y = y * weight
        x = self.norm1(x + y)
        z = self.ff(x)
        return self.norm2(x + z) 