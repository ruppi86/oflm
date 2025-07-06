import torch
from .spiral_attention import build_spiral_attention_mask

SILENCE_TOKEN_ID = 0

def build_glyph_conditioned_mask(tokens: torch.Tensor, base_mask: torch.Tensor) -> torch.Tensor:
    B, L = tokens.shape
    expanded = base_mask.unsqueeze(0).repeat(B, 1, 1)
    silence = tokens == SILENCE_TOKEN_ID
    for b in range(B):
        silent_idx = silence[b].nonzero(as_tuple=True)[0]
        if silent_idx.numel() == 0:
            continue
        expanded[b, silent_idx, :] = False
    return expanded 