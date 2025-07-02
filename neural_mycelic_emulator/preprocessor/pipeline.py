from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

from .detect_spikes import detect_spikes
from .group_spikes import group_spikes
from .glyph_encoder import encode_word

__all__ = ["tsv_to_glyph_sequences"]


def tsv_to_glyph_sequences(tsv_path: Path, channel_cols: List[str] | None = None) -> List[int]:
    """Convert TSV of multi-channel voltages to single glyph sequence.

    Steps:
    1. Load TSV via pandas (tab-separated, no header → col names ch0…).
    2. For each channel: z-score, detect spikes, group into words, encode words.
    3. Interleave channel-ID prefix glyphs (🅒0..).

    Returns a flat list of glyph IDs.
    """
    # Robust numeric load: skip possible text header, coerce non-numeric to NaN
    df = pd.read_csv(tsv_path, sep="\t", header=None, comment="#", engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(method="ffill").fillna(method="bfill")

    if channel_cols is None:
        channel_cols = df.columns.tolist()

    seq: List[int] = []
    for ch_idx in channel_cols:
        sig = df[ch_idx].astype(float).to_numpy()
        sig_z = (sig - sig.mean()) / (sig.std() + 1e-9)
        mask = detect_spikes(sig_z)
        words = group_spikes(mask)
        # Prefix channel glyph (simple 0x50 + idx)
        ch_prefix = 0x50 + ch_idx  # e.g., 0x50,0x51...
        for w in words:
            seq.append(ch_prefix)
            seq.append(encode_word(len(w)))
    return seq 