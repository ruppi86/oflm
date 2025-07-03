from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple

from .detect_spikes import detect_spikes
from .group_spikes import group_spikes
from .glyph_encoder import encode_word, GLYPHS

__all__ = ["tsv_to_glyph_sequences"]


def tsv_to_glyph_sequences(
    tsv_path: Path,
    channel_cols: List[str] | None = None,
    *,
    return_channels: bool = False,
) -> List[int] | Tuple[List[int], int]:
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
    # df = df.fillna(method="ffill").fillna(method="bfill")

    df = df.ffill().bfill()

    if channel_cols is None:
        channel_cols = df.columns.tolist()
    n_channels = len(channel_cols)

    seq: List[int] = []
    for ch_idx in channel_cols:
        sig = df[ch_idx].astype(float).to_numpy()
        sig_z = (sig - sig.mean()) / (sig.std() + 1e-9)
        mask = detect_spikes(sig_z, threshold_sigma=None)
        words = group_spikes(mask)
        # Prefix channel glyph mapped to small contiguous ids 8–15; wrap if >7
        if ch_idx >= 8:
            # Wrap-around to keep within 16-token vocab; logically channels 8,16,… share prefix 8 etc.
            ch_prefix = 8 + (ch_idx % 8)
        else:
            ch_prefix = 8 + ch_idx
        prev_end = None
        silence_gap = 20  # seconds (samples) threshold for explicit silence
        for w in words:
            # Insert silence glyph if large gap since previous word in this channel
            if prev_end is not None and (w[0] - prev_end) > silence_gap:
                seq.append(GLYPHS["SIL"])
            seq.append(ch_prefix)
            seq.append(encode_word(len(w)))
            prev_end = w[-1]

    if return_channels:
        return seq, n_channels
    return seq 