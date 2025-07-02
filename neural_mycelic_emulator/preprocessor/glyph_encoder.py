from typing import List

GLYPHS = {
    # Activity level glyphs
    "SIL": 0x31,   # ⭕ silence
    "LOW": 0x3A,   # 🌱 single spike
    "FLOW": 0x32,  # 🌊 2–3 spikes
    "BURST": 0x34, # 🌋 4–6 spikes
    "STORM": 0x33, # 🌪 7–10 spikes
    "CONST": 0x3E, # 🌌 >10 spikes
    # Channel prefix glyphs will be added dynamically (0x50–0x57)
}

__all__ = ["encode_word", "GLYPHS"]


def encode_word(word_len: int) -> int:
    """Map spike *length* to glyph id."""
    if word_len == 0:
        return GLYPHS["SIL"]
    if word_len == 1:
        return GLYPHS["LOW"]
    if word_len <= 3:
        return GLYPHS["FLOW"]
    if word_len <= 6:
        return GLYPHS["BURST"]
    if word_len <= 10:
        return GLYPHS["STORM"]
    return GLYPHS["CONST"] 