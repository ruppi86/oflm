from typing import List

GLYPHS = {
    # 8 amplitude/activity levels – small contiguous ids 0–7
    "SIL": 0,      # explicit silence marker
    "LOW1": 1,     # single spike
    "LOW2": 2,     # very gentle word (≤2 spikes)
    "MED1": 3,     # calm word   (≤3 spikes)
    "MED2": 4,     # moderate    (≤5 spikes)
    "HIGH1": 5,    # strong      (≤7 spikes)
    "HIGH2": 6,    # intense     (≤10 spikes)
    "CONST": 7,    # sustained burst (>10 spikes)
}

__all__ = ["encode_word", "GLYPHS"]


# Updated mapping to eight activity glyphs (including finer grained mid-levels).
# Mapping is based solely on *word length* – the number of contiguous spikes
# belonging to the same word – which serves as a reasonable proxy for
# amplitude/duration until we implement explicit peak-magnitude analysis.

def encode_word(word_len: int) -> int:
    """Map spike *length* to glyph id (8-level amplitude encoding)."""
    if word_len <= 0:
        return GLYPHS["SIL"]
    if word_len == 1:
        return GLYPHS["LOW1"]
    if word_len == 2:
        return GLYPHS["LOW2"]
    if word_len == 3:
        return GLYPHS["MED1"]
    if word_len <= 5:
        return GLYPHS["MED2"]
    if word_len <= 7:
        return GLYPHS["HIGH1"]
    if word_len <= 10:
        return GLYPHS["HIGH2"]
    return GLYPHS["CONST"] 