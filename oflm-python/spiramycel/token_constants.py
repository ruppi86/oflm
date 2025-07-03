START_TOKEN = 0x00  # Reserved for sequence start
END_TOKEN = 0x41    # Reserved for sequence end (after 64-glyph vocab)
PAD_TOKEN = 0x42    # Dedicated padding token (ignored in loss)

__all__ = [
    "START_TOKEN",
    "END_TOKEN",
    "PAD_TOKEN",
] 