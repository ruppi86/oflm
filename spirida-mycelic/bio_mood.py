"""Bio-Mood module
===================

Defines a minimal enumeration of mycelial *mood* states that can influence
contemplative glyph ecology, duty-cycle limits and other behavioural
parameters.  This is a lightweight placeholder — concrete logic is added in
`bio_interface.py` until a dedicated mood engine evolves.
"""

from enum import Enum, auto

class BioMood(Enum):
    """High-level bio-mood states for a fungal field."""

    CALM = auto()
    TIRED = auto()
    ALERT = auto()
    SUSPICIOUS = auto()

    def __str__(self) -> str:  # pragma: no cover
        return self.name.lower() 