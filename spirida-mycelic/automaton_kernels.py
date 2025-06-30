"""Automaton Kernels – Species-specific local rules
=================================================

Stub implementation for per-species cellular-automaton rules that will govern
how glyphs propagate inside a fungal field (Letter IX).
"""

from typing import Dict, Tuple

from .semantic_guardian import FungalSpecies  # re-use existing enum

# 3-neighbour rule: (left, centre, right) -> next_state glyph string
SPECIES_RULES: Dict[FungalSpecies, Dict[Tuple[str, str, str], str]] = {
    FungalSpecies.PLEUROTUS_OSTREATUS: {
        ("REST", "REST", "SEED"): "REST",
    },
    FungalSpecies.GANODERMA_RESINACEUM: {},
}


def next_state(species: FungalSpecies, left: str, centre: str, right: str) -> str:
    """Return next glyph state for neighbourhood pattern."""
    rules = SPECIES_RULES.get(species, {})
    return rules.get((left, centre, right), centre) 