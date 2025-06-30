"""Geometry Compiler – Topology ⇒ Logic
=======================================

Placeholder module inspired by Letter IX (4o).

Given a Boolean truth-table (coming from Adamatzky experiments) this module
will eventually synthesise an *electrode topology* or growth-pattern layout
that encourages the desired logic in living mycelium.

Current status: **stub**.  Provides only data-class shells so other parts of
the code can import it without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Electrode:
    x_mm: float
    y_mm: float
    z_mm: float = 0.0
    role: str = "sense"  # "stim" | "sense" | "ground"

@dataclass
class GeometryPlan:
    electrodes: List[Electrode]
    metadata: Dict[str, Any]


def compile_truth_table(truth_table: Dict[str, int]) -> GeometryPlan:  # pragma: no cover
    """Return a *very* naive default plan until real synthesis arrives."""
    # Simple square with four stimulators and one sense electrode.
    e = [Electrode(0, 0, role="stim"), Electrode(20, 0, role="stim"),
         Electrode(0, 20, role="stim"), Electrode(20, 20, role="stim"),
         Electrode(10, 10, role="sense")]
    return GeometryPlan(electrodes=e, metadata={"note": "placeholder"}) 