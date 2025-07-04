"""OOD analysis helpers (statistics, visualisations, reporting).

This interim module simply re-exports the corresponding functions from
`spiramycel.cross_validation_evaluation` so that other code (e.g.
`spiramycel.ood_cli`) can depend on a slim stable API while we continue to
refactor the original 1 700-line script.

In a later step we will *move* the full function bodies here and leave thin
shims in the old file, but doing a re-export first avoids large diffs and
keeps backward-compatibility.
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Lazy re-exports to avoid circular import with cross_validation_evaluation
# ---------------------------------------------------------------------------

def _lazy_cv():
    """Import spiramycel.cross_validation_evaluation only when needed."""
    from importlib import import_module
    return import_module("spiramycel.cross_validation_evaluation_old")


def perform_statistical_analysis(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_cv().perform_statistical_analysis(*args, **kwargs)


def create_visualizations(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_cv().create_visualizations(*args, **kwargs)


def generate_cross_validation_report(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_cv().generate_cross_validation_report(*args, **kwargs)


def generate_statistical_report(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_cv().generate_statistical_report(*args, **kwargs)


def calc_effect_size(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_cv().calc_effect_size(*args, **kwargs)


def safe_welch(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_cv().safe_welch(*args, **kwargs)


__all__: list[str] = [
    "perform_statistical_analysis",
    "create_visualizations",
    "generate_cross_validation_report",
    "generate_statistical_report",
    "calc_effect_size",
    "safe_welch",
] 

perform_statistical_analysis
create_visualizations
generate_cross_validation_report
generate_statistical_report