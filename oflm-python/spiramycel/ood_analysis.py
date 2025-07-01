"""Statistical analysis helpers extracted from cross_validation_evaluation.

External code can now do::

    from spiramycel.ood_analysis import perform_statistical_analysis

without importing the CLI script.
"""

from __future__ import annotations
import importlib

_cv = importlib.import_module("spiramycel.cross_validation_evaluation")

perform_statistical_analysis = _cv.perform_statistical_analysis  # type: ignore[attr-defined]
create_visualizations = _cv.create_visualizations  # type: ignore[attr-defined]
generate_cross_validation_report = _cv.generate_cross_validation_report  # type: ignore[attr-defined]
generate_statistical_report = _cv.generate_statistical_report  # type: ignore[attr-defined]
calc_effect_size = _cv.calc_effect_size  # type: ignore[attr-defined]
safe_welch = _cv.safe_welch  # type: ignore[attr-defined]

__all__: list[str] = [
    "perform_statistical_analysis",
    "create_visualizations",
    "generate_cross_validation_report",
    "generate_statistical_report",
    "calc_effect_size",
    "safe_welch",
] 