"""OOD utility helpers extracted from cross_validation_evaluation.

This thin wrapper allows notebooks or other packages to import the
OOD loading / evaluation helpers **without** depending on the 1 700-line
script directly.

Once the full refactor lands these functions will be moved here
permanently; for now we simply re-export the existing implementations
so that external callers can begin using a stable API immediately.
"""

from __future__ import annotations

from typing import Dict, Any

# Import the monolithic script lazily so CLI execution order stays intact
import importlib
_cv = importlib.import_module("spiramycel.cross_validation_evaluation")

load_trained_models = _cv.load_trained_models  # type: ignore[attr-defined]
load_ood_test_set = _cv.load_ood_test_set      # type: ignore[attr-defined]
evaluate_model_on_ood = _cv.evaluate_model_on_ood  # type: ignore[attr-defined]
filter_scenarios_for_model = _cv.filter_scenarios_for_model  # type: ignore[attr-defined]

__all__: list[str] = [
    "load_trained_models",
    "load_ood_test_set",
    "evaluate_model_on_ood",
    "filter_scenarios_for_model",
] 