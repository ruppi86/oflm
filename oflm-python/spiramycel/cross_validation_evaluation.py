"""Deprecated monolithic OOD evaluator.

The full implementation has been split into:
  •  ``spiramycel.ood_cli``   – CLI entry point
  •  ``spiramycel.ood_utils`` – data/model helpers
  •  ``spiramycel.ood_analysis`` – statistics & reporting

This stub keeps legacy imports working **and** delegates CLI execution to
``ood_cli.main`` so external scripts that still invoke

    python -m spiramycel.cross_validation_evaluation ...

continue to work without code changes.
"""

from __future__ import annotations

from .ood_analysis import (
    perform_statistical_analysis,
    create_visualizations,
    generate_cross_validation_report,
    generate_statistical_report,
)

# ---------------------------------------------------------------------------
# CLI shim – just call the new ood_cli.main
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    from . import ood_cli

    ood_cli.main()


if __name__ == "__main__":
    main()

# Re-export public helpers for backward compatibility
# Lazy re-exports to avoid circular import with `spiramycel.ood_utils`

from typing import Any

def _lazy_ood_utils():
    """Import spiramycel.ood_utils on first use to prevent import cycles."""
    from importlib import import_module
    return import_module("spiramycel.ood_utils")

def load_trained_models(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_ood_utils().load_trained_models(*args, **kwargs)

def load_ood_test_set(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_ood_utils().load_ood_test_set(*args, **kwargs)

def evaluate_model_on_ood(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_ood_utils().evaluate_model_on_ood(*args, **kwargs)

def filter_scenarios_for_model(*args: Any, **kwargs: Any):  # type: ignore[override]
    return _lazy_ood_utils().filter_scenarios_for_model(*args, **kwargs)