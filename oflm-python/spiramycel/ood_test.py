"""End-to-end OOD cross-validation smoke-test.

This lightweight script exercise the *modular* evaluation stack built
around
    • spiramycel.ood_utils    – dataset / model helpers
    • spiramycel.ood_analysis – statistics + reporting

It intentionally avoids the monolithic legacy implementation and can be
run either as a quick CLI tool **or** via pytest to serve as a regression
check in CI.

Usage (CLI) – run from package root::

    python -m spiramycel.ood_test --scale auto [--environment same]

Usage (pytest) – discovers automatically::

    pytest -q spiramycel/ood_test.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import argparse
import logging

from .ood_utils import (
    load_trained_models,
    load_ood_test_set,
    evaluate_model_on_ood,
    filter_scenarios_for_model,
)
from .ood_analysis import perform_statistical_analysis
from .logging_utils import setup_experiment_logging

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _evaluate_all_models(scale: str, environment: str) -> Dict[str, Dict[str, Any]]:
    """Load models + data and run OOD evaluation (no plots, no reports)."""
    models = load_trained_models(preferred_scale=scale)
    test_scenarios = load_ood_test_set(use_expanded=True, environment=environment)

    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        if model is None:
            logging.warning("%s unavailable ‑ skipping", name)
            continue
        scen_subset = (
            filter_scenarios_for_model(name, test_scenarios)
            if environment == "same"
            else test_scenarios
        )
        results[name] = evaluate_model_on_ood(model, name, scen_subset, codec=None)
    return results


def _basic_sanity_checks(all_results: Dict[str, Dict[str, Any]]):
    """Raise AssertionError if obvious problems are detected."""
    if not all_results:
        logging.warning("No models loaded – ensure correct checkpoints exist for the requested scale.")
        return  # treat as skipped rather than failure
    for model_name, model_results in all_results.items():
        assert model_results, f"{model_name} produced zero scenarios"
        for scen, data in model_results.items():
            assert "silence_ratio" in data, f"{model_name}/{scen} missing silence_ratio"
            assert 0.0 <= data["silence_ratio"] <= 1.0, "Silence ratio outside [0,1]"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Modular OOD cross-validation smoke test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scale", default="auto", choices=["25k", "200k", "600k", "6m", "auto"],
                        help="Model scale to evaluate")
    parser.add_argument("--environment", default="same", choices=["same", "switch"],
                        help="Stress-level crossover ('same') or alien environments ('switch')")
    args = parser.parse_args()

    log_file, _ = setup_experiment_logging()
    logging.info("🚀 Starting modular OOD smoke test – scale=%s env=%s", args.scale, args.environment)

    all_results = _evaluate_all_models(args.scale, args.environment)
    _basic_sanity_checks(all_results)

    stats = perform_statistical_analysis(all_results)
    logging.info("📊 Statistical analysis computed (keys: %s)", list(stats.keys()))

    # Minimal human output
    print("✅ OOD smoke test finished – see log:", Path(log_file).as_posix())


# ---------------------------------------------------------------------------
# Pytest integration – treated as ordinary test when imported by pytest
# ---------------------------------------------------------------------------

def test_modular_ood_smoke() -> None:  # pragma: no cover
    """CI regression test: evaluate tiny scale to ensure code paths work."""
    results = _evaluate_all_models(scale="25k", environment="same")
    _basic_sanity_checks(results)

    # Ensure at least two models ran → catches missing checkpoints quickly
    assert len(results) >= 2, "Expected ≥2 models for smoke test"


if __name__ == "__main__":
    main() 