#!/usr/bin/env python3
"""Thin CLI wrapper for out-of-distribution (OOD) evaluation.

Usage (identical flags to the legacy script)::

    python -m spiramycel.ood_cli --environment same --scale 25k [--no-plots]

The heavy lifting lives in `spiramycel.ood_utils` (model + dataset loading)
and `spiramycel.ood_analysis` (stats, reporting, visuals).  This file
contains < ~150 lines so refactors become painless.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime

from .ood_utils import (
    load_trained_models,
    load_ood_test_set,
    evaluate_model_on_ood,
    filter_scenarios_for_model,
)
from .ood_analysis import (
    perform_statistical_analysis,
    create_visualizations,
    generate_cross_validation_report,
    generate_statistical_report,
)

from .logging_utils import setup_experiment_logging

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contemplative-AI OOD evaluation with statistical analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--environment",
        choices=["same", "switch"],
        default="same",
        help="'same' = stress-level crossover, 'switch' = alien environments",
    )
    parser.add_argument(
        "--scale",
        choices=["25k", "200k", "600k", "6m", "auto"],
        default="auto",
        help="Model scale to test (or 'auto' to pick largest available)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib/seaborn visualizations",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup & banners
    # ------------------------------------------------------------------
    env_desc = {
        "same": "stress-level crossover",
        "switch": "alien environments",
    }[args.environment]

    print("🧪 CONTEMPLATIVE-AI – OOD EVALUATION (modular)")
    print("=" * 60)
    print(f"Environment mode : {args.environment} ({env_desc})")
    print(f"Model scale      : {args.scale}")

    log_file, ts = setup_experiment_logging()
    logging.info("Modular OOD CLI launched")

    # ------------------------------------------------------------------
    # Load models & data
    # ------------------------------------------------------------------
    models = load_trained_models(preferred_scale=args.scale)
    test_scenarios = load_ood_test_set(use_expanded=True, environment=args.environment)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    all_results: dict[str, dict] = {}
    for name, model in models.items():
        if model is None:
            continue
        print(f"🤖 Evaluating {name} …")
        if args.environment == "same":
            scen_subset = filter_scenarios_for_model(name, test_scenarios)
        else:
            scen_subset = test_scenarios
        all_results[name] = evaluate_model_on_ood(model, name, scen_subset, codec=None)  # Glyph codec not essential

    # ------------------------------------------------------------------
    # Analysis & reporting
    # ------------------------------------------------------------------
    stats = perform_statistical_analysis(all_results)
    visuals = [] if args.no_plots else create_visualizations(all_results, stats, ts)

    basic_report = generate_cross_validation_report(all_results, ts)
    stats_report = generate_statistical_report(all_results, stats, visuals, ts)

    print("\n✅ Modular OOD evaluation finished.")
    print(f"📄 Reports saved  : {Path(basic_report).as_posix()} | {Path(stats_report).as_posix()}")
    print(f"📝 Log            : {log_file}")
    if visuals:
        print(f"📊 Visualizations : {len(visuals)} files in results/visualizations/")


if __name__ == "__main__":
    main() 