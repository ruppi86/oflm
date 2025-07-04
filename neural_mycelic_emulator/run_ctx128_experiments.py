#!/usr/bin/env python3
"""Run all ctx128 experiments sequentially.

Usage:
    python -m neural_mycelic_emulator.run_ctx128_experiments [--skip-trained]

For each model tag listed in TAGS it will:
1. Train (unless a *_best.pt checkpoint already exists AND --skip-trained).
2. Evaluate perplexity.
3. Compare real vs synthetic stats.

All stdout/stderr from sub-processes are streamed to console, so you can
monitor progress in real time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
TAGS = [
    "cordyceps_medium_ctx128",
    "enoki_medium_ctx128",
    "ghost_medium_ctx128",
]

PY = sys.executable  # current python


def call(cmd: list[str]):
    print("\n»", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(res.returncode)


def main() -> None:
    p = argparse.ArgumentParser("Run ctx128 experiment batch")
    p.add_argument("--skip-trained", action="store_true", help="Skip training when checkpoint already exists")
    args = p.parse_args()

    for tag in TAGS:
        ckpt = BASE / "models" / tag / f"{tag}_best.pt"
        if ckpt.exists() and args.skip_trained:
            print(f"✅ Checkpoint exists for {tag}, skipping training")
        else:
            call([PY, "-m", "neural_mycelic_emulator.models.trainer", tag])

        # Perplexity
        call([PY, "-m", "neural_mycelic_emulator.models.evaluate_perplexity", tag, str(ckpt)])

        # Compare stats (tsv optional – default from yaml)
        call([PY, "-m", "neural_mycelic_emulator.models.compare_stats", tag, str(ckpt)])

    print("\n🎉 All ctx128 experiments completed")


if __name__ == "__main__":
    main() 