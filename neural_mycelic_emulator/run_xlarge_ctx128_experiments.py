#!/usr/bin/env python3
"""Run large_ctx128 experiments sequentially.

Usage:
    python -m neural_mycelic_emulator.run_large_ctx128_experiments [--skip-trained]
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

BASE = Path(__file__).parent
TAGS = [
    "cordyceps_xlarge_ctx128",
    "enoki_xlarge_ctx128",
    "ghost_xlarge_ctx128",
    "schizo_xlarge_ctx128",
]
PY = sys.executable

def call(cmd: list[str]):
    print("\n»", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-trained", action="store_true")
    args = ap.parse_args()

    for tag in TAGS:
        ckpt = BASE / "models" / tag / f"{tag}_best.pt"
        if ckpt.exists() and args.skip_trained:
            print(f"✅ {tag} checkpoint exists – skipping training")
        else:
            call([PY, "-m", "neural_mycelic_emulator.models.trainer", tag])
        call([PY, "-m", "neural_mycelic_emulator.models.evaluate_perplexity", tag, str(ckpt)])
        call([PY, "-m", "neural_mycelic_emulator.models.compare_stats", tag, str(ckpt)])
    print("\n🎉 xlarge_ctx128 batch finished")

if __name__ == "__main__":
    main() 