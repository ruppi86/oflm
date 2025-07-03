#!/usr/bin/env python
"""Dataset statistics helper for Neural-Mycelic Emulator.

Usage (from project root):
    python -m neural_mycelic_emulator.tools.dataset_stats [--cfg path]

Outputs a markdown-style table with, for every model tag defined in the YAML,
* dataset path (resolved)
* #channels detected
* #glyphs in whole sequence
* training window length (context window)
* #training windows (len(seq) - window)
* validation glyphs (10 % tail of sequence)

If multiple model tags share the same dataset the costly TSV parsing is done
only once.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import sys
import yaml

from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences


CacheType = Dict[Path, Tuple[int, int]]  # path -> (glyph_len, n_channels)


def analyse_dataset(tsv_path: Path, cache: CacheType) -> Tuple[int, int]:
    """Return (glyph_len, n_channels) for given TSV, using in-memory cache."""
    if tsv_path in cache:
        return cache[tsv_path]
    seq, n_channels = tsv_to_glyph_sequences(tsv_path, return_channels=True)
    cache[tsv_path] = (len(seq), n_channels)
    return cache[tsv_path]


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise dataset sizes per model tag")
    p.add_argument("--cfg", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "emulator_parameters.yml", help="Path to emulator_parameters.yml")
    args = p.parse_args()

    cfg_all = yaml.safe_load(args.cfg.read_text())
    models = cfg_all["models"]

    cache: CacheType = {}

    header = (
        "| Tag | Dataset | Channels | Glyphs | Window | Train windows | Val glyphs |\n"
        "|-----|---------|----------|--------|--------|---------------|------------|"
    )
    rows = []

    for tag, entry in models.items():
        tsv_path = Path(entry["dataset"]).expanduser()
        if tsv_path.is_dir():
            txt_files = list(tsv_path.glob("*.txt"))
            if not txt_files:
                print(f"⚠️  No .txt files in {tsv_path}", file=sys.stderr)
                continue
            tsv_path = txt_files[0]
        glyphs, n_channels = analyse_dataset(tsv_path, cache)
        window = entry.get("window", 64)
        train_glyphs = int(glyphs * 0.9)
        val_glyphs = glyphs - train_glyphs
        train_windows = max(0, train_glyphs - window)
        rows.append(
            f"| {tag} | {tsv_path.name} | {n_channels} | {glyphs} | {window} | {train_windows} | {val_glyphs} |"
        )

    print(header)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main() 