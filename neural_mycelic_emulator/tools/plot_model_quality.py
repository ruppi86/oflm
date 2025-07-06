#!/usr/bin/env python3
"""Plot glyph distribution error (L1-diff) for all Neural-Mycelic Emulator models.

Usage
-----
python -m neural_mycelic_emulator.tools.plot_model_quality \
       --results-file neural_mycelic_emulator/results/results_2025_07_02.md \
       --out glyph_l1_by_model.png

The script parses the Markdown tables, extracts *Glyph L1-diff* and
plots them as a horizontal bar chart, grouped by species.  Colours show
context length (ctx64 / ctx128 / ctx192, etc.).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import textwrap

TABLE_RE = re.compile(r"^\| (?P<row>.+) \|$")
SEP_LINE_RE = re.compile(r"^-+\|")


def _parse_tables(md_lines: List[str]) -> pd.DataFrame:
    """Return DataFrame with columns: species, model, params, silence, ks_p, d, l1"""
    species = None
    columns = [
        "Model tag",
        "Params (≈)",
        "Silence ratio",
        "ISI KS p-value",
        "Cohen's d (ISI)",
        "Glyph L1-diff",
    ]
    rows: List[Dict[str, str]] = []

    for line in md_lines:
        line = line.rstrip()
        if line.startswith("# Neural Mycelic Emulator"):
            # Extract species name inside parentheses
            m = re.search(r"– (.+?) \(", line)
            if m:
                species = m.group(1).strip()
            continue
        if not species:
            continue
        if SEP_LINE_RE.match(line):
            continue  # separator row
        m = TABLE_RE.match(line)
        if not m:
            continue
        parts = [p.strip() for p in m.group("row").split("|")]
        if parts[0] == "Model tag" or len(parts) < 6:
            continue  # header or malformed
        entry = {col: val for col, val in zip(columns, parts)}
        entry["species"] = species
        rows.append(entry)
    if not rows:
        raise ValueError("No table rows parsed – check markdown format")

    df = pd.DataFrame(rows)
    # Clean numeric columns
    for col in ["Silence ratio", "ISI KS p-value", "Cohen's d (ISI)", "Glyph L1-diff"]:
        df[col] = pd.to_numeric(df[col].str.replace("%", ""), errors="coerce")
       
    return df


def plot_glyph_l1(df: pd.DataFrame, outfile: Path):
    # 1) Sort: L1-diff DESCENDING
    df_sorted = df.sort_values("Glyph L1-diff", ascending=False)

    # 2) draw bars
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_sorted) * 0.3)))
    colors = ["#1f77b4" if "ctx128" in t else
              "#ff7f0e" if "ctx192" in t else
              "#2ca02c"
              for t in df_sorted["Model tag"]]
    ax.barh(df_sorted["Model tag"], df_sorted["Glyph L1-diff"], color=colors)

    ax.set_xlabel("Glyph distribution error (L1 distance)")
    ax.set_title("Neural-Mycelic Emulator – Glyph L1-diff by model (lower is better)")
    # 3) DO NOT invert axis – highest L1 already at the top

    # annotate bars
    for idx, val in enumerate(df_sorted["Glyph L1-diff"]):
        ax.text(val + 0.005, idx, f"{val:.3f}", va="center")

    # no species separators – flat list

    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"✅ Saved figure to {outfile}")


def main():
    ap = argparse.ArgumentParser(description="Plot glyph L1-diff across models")
    ap.add_argument("--results-file", required=True, type=Path, help="Markdown results file path")
    ap.add_argument("--out", default="glyph_l1_by_model.png", type=Path, help="Output PNG path")
    args = ap.parse_args()

    lines = args.results_file.read_text(encoding="utf-8").splitlines()
    df = _parse_tables(lines)
    plot_glyph_l1(df, args.out)


if __name__ == "__main__":
    main() 