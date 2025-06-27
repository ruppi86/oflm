#!/usr/bin/env python
"""
collect_sources.py  – bundle every *.py in a project into one TXT file

Usage
-----
    python collect_sources.py               # walks from current dir
    python collect_sources.py path/to/repo  # walks that dir instead
"""

from pathlib import Path
import sys

EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", ".mypy_cache"}
OUT_FILENAME = "all_python_sources.txt"


def gather_python_files(root: Path) -> list[Path]:
    """Return every *.py path under *root*, depth-first, skipping EXCLUDE_DIRS."""
    return sorted(
        p for p in root.rglob("*.py")
        if not any(part in EXCLUDE_DIRS for part in p.parts)
    )


def bundle_files(paths: list[Path], out_path: Path) -> None:
    lines: list[str] = []
    for p in paths:
        rel = p.relative_to(out_path.parent)
        lines.append(f"\n# ===== {rel} =====\n")
        lines.append(p.read_text(encoding="utf-8", errors="replace"))
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(paths)} files, {out_path.stat().st_size/1024:.1f} KB)")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".").resolve()
    out_path = root / OUT_FILENAME
    paths = gather_python_files(root)
    bundle_files(paths, out_path)


if __name__ == "__main__":
    main()
