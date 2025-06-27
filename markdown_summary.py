#!/usr/bin/env python
"""
markdown_summary.py  – bundle every *.md file in the repository into one TXT file

Usage
-----
    python markdown_summary.py  # bundles all .md files in this folder and subfolders
"""

from pathlib import Path
from datetime import datetime

EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", ".mypy_cache"}
OUT_FILENAME_BASE = "all_markdown_files"


def gather_markdown_files(root: Path) -> list[Path]:
    """Return every *.md path under *root*, depth-first, skipping EXCLUDE_DIRS."""
    return sorted(
        p for p in root.rglob("*.md")
        if not any(part in EXCLUDE_DIRS for part in p.parts)
    )


def bundle_files(paths: list[Path], out_path: Path) -> None:
    lines: list[str] = []
    for p in paths:
        rel = p.relative_to(out_path.parent)
        lines.append(f"\n# ===== {rel} =====\n")
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            lines.append(content)
        except Exception as e:
            lines.append(f"# Error reading file: {e}\n")
    
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(paths)} files, {out_path.stat().st_size/1024:.1f} KB)")


def main() -> None:
    # Use the directory where this script is located (root folder)
    script_dir = Path(__file__).parent.resolve()
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{OUT_FILENAME_BASE}_{timestamp}.txt"
    out_path = script_dir / out_filename
    
    paths = gather_markdown_files(script_dir)
    print(f"Found {len(paths)} markdown files to bundle...")
    bundle_files(paths, out_path)


if __name__ == "__main__":
    main() 