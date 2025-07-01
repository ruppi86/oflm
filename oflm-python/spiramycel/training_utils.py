from pathlib import Path
import os
import re
from datetime import datetime
from typing import List, Tuple
import random

__all__ = [
    "determine_model_scale_and_folders",
    "discover_training_data",
    "get_file_size_kb",
    "set_deterministic",
]

def determine_model_scale_and_folders(model, paradigm: str) -> Tuple[str, str, str]:
    """Determine model scale based on parameter count and return
    (scale_name, model_dir, scale_suffix).
    Keeps the exact threshold logic previously duplicated in both
    *abstract_training.py* and *ecological_training.py* so behaviour
    is unchanged.
    """
    param_count = model.count_parameters()

    if param_count < 50_000:       # < 50 K parameters
        scale_name = scale_suffix = "25k"
    elif param_count < 300_000:    # 50 K – 300 K
        scale_name = scale_suffix = "200k"
    elif param_count < 2_000_000:  # 300 K – 2 M
        scale_name = scale_suffix = "600k"
    else:                          # ≥ 2 M
        scale_name = scale_suffix = "6m"

    model_dir = f"{paradigm}_models_{scale_suffix}"
    print(f"🏷️  Auto-detected {scale_name} scale model ({param_count:,} parameters)")
    print(f"📁 Using scale-specific directory: {model_dir}/")
    return scale_name, model_dir, scale_suffix


def discover_training_data(paradigm: str, data_dir: str = "training_scenarios") -> List[Path]:
    """Return list of *.jsonl* files for given paradigm, ordered by recency.
    Merges the date-parsing logic that existed in both training scripts.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    files = list(data_path.glob(f"{paradigm}_*.jsonl"))
    if not files:
        return []

    date_patterns = [
        r"(\d{8}_\d{6})",   # YYYYMMDD_HHMMSS
        r"(\d{8})",         # YYYYMMDD
        r"(\d{4}_\d{2}_\d{2})",  # YYYY_MM_DD
        r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
    ]

    dated_files, undated_files = [], []
    for file_path in files:
        filename = file_path.name
        tagged = False
        for pat in date_patterns:
            m = re.search(pat, filename)
            if m:
                date_str = m.group(1).replace("_", "").replace("-", "")
                if len(date_str) == 8:      # YYYYMMDD
                    date_str += "000000"
                # Validate date component to avoid sorting junk
                try:
                    datetime.strptime(date_str[:8], "%Y%m%d")
                    dated_files.append((date_str, file_path))
                    tagged = True
                    break
                except ValueError:
                    continue
        if not tagged:
            undated_files.append(file_path)

    # Sort dated newest-first, undated largest-first
    dated_files.sort(key=lambda t: t[0], reverse=True)
    undated_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return [p for _, p in dated_files] + undated_files


def get_file_size_kb(file_path: str) -> str:
    """Return human-friendly file size in KB (rounded) or 'Unknown'."""
    try:
        return f"{Path(file_path).stat().st_size // 1024}KB"
    except Exception:
        return "Unknown"

def set_deterministic(seed: int = 42):
    """Set python, NumPy and PyTorch (if available) seeds for full determinism."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        # Deterministic CuDNN (may hurt perf but ensures repeatability)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass 