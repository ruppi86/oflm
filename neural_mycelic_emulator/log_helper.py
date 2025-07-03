from __future__ import annotations

"""Simple helper util to keep all Neural-Mycelic Emulator logs in the
`neural_mycelic_emulator/logs/` folder regardless of the script that
produces them.

Importable via::

    from neural_mycelic_emulator.log_helper import init_file_logger

It sets up the *root* logger with both a FileHandler (to the shared logs
folder) and a StreamHandler (stdout) so legacy `print()` lines remain
visible while structured logs are captured.
"""

from pathlib import Path
import logging
import datetime
import sys
import os

__all__ = ["init_file_logger", "get_logs_dir"]


def get_logs_dir() -> Path:
    """Return (and create) the shared logs directory."""
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def init_file_logger(prefix: str, *, level: int = logging.INFO) -> Path:
    """Configure the root logger to also write to *prefix*_<timestamp>.log.

    Parameters
    ----------
    prefix : str
        Human-readable prefix, e.g. "train_cordyceps_small".
    level : int, default ``logging.INFO``
        Logging level.
    Returns
    -------
    Path
        Path to the created log file.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = get_logs_dir() / f"{prefix}_{ts}.log"

    # Avoid re-adding handlers if init_file_logger is called twice
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(logfile) for h in root.handlers):
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(logfile)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        # Windows console encoding fix – replace unencodable chars
        sh.stream = getattr(sh, 'stream', None) or __import__('sys').stdout
        try:
            sh.stream.reconfigure(errors='replace')  # py ≥3.7
        except Exception:
            pass
        root.setLevel(level)
        root.addHandler(fh)
        root.addHandler(sh)

        # ------------------------------------------------------
        # Optional: Redirect print() so console & file see the same text.
        # Enable by setting MYCELIC_CAPTURE_STDOUT=1 to avoid overwhelming logs.
        # ------------------------------------------------------
        if os.getenv("MYCELIC_CAPTURE_STDOUT", "0") == "1":
            class _StreamToLogger:
                def __init__(self, logger: logging.Logger, level: int):
                    self.logger = logger
                    self.level = level

                def write(self, message: str):
                    message = message.rstrip()
                    if message:
                        self.logger.log(self.level, message)

                def flush(self):  # pragma: no cover
                    pass

            sys.stdout = _StreamToLogger(root, logging.INFO)
            sys.stderr = _StreamToLogger(root, logging.ERROR)

    return logfile 