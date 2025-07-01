import logging
from pathlib import Path
from datetime import datetime

__all__ = [
    "setup_experiment_logging",
    "create_condition_logger",
]

def setup_experiment_logging() -> tuple[str, str]:
    """Configure root logger and create main experiment logfile.

    Returns
    -------
    (main_log_path, timestamp_str)
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = logs_dir / f"experiment_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(main_log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return str(main_log_file), timestamp


def create_condition_logger(name: str, timestamp: str) -> tuple[logging.Logger, str]:
    """Return a child logger that writes to its own file and doesn't propagate."""
    logs_dir = Path("logs")
    log_path = logs_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate lines in root logger

    # Clean handlers if func called twice for same name
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger, str(log_path) 