"""Logging configuration for the deepfake rPPG pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        log_file: Optional path to a log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
