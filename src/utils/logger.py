"""Logging setup shared by all modules (console + ``logs/pipeline.log``)."""
from __future__ import annotations

import logging
import sys

from src.utils.config import ROOT

LOG_FILE = ROOT / "logs" / "pipeline.log"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger that writes to stdout and to logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:                       # already configured (avoid duplicate lines)
        return logger
    logger.setLevel(logging.INFO)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False                  # don't double-print through the root logger
    return logger
