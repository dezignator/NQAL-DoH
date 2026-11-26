#!/usr/bin/env python3
"""utils/logging_utils.py — consistent structured logging."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = "artifacts/logs",
    level: int = logging.INFO,
    name: Optional[str] = None,
) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name or "nqal")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        os.path.join(log_dir, f"{(name or 'nqal')}.log")
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
