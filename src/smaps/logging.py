"""Structured logging with optional run_id."""

from __future__ import annotations

import logging


def get_logger(name: str, run_id: str | None = None) -> logging.Logger:
    """Return a configured logger. Includes *run_id* in output when provided."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        if run_id:
            fmt = f"%(asctime)s %(levelname)s [{run_id}] %(name)s: %(message)s"
        else:
            fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger
