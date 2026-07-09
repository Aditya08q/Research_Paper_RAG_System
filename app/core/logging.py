"""
Application-wide logging configuration.

The spec calls out specific events to log (upload, chunking, embedding,
retrieval, LLM latency, errors). Rather than each service configuring its
own logger differently, we set up one consistent format here and have every
module call `get_logger(__name__)`.
"""

import logging
import sys


_CONFIGURED = False


def _configure_root_logger() -> None:
    """Configure the root logger exactly once for the whole application."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Chunk created: %s chars", len(chunk))
    """
    _configure_root_logger()
    return logging.getLogger(name)
