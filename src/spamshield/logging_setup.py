"""One place to configure logging for the CLI and the web app."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Send log records to stderr. Safe to call more than once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
