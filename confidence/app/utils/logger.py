"""
logger.py — confidence sub-module logger shim.

Uses stdlib propagation so records flow to the root logger configured
by api.logging_config.setup_logging() (with full rotation support).
Does NOT add a local RotatingFileHandler — that would bypass the
central QueueHandler and write to a separate uncoordinated file.
"""
import logging


def get_logger() -> logging.Logger:
    """Return the confidence module logger (propagates to root)."""
    log = logging.getLogger("confidence")
    log.propagate = True
    return log