"""
logger.py — final_integration sub-module logger.

A thin wrapper around stdlib logging.Logger that:
  - Accepts arbitrary keyword arguments in log calls (like structlog)
    by folding them into the message string, avoiding TypeError.
  - Propagates to root so all records are captured by the central
    rotating handlers set up in api.logging_config.setup_logging().
  - Does NOT add a local handler — propagation handles everything.
"""
from __future__ import annotations

import logging
import sys


class _KwargAdapter(logging.LoggerAdapter):
    """Lets call sites pass key=value kwargs that aren't stdlib logging
    kwargs (exc_info, stack_info, stacklevel, extra) and have them
    appended to the message instead of raising TypeError."""

    _LOGGING_KWARGS = {"exc_info", "stack_info", "stacklevel", "extra"}

    def process(self, msg, kwargs):
        custom = {k: v for k, v in kwargs.items() if k not in self._LOGGING_KWARGS}
        for k in custom:
            kwargs.pop(k)
        if custom:
            extras = " | ".join(f"{k}={v!r}" for k, v in custom.items())
            msg = f"{msg} | {extras}"
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a _KwargAdapter for *name* that propagates to root logger."""
    log = logging.getLogger(name)
    log.propagate = True   # records flow to root's rotating QueueHandler
    return _KwargAdapter(log, {})
