"""
logger.py — tone sub-module logger shim.

Uses stdlib propagation so records flow to the root logger configured
by api.logging_config.setup_logging() (with full rotation support).
Does NOT add a local handler — that would duplicate records and bypass rotation.
"""
import logging

# Propagate to root — root has the rotating QueueHandler after setup_logging().
logger = logging.getLogger("tone_prosody")
logger.propagate = True
# Fallback: if root has no handlers yet (standalone / test), Python's
# lastResort handler (stderr) will catch the record — no config needed.
