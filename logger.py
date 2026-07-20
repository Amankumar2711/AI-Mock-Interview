"""
logger.py — project-root logger shim.

Provides a module-level `logger` used by:
  - technical_eval/core_tech/semantic_eval.py  (from logger import logger)
  - Any other modules that do `from logger import logger`

Records propagate to root so they are captured by api.logging_config's
rotating handlers when running inside the full API stack.  When running
standalone (e.g. tests), Python's lastResort stderr handler catches them.
"""
import logging

logger = logging.getLogger("technical_eval")
logger.propagate = True
