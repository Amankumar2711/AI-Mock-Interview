"""
logging_config.py — Unified, time-based rotational logging for the full
AI-interview stack (FastAPI + Celery workers + all sub-pipelines).

Design
------
One call to setup_logging() at startup configures the ROOT logger with:

  • RotatingFileHandler per subsystem — rotates at max_bytes (default 20 MB),
    keeps LOG_RETENTION_DAYS backup copies.  Rotated files carry a date stamp
    in their name via a custom namer:
        api.log  →  api.log.2026-06-18.1  →  api.log.2026-06-18.2  …

  • Four separate log files:
      logs/api/api.log      — FastAPI / uvicorn / HTTP layer
      logs/api/worker.log   — Celery tasks / pipeline orchestration
      logs/api/model.log    — Heavy model loading (Whisper, SpeechBrain …)
      logs/api/combined.log — Everything (catch-all, always written)

  • QueueHandler front-end — all real handlers live in a background daemon
    thread so log calls never block the asyncio event loop or Celery workers.

  • JSON formatter in production, human-readable colored formatter in dev.

  • get_logger(name) — returns a _KwargAdapter that propagates to root so
    every module participates in rotation automatically without any handler
    of its own.

  • shutdown_logging() — call from lifespan / Celery worker shutdown to
    flush the queue cleanly before process exit.

Environment variables
---------------------
  LOG_RETENTION_DAYS   Backup copies to retain  (default: 30)
  LOG_MAX_BYTES        Per-file bytes cap        (default: 20 MB)
  LOG_LEVEL            Root log level            (default: INFO)
  APP_ENV              dev | prod                (default: dev)

Idempotent — safe to call from FastAPI lifespan AND the Celery
worker_ready signal in the same process.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Constants — overridable via env vars
# ---------------------------------------------------------------------------
_DEFAULT_RETENTION_DAYS: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))
_DEFAULT_MAX_BYTES: int      = int(os.getenv("LOG_MAX_BYTES", str(20 * 1_048_576)))  # 20 MB
_DEFAULT_LOG_LEVEL: str      = os.getenv("LOG_LEVEL", "INFO").upper()

# Subsystem routing prefixes
_API_PREFIXES: tuple[str, ...] = (
    "api", "uvicorn", "fastapi", "starlette", "httpx",
)
_WORKER_PREFIXES: tuple[str, ...] = (
    "celery", "api.workers", "core_integration", "pipeline",
    "config_integration", "models_integration", "final_integration",
)
_MODEL_PREFIXES: tuple[str, ...] = (
    "warmup", "api.workers.warmup",
    "tone", "tone_prosody", "AI_Analyzer",
    "STT", "Pronunciation", "confidence",
    "technical_eval", "ModelRegistry",
)

# Noisy third-party loggers clamped to WARNING
_NOISY_LOGGERS: tuple[str, ...] = (
    "urllib3", "httpcore", "h11", "asyncio",
    "filelock", "huggingface_hub", "transformers",
    "sentence_transformers",
)

# Module-level guards
_setup_done: bool = False
_setup_lock  = threading.Lock()
_queue_listener: Optional[logging.handlers.QueueListener] = None


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """Emit every record as a single UTF-8 JSON line (production format)."""

    _EXTRA_KEYS = (
        "session_id", "task_id", "batch_id", "unit_id",
        "latency_ms", "status", "task_type",
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict = {
            "ts":     datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":  record.levelname,
            "logger": record.name,
            "pid":    record.process,
            "msg":    record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in self._EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False)


class _DevFormatter(logging.Formatter):
    """Human-readable colored formatter for development console output."""

    _COLORS = {
        "DEBUG":    "\033[36m",    # cyan
        "INFO":     "\033[32m",    # green
        "WARNING":  "\033[33m",    # yellow
        "ERROR":    "\033[31m",    # red
        "CRITICAL": "\033[35;1m",  # bold magenta
    }
    _RESET = "\033[0m"
    _DIM   = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color = self._COLORS.get(record.levelname, "")
        ts    = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        level = f"{color}{record.levelname:<8}{self._RESET}"
        name  = f"{self._DIM}{record.name}{self._RESET}"
        pid   = f"{self._DIM}[pid={record.process}]{self._RESET}"
        line  = f"[{ts}] {level} {pid} {name}: {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ---------------------------------------------------------------------------
# Prefix-based filter
# ---------------------------------------------------------------------------
class _PrefixFilter(logging.Filter):
    """Allow only records whose logger name starts with one of the prefixes."""

    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return any(record.name.startswith(p) for p in self.prefixes)


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------
def _make_rotating_handler(
    filepath: str,
    json_fmt: _JsonFormatter,
    level: int,
    prefix_filter: Optional[_PrefixFilter] = None,
    retention_days: int = _DEFAULT_RETENTION_DAYS,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> logging.handlers.RotatingFileHandler:
    """
    Build a RotatingFileHandler whose rotated files carry a date stamp.

    Rotated filename pattern:  <name>.log.YYYY-MM-DD.<n>
    e.g.  api.log.2026-06-18.1,  api.log.2026-06-17.2

    delay=True — file is not created/opened until the first record arrives
                 (avoids empty log files on startup).
    """
    handler = logging.handlers.RotatingFileHandler(
        filename    = filepath,
        maxBytes    = max_bytes,
        backupCount = retention_days,
        encoding    = "utf-8",
        delay       = True,
    )

    # Custom namer: inject today's UTC date before the numeric backup index
    def _namer(default_name: str) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # default_name looks like  /logs/api/api.log.1
        if "." in os.path.basename(default_name).split("log")[-1]:
            base, idx = default_name.rsplit(".", 1)
            return f"{base}.{today}.{idx}"
        return f"{default_name}.{today}"

    handler.namer = _namer
    handler.setFormatter(json_fmt)
    handler.setLevel(level)
    if prefix_filter is not None:
        handler.addFilter(prefix_filter)
    return handler


# ---------------------------------------------------------------------------
# Public: setup_logging
# ---------------------------------------------------------------------------
def setup_logging(
    log_dir: str,
    app_env: str = "dev",
    log_level: Optional[int] = None,
    api_log_level: int = logging.INFO,
    worker_log_level: int = logging.INFO,
    model_log_level: int = logging.INFO,
    retention_days: int = _DEFAULT_RETENTION_DAYS,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    force: bool = False,
) -> None:
    """
    Configure the root logger with rotating file handlers and an async-safe
    QueueHandler front-end.

    Idempotent — subsequent calls are no-ops unless force=True.

    Args:
        log_dir:          Directory for log files (created if missing).
        app_env:          'dev' -> colored console output; else JSON.
        log_level:        Root logger level (overrides LOG_LEVEL env var).
        api_log_level:    Level for api.log handler.
        worker_log_level: Level for worker.log handler.
        model_log_level:  Level for model.log handler.
        retention_days:   Number of rotated backup files to keep.
        max_bytes:        File size in bytes that triggers rotation.
        force:            Tear down existing handlers and reconfigure.
    """
    global _setup_done, _queue_listener

    if _setup_done and not force:
        return  # fast-path — no lock needed for read

    with _setup_lock:
        if _setup_done and not force:
            return  # another thread won the race

        _root_level = log_level or getattr(logging, _DEFAULT_LOG_LEVEL, logging.INFO)
        os.makedirs(log_dir, exist_ok=True)

        root = logging.getLogger()

        # Tear down if forced
        if root.handlers:
            if not force:
                _setup_done = True
                return
            if _queue_listener is not None:
                _queue_listener.stop()
                _queue_listener = None
            for h in root.handlers[:]:
                root.removeHandler(h)
                h.close()

        root.setLevel(logging.DEBUG)   # root captures all; handlers filter

        json_fmt    = _JsonFormatter()
        console_fmt = _DevFormatter() if app_env == "dev" else json_fmt

        # ── Subsystem file handlers ─────────────────────────────────────────
        api_handler = _make_rotating_handler(
            filepath       = os.path.join(log_dir, "api.log"),
            json_fmt       = json_fmt,
            level          = api_log_level,
            prefix_filter  = _PrefixFilter(_API_PREFIXES),
            retention_days = retention_days,
            max_bytes      = max_bytes,
        )
        worker_handler = _make_rotating_handler(
            filepath       = os.path.join(log_dir, "worker.log"),
            json_fmt       = json_fmt,
            level          = worker_log_level,
            prefix_filter  = _PrefixFilter(_WORKER_PREFIXES),
            retention_days = retention_days,
            max_bytes      = max_bytes,
        )
        model_handler = _make_rotating_handler(
            filepath       = os.path.join(log_dir, "model.log"),
            json_fmt       = json_fmt,
            level          = model_log_level,
            prefix_filter  = _PrefixFilter(_MODEL_PREFIXES),
            retention_days = retention_days,
            max_bytes      = max_bytes,
        )
        combined_handler = _make_rotating_handler(
            filepath       = os.path.join(log_dir, "combined.log"),
            json_fmt       = json_fmt,
            level          = logging.DEBUG,
            prefix_filter  = None,   # catch-all — no prefix filter
            retention_days = retention_days,
            max_bytes      = max_bytes,
        )

        # ── Console handler ─────────────────────────────────────────────────
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_fmt)
        console_handler.setLevel(logging.DEBUG if app_env == "dev" else logging.INFO)

        # ── QueueHandler — async-safe, non-blocking ─────────────────────────
        # Records are enqueued instantly (no disk I/O on caller thread).
        # QueueListener drains the queue in a separate daemon thread so
        # neither the asyncio event loop nor Celery workers block on I/O.
        log_queue: queue.Queue = queue.Queue(maxsize=-1)  # unbounded
        queue_handler = logging.handlers.QueueHandler(log_queue)
        queue_handler.setLevel(logging.DEBUG)

        _queue_listener = logging.handlers.QueueListener(
            log_queue,
            api_handler,
            worker_handler,
            model_handler,
            combined_handler,
            console_handler,
            respect_handler_level=True,
        )
        _queue_listener.start()

        root.addHandler(queue_handler)

        # ── Suppress noisy third-party loggers ──────────────────────────────
        for noisy in _NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _setup_done = True

        logging.getLogger(__name__).info(
            "[logging] Rotational logging active — dir=%s env=%s "
            "max_mb=%.0f retention=%d files",
            log_dir, app_env, max_bytes / 1_048_576, retention_days,
        )


# ---------------------------------------------------------------------------
# Public: get_logger
# ---------------------------------------------------------------------------
class _KwargAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that accepts arbitrary keyword arguments in log calls
    and folds them into the message string, e.g.::

        logger.info("started", session_id="abc", latency_ms=42)
        # emits: "started | session_id='abc' | latency_ms=42"
    """
    _STD_KWARGS = frozenset({"exc_info", "stack_info", "stacklevel", "extra"})

    def process(self, msg: str, kwargs: dict) -> tuple:
        custom = {k: v for k, v in kwargs.items() if k not in self._STD_KWARGS}
        for k in custom:
            kwargs.pop(k)
        if custom:
            extras = " | ".join(f"{k}={v!r}" for k, v in custom.items())
            msg = f"{msg} | {extras}"
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """
    Return a _KwargAdapter for *name* that propagates to the root logger.

    This is the single entry point all sub-modules should use instead of
    calling ``logging.getLogger()`` directly.  Because ``propagate=True``
    (the default), every record flows to root and is captured by the
    rotating handlers configured in setup_logging() — no local handler
    setup is required.

    Usage::

        from api.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("pipeline started", session_id="abc")
    """
    base = logging.getLogger(name)
    base.propagate = True   # ensure records reach the rotating handlers
    return _KwargAdapter(base, {})


# ---------------------------------------------------------------------------
# Public: shutdown_logging
# ---------------------------------------------------------------------------
def shutdown_logging() -> None:
    """
    Flush and stop the background QueueListener.

    Call from the FastAPI shutdown lifespan and Celery worker shutdown
    signal to ensure all queued log records are written before process exit.
    """
    global _queue_listener, _setup_done
    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None
    _setup_done = False
