"""
celery_app.py — Celery application factory.

Configuration highlights
-------------------------
- Broker + result backend: Redis (same URL, separate DB indices supported)
- Serializer: JSON (content-type enforcement)
- Task acknowledgement: late ack + reject-on-worker-lost for reliability
- Worker concurrency sourced from hardware-detected WORKER_COUNT
- Three named queues: evaluate | batch | celery (default)
- Result expiry aligned with RESULT_TTL from config
- Windows compatibility: pool defaults to 'solo'; override with
  CELERY_POOL env var (e.g. 'prefork' on Linux, 'gevent' on Windows+gevent)

Starting workers
-----------------
  # Linux / macOS (prefork — recommended for production)
  celery -A api.workers.celery_app worker \\
         -Q evaluate,batch,celery \\
         -c 4 --loglevel=info

  # Windows (solo pool)
  celery -A api.workers.celery_app worker \\
         -Q evaluate,batch,celery \\
         --pool=solo --loglevel=info
"""
from __future__ import annotations

import os
import platform

from celery import Celery
import api.workers.warmup
from api.config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    QUEUE_BATCH,
    QUEUE_DEFAULT,
    QUEUE_EVALUATE,
    RESULT_TTL,
    WORKER_COUNT,
)

# ---------------------------------------------------------------------------
# Create app
# ---------------------------------------------------------------------------
for _p in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "final_integration")),
]:
    if _p not in os.sys.path:
        os.sys.path.insert(0, _p)

celery_app = Celery(
    "ai_interview",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["api.workers.tasks"],   # auto-discover tasks
)

# ---------------------------------------------------------------------------
# IMPORTANT: Import warmup here to guarantee signal handlers are attached
# BEFORE the worker boots up. `include=["api.workers.tasks"]` does not
# reliably import tasks before the worker_init signal fires!
# ---------------------------------------------------------------------------
import api.workers.warmup  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Core configuration
# ---------------------------------------------------------------------------
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Result storage
    result_expires=RESULT_TTL,
    result_extended=True,          # store task metadata (name, args, kwargs)
    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=False,
    # Routing
    task_default_queue=QUEUE_DEFAULT,
    task_routes={
        "api.workers.tasks.evaluate_unit_task": {"queue": QUEUE_EVALUATE},
        "api.workers.tasks.evaluate_batch_task": {"queue": QUEUE_BATCH},
    },
    # Worker concurrency (can be overridden by -c CLI flag)
    worker_concurrency=WORKER_COUNT,
    # Pool selection — defaults to 'solo' on Windows for compatibility
    worker_pool=os.getenv(
        "CELERY_POOL",
        "solo" if platform.system() == "Windows" else "prefork",
    ),
    # Prefetch: process one task at a time per worker slot (audio tasks are heavy)
    worker_prefetch_multiplier=1,
    # Task time limits
    task_soft_time_limit=int(os.getenv("TASK_SOFT_TIMEOUT", "300")),   # 5 min soft
    task_time_limit=int(os.getenv("TASK_HARD_TIMEOUT", "360")),        # 6 min hard kill
    # Timezone
    timezone="UTC",
    enable_utc=True,
)

# ---------------------------------------------------------------------------
# Named queue definitions (ensures queues are created on startup)
# ---------------------------------------------------------------------------
celery_app.conf.task_queues = None  # rely on auto-create from task_routes

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
__all__ = ["celery_app"]
