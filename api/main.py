"""
main.py — FastAPI application entry point for the AI-interview production API.

Startup sequence
----------------
1. Logging configured (rotational JSON files + console)
2. Redis connection pool initialised
3. Redis connectivity verified
4. Routers mounted with their prefixes
5. CORS / global exception handlers attached
6. Background upload cleanup task started
7. Prometheus metrics loop started (queue depth, workers, active tasks)

Shutdown sequence
-----------------
1. Prometheus metrics loop cancelled
2. Upload cleanup task cancelled
3. Redis pool drained

Running
-------
  # Development
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

  # Production (Gunicorn with Uvicorn workers)
  gunicorn api.main:app -k uvicorn.workers.UvicornWorker \\
      -w 2 --bind 0.0.0.0:8000

Environment variables (see .env.example for full list)
-------------------------------------------------------
  APP_ENV                    dev | prod
  REDIS_URL                  redis://localhost:6379/0
  APP_HOST                   0.0.0.0
  APP_PORT                   8000
  CORS_ORIGINS               * (comma-separated list for production)
  UPLOAD_MAX_AGE_S           Max age of an upload file before it is deleted (default: 3600 = 1 hr)
  UPLOAD_CLEANUP_INTERVAL_S  How often the cleanup sweep runs (default: 300 = 5 min)
  METRICS_POLL_INTERVAL_S    How often the Prometheus metrics loop runs (default: 15 s)
  WARN_QUEUE_DEPTH           Log WARNING when any queue exceeds this depth (default: 50)
  WARN_ACTIVE_TASKS          Log WARNING when active tasks across all workers exceed this (default: 20)
  WARN_ZERO_WORKERS          Log WARNING when no Celery workers are detected (default: true)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Gauge, Info
from prometheus_fastapi_instrumentator import Instrumentator

from api.config import APP_ENV, CORS_ORIGINS, LOG_DIR, REDIS_URL, UPLOAD_DIR
from api.config import QUEUE_EVALUATE, QUEUE_BATCH, QUEUE_DEFAULT
from api.dependencies import close_redis_pool, init_redis_pool
from api.logging_config import setup_logging, shutdown_logging

# Configure logging before anything else
setup_logging(log_dir=LOG_DIR, app_env=APP_ENV)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus — Custom metrics definitions
# ---------------------------------------------------------------------------
# Queue depth gauges — one label value per Celery queue
_QUEUE_DEPTH = Gauge(
    "ai_interview_queue_depth",
    "Number of pending tasks in each Celery queue (read from Redis LLEN)",
    ["queue"],
)

# Active tasks across all workers (from Celery inspect)
_ACTIVE_TASKS = Gauge(
    "ai_interview_active_tasks_total",
    "Total number of tasks currently being executed across all Celery workers",
)

# Number of active Celery worker processes
_ACTIVE_WORKERS = Gauge(
    "ai_interview_active_workers",
    "Number of active Celery worker processes",
)

# Static build-info label (useful for filtering in Grafana)
_BUILD_INFO = Info(
    "ai_interview_build",
    "Static build / environment metadata",
)
_BUILD_INFO.info({"app_env": APP_ENV, "version": "1.0.0"})

# ---------------------------------------------------------------------------
# Prometheus — Threshold warning configuration
# ---------------------------------------------------------------------------
# All thresholds are env-var overridable so they can be tuned per environment.

# Warn when any single queue has more pending tasks than this.
WARN_QUEUE_DEPTH: int = int(os.getenv("WARN_QUEUE_DEPTH", "50"))
# Warn when the total active-task count across all workers exceeds this.
WARN_ACTIVE_TASKS: int = int(os.getenv("WARN_ACTIVE_TASKS", "20"))
# Warn when the metrics loop detects zero active workers.
WARN_ZERO_WORKERS: bool = os.getenv("WARN_ZERO_WORKERS", "true").lower() == "true"
# How often (seconds) the metrics polling loop runs.
METRICS_POLL_INTERVAL_S: float = float(os.getenv("METRICS_POLL_INTERVAL_S", "15"))

# ---------------------------------------------------------------------------
# Upload cleanup configuration
# ---------------------------------------------------------------------------
# Files older than this (in seconds) are deleted by the background sweep.
UPLOAD_MAX_AGE_S: float = float(os.getenv("UPLOAD_MAX_AGE_S", "3600"))       # 1 hour
# How often the sweep runs (in seconds).
UPLOAD_CLEANUP_INTERVAL_S: float = float(os.getenv("UPLOAD_CLEANUP_INTERVAL_S", "300"))  # 5 minutes


async def _upload_cleanup_loop() -> None:
    """
    Background task: periodically scan UPLOAD_DIR and delete any file whose
    modification time is older than UPLOAD_MAX_AGE_S seconds.

    Runs indefinitely on the event loop until cancelled at shutdown.
    """
    logger.info(
        "Upload cleanup task started — max_age=%.0fs, interval=%.0fs, dir=%s",
        UPLOAD_MAX_AGE_S, UPLOAD_CLEANUP_INTERVAL_S, UPLOAD_DIR,
    )
    while True:
        try:
            await asyncio.sleep(UPLOAD_CLEANUP_INTERVAL_S)
            now = time.time()
            deleted = 0
            errors = 0
            if os.path.isdir(UPLOAD_DIR):
                for fname in os.listdir(UPLOAD_DIR):
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    try:
                        if os.path.isfile(fpath):
                            age = now - os.path.getmtime(fpath)
                            if age > UPLOAD_MAX_AGE_S:
                                os.remove(fpath)
                                deleted += 1
                                logger.debug("Cleanup: deleted stale upload %s (age=%.0fs)", fname, age)
                    except OSError as e:
                        errors += 1
                        logger.warning("Cleanup: could not delete %s: %s", fpath, e)
            if deleted or errors:
                logger.info(
                    "Upload cleanup sweep complete — deleted=%d errors=%d",
                    deleted, errors,
                )
        except asyncio.CancelledError:
            logger.info("Upload cleanup task cancelled.")
            return
        except Exception as exc:
            # Never let a crash kill the background task
            logger.error("Upload cleanup task unexpected error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Prometheus — Background metrics polling loop
# ---------------------------------------------------------------------------
async def _metrics_loop() -> None:
    """
    Background task: poll Celery queue depths and worker counts every
    METRICS_POLL_INTERVAL_S seconds. Updates Prometheus Gauges and emits
    structured WARNING log entries when configurable thresholds are breached.

    Thresholds (all env-var overridable):
      WARN_QUEUE_DEPTH   — warn when any single queue depth exceeds this (default: 50)
      WARN_ACTIVE_TASKS  — warn when total active tasks across all workers exceeds this (default: 20)
      WARN_ZERO_WORKERS  — warn when no Celery workers are detected (default: true)
    """
    import redis as _redis_sync  # sync client is fine for a background thread-safe poll
    from api.workers.celery_app import celery_app as _celery_app

    logger.info(
        "Prometheus metrics loop started — poll_interval=%.0fs, "
        "warn_queue_depth=%d, warn_active_tasks=%d, warn_zero_workers=%s",
        METRICS_POLL_INTERVAL_S, WARN_QUEUE_DEPTH, WARN_ACTIVE_TASKS, WARN_ZERO_WORKERS,
    )

    _queue_names = {
        QUEUE_EVALUATE: f"celery:{QUEUE_EVALUATE}",  # Redis key Celery uses for the queue
        QUEUE_BATCH:    f"celery:{QUEUE_BATCH}",
        QUEUE_DEFAULT:  "celery",                    # the default 'celery' queue key
    }

    _r = _redis_sync.Redis.from_url(REDIS_URL, socket_timeout=2, decode_responses=True)

    try:
        while True:
            try:
                await asyncio.sleep(METRICS_POLL_INTERVAL_S)

                # ──────────────────────────────────────────────────────────────
                # 1. Queue depths — read directly from Redis via LLEN
                # ──────────────────────────────────────────────────────────────
                try:
                    for q_label, redis_key in _queue_names.items():
                        depth: int = _r.llen(redis_key) or 0
                        _QUEUE_DEPTH.labels(queue=q_label).set(depth)

                        if depth > WARN_QUEUE_DEPTH:
                            logger.warning(
                                "[metrics] THRESHOLD BREACHED — queue '%s' depth=%d > warn_threshold=%d. "
                                "Consider scaling workers or investigating slow tasks.",
                                q_label, depth, WARN_QUEUE_DEPTH,
                            )
                except Exception as _redis_err:
                    logger.debug("Metrics loop: Redis queue depth read failed: %s", _redis_err)

                # ──────────────────────────────────────────────────────────────
                # 2. Worker count + active tasks — via Celery inspect (short timeout)
                # ──────────────────────────────────────────────────────────────
                try:
                    inspector = _celery_app.control.inspect(timeout=2.0)
                    active_map = inspector.active() or {}
                    worker_count = len(active_map)
                    active_task_count = sum(len(tasks) for tasks in active_map.values())

                    _ACTIVE_WORKERS.set(worker_count)
                    _ACTIVE_TASKS.set(active_task_count)

                    # ── Threshold checks ──
                    if WARN_ZERO_WORKERS and worker_count == 0:
                        logger.warning(
                            "[metrics] THRESHOLD BREACHED — no active Celery workers detected. "
                            "Submitted tasks will queue but not be processed until a worker comes online.",
                        )

                    if active_task_count > WARN_ACTIVE_TASKS:
                        logger.warning(
                            "[metrics] THRESHOLD BREACHED — active_tasks=%d > warn_threshold=%d "
                            "across %d worker(s). System may be approaching capacity.",
                            active_task_count, WARN_ACTIVE_TASKS, worker_count,
                        )
                except Exception as _celery_err:
                    logger.debug("Metrics loop: Celery inspect failed: %s", _celery_err)

            except asyncio.CancelledError:
                logger.info("Prometheus metrics loop cancelled.")
                return
            except Exception as exc:
                # Never let a crash kill the background task
                logger.error("Prometheus metrics loop unexpected error: %s", exc, exc_info=True)
    finally:
        try:
            _r.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("AI-Interview API starting up (env=%s)…", APP_ENV)
    init_redis_pool()

    # Verify Redis is reachable before accepting traffic
    from api.dependencies import get_redis
    from api.session_store import ping_redis
    try:
        # Quick connectivity check using a one-shot client
        from redis.asyncio import Redis, ConnectionPool
        from api.config import REDIS_URL
        _pool = ConnectionPool.from_url(REDIS_URL, max_connections=2, decode_responses=True)
        _client = Redis(connection_pool=_pool)
        redis_ok = await ping_redis(_client)
        await _client.aclose()
        await _pool.aclose()

        if redis_ok:
            logger.info("Redis connectivity verified ✓")
        else:
            logger.warning("Redis ping failed — API will start but may be degraded.")
    except Exception as e:
        logger.error("Redis startup check error: %s", e)

    # Start background tasks
    _cleanup_task = asyncio.create_task(_upload_cleanup_loop(), name="upload-cleanup")
    _metrics_task = asyncio.create_task(_metrics_loop(), name="prometheus-metrics")

    logger.info("AI-Interview API ready.")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("AI-Interview API shutting down…")
    _metrics_task.cancel()
    _cleanup_task.cancel()
    for _t in (_metrics_task, _cleanup_task):
        try:
            await _t
        except asyncio.CancelledError:
            pass
    await close_redis_pool()
    logger.info("Redis pool closed.")
    shutdown_logging()   # flush the log queue before process exits


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
from fastapi.security import APIKeyHeader

app = FastAPI(
    title="AI-Interview Evaluation API",
    description=(
        "Production-grade REST + WebSocket API for AI-driven interview evaluation. "
        "Supports speaking, writing, behavioral, and technical task types with "
        "async Celery workers and Redis caching."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    # This ensures the "Authorize" padlock appears at the top of the Swagger docs
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
        "clientId": "swagger-ui"
    }
)

# ---------------------------------------------------------------------------
# Prometheus instrumentation
# — Must be called AFTER app is created and BEFORE the first request.
# — Exposes GET /metrics in the standard Prometheus text format.
# — Auto-instruments every HTTP endpoint with:
#     http_requests_total{handler, method, status_code}
#     http_request_duration_seconds{handler, method}
#     http_requests_in_progress{handler, method}
# ---------------------------------------------------------------------------
Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health/live"],  # avoid self-scrape noise
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True, tags=["Monitoring"])

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _add_process_time_header(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from fastapi import Depends
from api.routers import evaluate, health, sessions, ws, tts  # noqa: E402
from api.dependencies import verify_api_key

# Core endpoints are protected by the X-API-Key header.
app.include_router(evaluate.router, dependencies=[Depends(verify_api_key)])
app.include_router(sessions.router, dependencies=[Depends(verify_api_key)])
app.include_router(tts.router, dependencies=[Depends(verify_api_key)])

# WebSockets handle their own API key validation internally to avoid Pydantic validation errors
app.include_router(ws.router)

# Health checks remain strictly public for load-balancers and Kubernetes.
app.include_router(health.router)

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "service": "AI-Interview Evaluation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health/live",
    }

# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    from api.config import APP_HOST, APP_PORT

    uvicorn.run(
        "api.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "dev",
        log_level="debug" if APP_ENV == "dev" else "info",
    )
