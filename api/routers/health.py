"""
health.py — /health/* API router.

Endpoints
---------
GET /health/live      Liveness probe  — always 200 if FastAPI is running
GET /health/ready     Readiness probe — checks Redis + at least 1 Celery worker
GET /health/workers   Worker details  — active count, queues, queue depths
GET /health/hardware  Hardware info   — auto-detected HARDWARE dict
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from api.config import HARDWARE, CELERY_BROKER_URL
from api.dependencies import get_redis
from api.models import (
    HardwareResponse,
    LivenessResponse,
    ReadinessResponse,
    WorkerQueueInfo,
    WorkersResponse,
)
from api.session_store import ping_redis
from api.workers.celery_app import celery_app

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# GET /health/live
# ---------------------------------------------------------------------------
@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Returns 200 OK immediately. Confirms FastAPI process is alive.",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse(status="ok", timestamp=_now_iso())


# ---------------------------------------------------------------------------
# GET /health/ready
# ---------------------------------------------------------------------------
@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Checks Redis connectivity and at least one active Celery worker.",
)
async def readiness(
    redis: Annotated[Redis, Depends(get_redis)],
) -> JSONResponse:

    redis_ok = await ping_redis(redis)

    # Check for at least one active worker via Celery Inspect
    active_workers = 0
    try:
        inspector = celery_app.control.inspect(timeout=3.0)
        active = inspector.active()        # {worker_name: [tasks]} or None
        if active:
            active_workers = len(active)
    except Exception as e:
        logger.warning("Celery inspect failed during readiness check: %s", e)

    overall_ready = redis_ok and active_workers > 0
    http_status = status.HTTP_200_OK if overall_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    body = ReadinessResponse(
        status="ready" if overall_ready else ("degraded" if redis_ok else "unavailable"),
        redis=redis_ok,
        celery_workers=active_workers,
        timestamp=_now_iso(),
    )
    return JSONResponse(content=body.model_dump(), status_code=http_status)


# ---------------------------------------------------------------------------
# GET /health/workers
# ---------------------------------------------------------------------------
@router.get(
    "/workers",
    response_model=WorkersResponse,
    summary="Celery worker details",
    description="Returns active worker count, queue list, and per-queue depth.",
)
async def workers_info() -> WorkersResponse:
    try:
        inspector = celery_app.control.inspect(timeout=5.0)
        active_map = inspector.active() or {}
        reserved_map = inspector.reserved() or {}
        stats_map = inspector.stats() or {}

        worker_names = list(active_map.keys())
        active_workers = len(worker_names)

        # Count reserved (queued) tasks per worker
        total_reserved = sum(len(tasks) for tasks in reserved_map.values())

        # Queue depth — query Redis directly via Celery broker connection
        queue_infos: list[WorkerQueueInfo] = []
        try:
            with celery_app.connection_for_read() as conn:
                with conn.channel() as ch:
                    from api.config import QUEUE_EVALUATE, QUEUE_BATCH, QUEUE_DEFAULT
                    for q_name in (QUEUE_EVALUATE, QUEUE_BATCH, QUEUE_DEFAULT):
                        try:
                            depth = ch.queue_declare(queue=q_name, passive=True).message_count
                        except Exception:
                            depth = -1
                        queue_infos.append(WorkerQueueInfo(name=q_name, depth=depth))
        except Exception as q_err:
            logger.debug("Queue depth check failed: %s", q_err)

        return WorkersResponse(
            active_workers=active_workers,
            worker_names=worker_names,
            queues=queue_infos,
            reserved_tasks=total_reserved,
        )

    except Exception as e:
        logger.warning("workers_info: Celery inspect error: %s", e)
        return WorkersResponse(
            active_workers=0,
            worker_names=[],
            queues=[],
            reserved_tasks=0,
        )


# ---------------------------------------------------------------------------
# GET /health/hardware
# ---------------------------------------------------------------------------
@router.get(
    "/hardware",
    response_model=HardwareResponse,
    summary="Hardware configuration",
    description="Returns the auto-detected hardware profile used by this deployment.",
)
async def hardware_info() -> HardwareResponse:
    return HardwareResponse(**HARDWARE)
