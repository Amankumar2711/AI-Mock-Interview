"""
Celery tasks that execute LLM batch inference.

Each active TaskType has its own task function so queues remain homogeneous
and workers can be scaled per domain (e.g., more workers for code_review).

Task flow per batch:
  1. Acquire a Redis lock (idempotency guard).
  2. Call the LLM client with the full batch concurrently.
  3. Parse each response.
  4. Cache individual results in Redis.
  5. Publish completion signal per result (for EvaluationService pub/sub).
  6. Release the lock.


"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from typing import Any

from celery import Task
from celery.utils.log import get_task_logger

from config_llm.settings import TaskType, settings
from core_llm.llm_client import get_llm_client
from core_llm.evaluation_service import notify_result_ready
from models_llm.models import BatchTaskPayload, EvalStatus, EvaluationResult
from core_llm.response_parser import parse_llm_response
from utils_llm.redis_client import (
    acquire_batch_lock,
    cache_result_sync,   # sync version — task bodies cannot await
    release_batch_lock,
    set_dedup_result,
    set_request_status,
)
from workers.celery_app import celery_app

logger = get_task_logger(__name__)


# ---------------------------------------------------------------------------
# Persistent event loop for this worker process
# ---------------------------------------------------------------------------
_batch_loop: asyncio.AbstractEventLoop | None = None
_batch_loop_thread: threading.Thread | None = None


def _get_batch_loop() -> asyncio.AbstractEventLoop:
    """
    Return (and lazily create) the persistent event loop for LLM batch calls.
    Thread-safe: protected by a module-level lock so concurrent task starts
    cannot create two loops.
    """
    global _batch_loop, _batch_loop_thread
    if _batch_loop is not None and _batch_loop.is_running():
        return _batch_loop

    with _loop_create_lock:
        # Double-check after acquiring lock
        if _batch_loop is not None and _batch_loop.is_running():
            return _batch_loop

        loop = asyncio.new_event_loop()
        t = threading.Thread(
            target=loop.run_forever,
            daemon=True,
            name="llm-eval-worker-loop",
        )
        t.start()
        _batch_loop = loop
        _batch_loop_thread = t
        logger.info("LLM eval persistent event loop started in thread %s", t.name)

    return _batch_loop


_loop_create_lock = threading.Lock()


def _run_async_on_loop(coro) -> Any:
    """Submit coroutine to the persistent loop and block until result."""
    loop = _get_batch_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ---------------------------------------------------------------------------
# Core execution logic (shared by all task variants)
# ---------------------------------------------------------------------------

def _run_batch_sync(payload_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Synchronous wrapper around async LLM calls.
    Returns a summary dict: {batch_id, processed, failed}.
    """
    payload = BatchTaskPayload(**payload_dict)
    batch_id = payload.batch_id

    # Idempotency: if another worker already picked this up, skip.
    if not acquire_batch_lock(batch_id, ttl=settings.CELERY_TASK_HARD_TIME_LIMIT):
        logger.warning("batch_already_locked batch_id=%s", batch_id)
        return {"batch_id": batch_id, "processed": 0, "failed": 0, "skipped": True}

    try:
        llm_client = get_llm_client()
        requests = payload.requests

        # Mark all as running
        for req in requests:
            set_request_status(req["request_id"], EvalStatus.RUNNING)

        # KEY FIX: use persistent loop instead of asyncio.run()
        # This keeps the httpx connection pool to vLLM alive across batches.
        responses = _run_async_on_loop(
            llm_client.complete_batch(
                system_prompt=payload.system_prompt,
                requests=[
                    {"request_id": r["request_id"], "user_message": r["user_message"]}
                    for r in requests
                ],
                task_type_str=payload.task_type,  # passes TaskType string for parser selection
            )
        )

        orig_by_id = {r["request_id"]: r["original_request"] for r in requests}
        processed = failed = 0

        for resp in responses:
            orig = orig_by_id.get(resp.request_id, {})
            task_type = payload.task_type
            question = orig.get("question", "")
            student_answer = orig.get("student_answer", "")

            if resp.error:
                result = EvaluationResult(
                    request_id=resp.request_id,
                    session_id=orig.get("session_id", ""),
                    student_id=orig.get("student_id", ""),
                    task_type=task_type,
                    status=EvalStatus.FAILED,
                    error=resp.error,
                    latency_ms=resp.latency_ms,
                    completed_at=datetime.utcnow(),
                )
                failed += 1
            else:
                parsed, score, parse_error = parse_llm_response(
                    resp.raw_text, resp.request_id
                )
                if parse_error:
                    result = EvaluationResult(
                        request_id=resp.request_id,
                        session_id=orig.get("session_id", ""),
                        student_id=orig.get("student_id", ""),
                        task_type=task_type,
                        status=EvalStatus.FAILED,
                        raw_response=resp.raw_text,
                        error=parse_error,
                        latency_ms=resp.latency_ms,
                        completed_at=datetime.utcnow(),
                    )
                    failed += 1
                else:
                    result = EvaluationResult(
                        request_id=resp.request_id,
                        session_id=orig.get("session_id", ""),
                        student_id=orig.get("student_id", ""),
                        task_type=task_type,
                        status=EvalStatus.COMPLETED,
                        score=score,
                        raw_response=resp.raw_text,
                        parsed_evaluation=parsed,
                        latency_ms=resp.latency_ms,
                        completed_at=datetime.utcnow(),
                    )
                    set_dedup_result(task_type, question, student_answer, result.model_dump(mode="json"))
                    processed += 1

            result_dict = result.model_dump(mode="json")
            cache_result_sync(resp.request_id, result_dict)
            set_request_status(resp.request_id, result.status)

            # Publish completion signal — wakes await_result() pub/sub subscriber
            # immediately instead of waiting for the next poll interval.
            notify_result_ready(resp.request_id)

        logger.info(
            "batch_completed batch_id=%s task_type=%s processed=%d failed=%d",
            batch_id, task_type, processed, failed,
        )
        return {"batch_id": batch_id, "processed": processed, "failed": failed}

    finally:
        release_batch_lock(batch_id)


# ---------------------------------------------------------------------------
# Celery task factory — one task per TaskType
# ---------------------------------------------------------------------------

def _make_eval_task(task_type: TaskType) -> Task:
    task_name = f"process_batch_{task_type.value}"

    @celery_app.task(
        name=f"workers.eval_worker.{task_name}",
        bind=True,
        max_retries=settings.CELERY_MAX_RETRIES,
        default_retry_delay=settings.CELERY_RETRY_BACKOFF,
        queue=f"eval_{task_type.value}",
        acks_late=True,
    )
    def _task(self: Task, payload_dict: dict[str, Any]) -> dict[str, Any]:
        try:
            return _run_batch_sync(payload_dict)
        except Exception as exc:
            logger.error(
                "batch_task_error task_type=%s error=%s retries=%d",
                task_type.value, str(exc), self.request.retries,
            )
            raise self.retry(
                exc=exc,
                countdown=settings.CELERY_RETRY_BACKOFF * (self.request.retries + 1),
            )

    _task.__name__ = task_name
    return _task


# Register one Celery task per active TaskType and expose them in module scope
_task_registry: dict[TaskType, Task] = {}
for _tt in settings.active_task_types:
    _registered = _make_eval_task(_tt)
    _task_registry[_tt] = _registered
    globals()[f"process_batch_{_tt.value}"] = _registered


def get_task_for_type(task_type: TaskType) -> Task:
    if task_type not in _task_registry:
        raise ValueError(f"No Celery task registered for {task_type!r}")
    return _task_registry[task_type]