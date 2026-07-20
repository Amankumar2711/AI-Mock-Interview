"""
EvaluationService — the single entry-point for callers.

Responsibilities:
  • Dedup check  → return cached result immediately if available.
  • Submit new requests to BatchManager.
  • Poll / retrieve results from Redis.


"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from batching.batch_manager import BatchManager
from batching.dispatcher import dispatch_batch
from config_llm.settings import TaskType
from models_llm.models import EvalRequest, EvalStatus, EvaluationRequest, EvaluationResult
from utils_llm.logger import get_logger
from config_llm.config import settings as _llm_settings
from utils_llm.redis_client import (
    _get_sync_redis,
    cache_result_sync,       # sync — evaluation_service runs in sync context
    get_cached_result_sync,  # sync — evaluation_service runs in sync context
    get_dedup_result,
    get_request_status,
    set_request_status,
)


def get_redis_url() -> str:
    return _llm_settings.redis.url


logger = get_logger(__name__)

_RESULT_CHANNEL_PREFIX = "result_ready:"
_FALLBACK_POLL_INTERVAL = 0.5
_DEFAULT_TIMEOUT = 120


def _result_channel(request_id: str) -> str:
    return f"{_RESULT_CHANNEL_PREFIX}{request_id}"


def _submit_to_manager(manager: BatchManager, request: EvalRequest) -> None:
    """
    Submit a request to a BatchManager (which has async add_request).
    Uses the persistent worker loop if available, falls back to asyncio.run.
    """
    try:
        # Try to use the persistent worker loop (warmup.py creates it)
        from api.workers.warmup import get_worker_loop  # noqa: PLC0415
        loop = get_worker_loop()
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                manager.add_request(request), loop
            )
            future.result(timeout=5)
            return
    except (ImportError, Exception):
        pass
    # Fallback: create a temporary event loop
    asyncio.run(manager.add_request(request))


class EvaluationService:
    """
    Initialise once per process (e.g. at API startup).
    Thread-safe; async-friendly via asyncio.to_thread wrappers.
    """

    def __init__(self) -> None:
        # One BatchManager per TaskType — matches BatchManager.__init__ signature
        self._managers: dict[TaskType, BatchManager] = {
            task_type: BatchManager(
                task_type=task_type,
                dispatch_fn=dispatch_batch,
            )
            for task_type in TaskType
        }
        logger.info("evaluation_service_ready")

    # ── Submit ───────────────────────────────────────────────────────────────

    def submit(self, request: EvaluationRequest) -> str:
        """
        Enqueue a request. Returns the request_id.
        If an identical evaluation is cached, marks it CACHED immediately.
        """
        task_type = TaskType(request.task_type)

        # Read question/student_answer from metadata — EvalRequest carries
        # these in .metadata since the base model has only user_message.
        question       = request.metadata.get("question", request.user_message)
        student_answer = request.metadata.get("student_answer", request.user_message)

        cached = get_dedup_result(task_type.value, question, student_answer)
        if cached:
            cached["request_id"] = request.request_id
            cached["cached"]     = True
            cached["status"]     = EvalStatus.CACHED
            cache_result_sync(request.request_id, cached)
            set_request_status(request.request_id, EvalStatus.CACHED)
            logger.info("dedup_hit request_id=%s", request.request_id)
            return request.request_id

        set_request_status(request.request_id, EvalStatus.PENDING)
        manager = self._managers[task_type]
        _submit_to_manager(manager, request)
        return request.request_id

    # ── Poll ─────────────────────────────────────────────────────────────────

    def get_result(self, request_id: str) -> EvaluationResult | None:
        data = get_cached_result_sync(request_id)
        if data:
            return EvaluationResult(**data)
        return None

    def get_status(self, request_id: str) -> str:
        return get_request_status(request_id) or EvalStatus.PENDING

    # ── Blocking await ────────────────────────────────────────────────────────

    def await_result(
        self, request_id: str, timeout: float = _DEFAULT_TIMEOUT
    ) -> EvaluationResult | None:
        """
        Block until the result is available or timeout is reached.
        Uses Redis pub/sub on the shared connection pool.
        """
        result = self.get_result(request_id)
        if result:
            return result

        deadline = time.monotonic() + timeout

        try:
            r = _get_sync_redis()
            pubsub = r.pubsub()
            pubsub.subscribe(_result_channel(request_id))

            try:
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    message = pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=min(remaining, 5.0),
                    )
                    if message and message.get("type") == "message":
                        result = self.get_result(request_id)
                        if result:
                            return result
                    result = self.get_result(request_id)
                    if result:
                        return result
            finally:
                pubsub.unsubscribe(_result_channel(request_id))
                pubsub.close()
                # No r.close() — connection returns to pool automatically.

        except Exception as sub_exc:
            logger.warning(
                "await_result pubsub unavailable, falling back to polling: %s", sub_exc
            )
            while time.monotonic() < deadline:
                result = self.get_result(request_id)
                if result:
                    return result
                remaining = deadline - time.monotonic()
                time.sleep(min(_FALLBACK_POLL_INTERVAL, remaining))

        logger.warning("await_result_timeout request_id=%s", request_id)
        return None

    async def await_result_async(
        self, request_id: str, timeout: float = _DEFAULT_TIMEOUT
    ) -> EvaluationResult | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        interval = 0.05
        max_interval = 2.0

        result = await asyncio.to_thread(self.get_result, request_id)
        if result:
            return result

        while loop.time() < deadline:
            await asyncio.sleep(min(interval, deadline - loop.time()))
            result = await asyncio.to_thread(self.get_result, request_id)
            if result:
                return result
            interval = min(interval * 1.5, max_interval)

        return None

    # ── Batch submit ─────────────────────────────────────────────────────────

    def submit_batch(self, requests: list[EvaluationRequest]) -> list[str]:
        return [self.submit(r) for r in requests]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Flush all pending batches on shutdown."""
        for manager in self._managers.values():
            try:
                asyncio.run(manager.flush_now())
            except Exception as e:
                logger.warning("flush_on_shutdown failed: %s", e)
        logger.info("evaluation_service_stopped")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    global _service
    if _service is None:
        _service = EvaluationService()
    return _service


# ---------------------------------------------------------------------------
# notify_result_ready — called by eval_workers after caching a result.
# ---------------------------------------------------------------------------

def notify_result_ready(request_id: str) -> None:
    """
    Publish a completion signal on the per-request pub/sub channel.
    Fire-and-forget — failure is non-fatal.
    Uses shared sync pool — no new TCP connection per call.
    """
    try:
        r = _get_sync_redis()
        r.publish(_result_channel(request_id), "1")
        # No r.close() — connection returns to pool automatically.
    except Exception as exc:
        logger.warning("notify_result_ready failed request_id=%s: %s", request_id, exc)