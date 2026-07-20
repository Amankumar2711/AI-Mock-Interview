"""
BatchManager – one instance per system-prompt (task type).

Each BatchManager:
  - Accepts incoming EvalRequests
  - Accumulates them into the current open Batch
  - Flushes a batch when EITHER:
      (a) max_batch_size is reached, OR
      (b) max_wait_seconds elapses since the first request in the batch
  - Dispatches flushed batches to Celery for async processing
  - Enforces max_concurrent_batches to apply back-pressure
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Coroutine

from config_llm.config import settings
from models_llm.models import Batch, BatchItem, EvalRequest, EvalStatus, TaskType
from utils_llm.logger import get_logger

logger = get_logger(__name__)

# Type alias for the dispatch callback injected by the orchestrator
DispatchFn = Callable[[Batch], Coroutine]


class BatchManager:
    def __init__(self, task_type: TaskType, dispatch_fn: DispatchFn) -> None:
        self.task_type = task_type
        self._dispatch_fn = dispatch_fn
        self._cfg = settings.batching

        self._lock = asyncio.Lock()
        self._current_batch: Batch = self._new_batch()
        self._flush_timer_handle: asyncio.TimerHandle | None = None

        # Semaphore limits concurrent in-flight batches
        self._concurrency_sem = asyncio.Semaphore(self._cfg.max_concurrent_batches)

        # Statistics
        self._dispatched_count = 0
        self._request_count = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def add_request(self, request: EvalRequest) -> None:
        """Add a request to the current open batch; flush if size limit hit."""
        async with self._lock:
            self._request_count += 1
            self._current_batch.items.append(BatchItem(request=request))

            if self._current_batch.size == 1:
                # First item – start the timer
                self._schedule_flush()

            if self._current_batch.size >= self._cfg.max_batch_size:
                await self._flush_locked()

    async def flush_now(self) -> None:
        """Force-flush whatever is in the current batch (e.g. on shutdown)."""
        async with self._lock:
            if self._current_batch.size > 0:
                await self._flush_locked()

    @property
    def stats(self) -> dict:
        return {
            "task_type": self.task_type.value,
            "current_batch_size": self._current_batch.size,
            "dispatched_batches": self._dispatched_count,
            "total_requests": self._request_count,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _new_batch(self) -> Batch:
        return Batch(task_type=self.task_type)

    def _schedule_flush(self) -> None:
        """Schedule a timer-based flush. Must be called with lock held."""
        loop = asyncio.get_event_loop()
        self._flush_timer_handle = loop.call_later(
            self._cfg.max_wait_seconds,
            lambda: asyncio.ensure_future(self._timer_flush()),
        )

    def _cancel_timer(self) -> None:
        if self._flush_timer_handle is not None:
            self._flush_timer_handle.cancel()
            self._flush_timer_handle = None

    async def _timer_flush(self) -> None:
        async with self._lock:
            if self._current_batch.size > 0:
                logger.debug(
                    "timer_flush_triggered",
                    task_type=self.task_type.value,
                    batch_size=self._current_batch.size,
                )
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Swap out the current batch and dispatch it. Lock MUST be held."""
        self._cancel_timer()
        batch = self._current_batch
        self._current_batch = self._new_batch()
        batch.status = EvalStatus.QUEUED
        batch.dispatched_at = datetime.now(timezone.utc)
        self._dispatched_count += 1

        logger.info(
            "batch_flushed",
            batch_id=batch.batch_id,
            task_type=self.task_type.value,
            size=batch.size,
        )

        # Dispatch asynchronously, honouring concurrency limit
        asyncio.ensure_future(self._dispatch_with_semaphore(batch))

    async def _dispatch_with_semaphore(self, batch: Batch) -> None:
        async with self._concurrency_sem:
            try:
                await self._dispatch_fn(batch)
            except Exception:
                logger.exception("dispatch_failed", batch_id=batch.batch_id)