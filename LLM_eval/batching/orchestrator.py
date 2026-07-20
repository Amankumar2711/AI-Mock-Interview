"""
EvalOrchestrator – singleton that owns one BatchManager per TaskType.

Responsibilities:
  - Routes incoming EvalRequests to the correct BatchManager
  - Provides the dispatch callback that sends a Batch to Celery
  - Exposes aggregate stats
"""

from __future__ import annotations

from models_llm.models import Batch, EvalRequest, TaskType
from batching.batch_manager import BatchManager
from workers.celery_app import celery_app
from utils_llm.logger import get_logger

logger = get_logger(__name__)


class EvalOrchestrator:
    def __init__(self) -> None:
        self._managers: dict[TaskType, BatchManager] = {
            task_type: BatchManager(
                task_type=task_type,
                dispatch_fn=self._dispatch_batch,
            )
            for task_type in TaskType
        }
        logger.info(
            "orchestrator_initialised",
            num_managers=len(self._managers),
            task_types=[t.value for t in TaskType],
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def submit(self, request: EvalRequest) -> None:
        """Accept a single evaluation request and route it to the right queue."""
        manager = self._managers[request.task_type]
        await manager.add_request(request)
        logger.debug(
            "request_submitted",
            request_id=request.request_id,
            task_type=request.task_type.value,
        )

    async def flush_all(self) -> None:
        """Graceful shutdown: flush all pending batches."""
        for manager in self._managers.values():
            await manager.flush_now()
        logger.info("all_batches_flushed")

    def stats(self) -> list[dict]:
        return [m.stats for m in self._managers.values()]

    # ------------------------------------------------------------------
    # Dispatch callback
    # ------------------------------------------------------------------

    async def _dispatch_batch(self, batch: Batch) -> None:
        """
        Called by BatchManager when a batch is ready.
        Serialises the batch and enqueues a Celery task.
        """
        # Celery tasks are defined in workers/tasks.py
        # We import here to avoid circular imports at module load time.
        from workers.tasks import process_eval_batch  # noqa: PLC0415

        batch_payload = batch.model_dump(mode="json")

        result = process_eval_batch.apply_async(
            kwargs={"batch_payload": batch_payload},
            # Route to the dedicated queue for this task type
            queue=f"eval_{batch.task_type.value}",
            priority=1,  # Celery priority (lower = higher in some brokers)
        )

        logger.info(
            "batch_dispatched_to_celery",
            batch_id=batch.batch_id,
            celery_task_id=result.id,
            task_type=batch.task_type.value,
            batch_size=batch.size,
        )