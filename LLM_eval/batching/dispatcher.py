"""
batching/dispatcher.py — routes a flushed Batch to the correct Celery task.


"""

from __future__ import annotations

from models_llm.models import Batch, TaskType
from config_llm.config import settings, SYSTEM_PROMPTS
from utils_llm.logger import get_logger

logger = get_logger(__name__)


async def dispatch_batch(batch: Batch) -> None:
    """
    Serialize a Batch and send it to the matching Celery task queue.

    Called by BatchManager._dispatch_fn when a batch is flushed.
    Imports get_task_for_type lazily to avoid circular imports at module
    load time (eval_workers imports evaluation_service which imports here).
    """
    from workers.eval_workers import get_task_for_type  # lazy — avoids circular

    if batch.size == 0:
        logger.warning("dispatch_batch called with empty batch — skipping")
        return

    task_type_enum = batch.task_type
    system_prompt = SYSTEM_PROMPTS.get(task_type_enum.value, "")

    # Build per-request dicts — keep only what the worker needs
    requests = []
    for item in batch.items:
        req = item.request
        requests.append({
            "request_id":       req.request_id,
            "user_message":     req.user_message,
            "original_request": req.metadata,   # carries question, student_answer etc.
        })

    payload_dict = {
        "batch_id":      batch.batch_id,
        "task_type":     task_type_enum.value,
        "system_prompt": system_prompt,
        "requests":      requests,
    }

    celery_task = get_task_for_type(task_type_enum)
    celery_task.apply_async(
        args=[payload_dict],
        queue=f"eval_{task_type_enum.value}",
    )

    logger.info(
        "batch_dispatched batch_id=%s task_type=%s size=%d",
        batch.batch_id, task_type_enum.value, batch.size,
    )