"""
tasks.py — Celery task definitions for the AI-interview evaluation pipeline.

Tasks
-----
evaluate_unit_task  : Run the full evaluation pipeline for a single audio unit.
evaluate_batch_task : Fan-out N evaluate_unit_task calls and aggregate.

Design notes
------------
* Tasks use run_async() from warmup.py instead of asyncio.run().
  run_async() schedules the coroutine on the persistent worker event
  loop (created at worker_ready time) and blocks until done.
  This keeps the httpx.AsyncClient connection pool to vLLM alive
  across task invocations — no TCP handshake on every task.

* Pre-loaded models (ASR, ToneAnalyzer, PronunciationPipeline, LLMClient)
  from warmup.py are injected into SharedResources before init() runs,
  so init() skips all model loading and only does a no-op check.

* Audio is stored as a temp file in UPLOAD_DIR, path passed in the
  task payload. The file is deleted after processing.

* Results are cached in Redis under result:{session_id} (JSON).

* Session status is updated in Redis at each stage (STARTED → SUCCESS/FAILURE).


"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Any, Optional

# Ensure warmup runs (registers worker_ready signal + creates persistent loop)
import api.workers.warmup  # noqa: F401  — side-effect import

from api.workers.celery_app import celery_app
from api.workers.warmup import get_warm_resources, run_async
from api.config import (
    QUEUE_EVALUATE,
    QUEUE_BATCH,
    RESULT_TTL,
    SESSION_TTL,
    REDIS_URL,
    UPLOAD_DIR,
    FINAL_INTEGRATION_DIR,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


def _ensure_repo_paths() -> None:
    """Ensure repository root and final_integration stay importable in worker processes."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for _p in [repo_root, os.path.join(repo_root, "final_integration")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Ensure final_integration paths are in sys.path inside worker process
# ---------------------------------------------------------------------------
_ensure_repo_paths()
for _p in [PROJECT_ROOT, FINAL_INTEGRATION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Sync Redis helper (used inside Celery task bodies)
# ---------------------------------------------------------------------------
import redis as _redis_sync  # noqa: E402

_sync_redis_pool = _redis_sync.ConnectionPool.from_url(
    REDIS_URL,
    decode_responses=True,
    max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
)
_sync_redis_client = _redis_sync.Redis(connection_pool=_sync_redis_pool)


def _get_sync_redis():
    return _sync_redis_client


def _update_session_sync(r, session_id: str, mapping: dict) -> None:
    mapping["updated_at"] = str(time.time())
    r.hset(f"session:{session_id}", mapping=mapping)
    r.expire(f"session:{session_id}", SESSION_TTL)


# ---------------------------------------------------------------------------
# PipelineRecord serializer
# ---------------------------------------------------------------------------
def _serialize(obj: Any) -> Any:
    """Recursively convert dataclass / arbitrary objects to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _serialize(dataclasses.asdict(obj))
    if hasattr(obj, "value"):          # Enum
        return obj.value
    if hasattr(obj, "__dict__"):
        return _serialize(vars(obj))
    return str(obj)


def _serialize_record(record) -> dict:
    """Convert a PipelineRecord dataclass to a JSON-safe dict."""
    return {
        "unit_id": record.unit_id,
        "task_type": record.task_type.value if hasattr(record.task_type, "value") else str(record.task_type),
        "transcript": record.transcript,
        "confidence": _serialize(record.confidence),
        "tone": _serialize(record.tone),
        "pronunciation": _serialize(record.pronunciation),
        "llm_eval": _serialize(record.llm_eval),
        "semantic_eval": _serialize(record.semantic_eval),
        "llm_score": record.llm_score,
        "overall_score": record.overall_score,
        "errors": dict(record.errors),
        "total_latency_ms": record.total_latency_ms,
        "cached_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Async pipeline runner (with pre-loaded resources)
# ---------------------------------------------------------------------------
async def _run_pipeline_with_resources(units: list, warm) -> list:
    """
    Run the evaluation pipeline using pre-loaded model objects from `warm`.

    Replicates the orchestration logic of run_evaluation_pipeline() but
    accepts an externally-provided SharedResources so model loading is skipped.

    The LLMClient is now also pre-loaded in warm (warmup.py creates it on
    the persistent loop at worker_ready time). If warm.llm_client is set,
    it is injected directly — no new httpx.AsyncClient is created.
    """
    _ensure_repo_paths()
    from final_integration.core_integration.pipeline import (  # noqa: PLC0415
        SharedResources, producer, consumer,
    )
    from final_integration.config_integration.config import settings  # noqa: PLC0415

    resources = SharedResources()
    if warm is not None and warm.is_ready:
        resources.asr = warm.asr
        resources.tone_analyzer = warm.tone_analyzer
        resources.pronunciation_pipeline = warm.pronunciation_pipeline
        resources.pipeline_cfg = warm.pipeline_cfg
        if warm.semantic_model is not None:
            from technical_eval.core_tech import semantic_eval as semantic_eval_module  # noqa: PLC0415
            semantic_eval_module._local_embedding_model = warm.semantic_model
        # Inject pre-created LLMClient — skips TCP handshake entirely.
        if warm.llm_client is not None:
            resources.llm_client = warm.llm_client
        logger.debug("Using pre-loaded warm resources (llm_client=%s) for pipeline run.",
                     "pre-created" if warm.llm_client else "will create")
    else:
        logger.warning("Warm resources unavailable — pipeline will load models on demand.")

    # init() sees non-None fields → skips model loading.
    # If llm_client is already set, init() skips that too.
    await resources.init()

    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.queue_max_size)
    results: list = []
    results_lock = asyncio.Lock()

    try:
        prod_task = asyncio.create_task(producer(queue, units, resources))
        consumer_tasks = [
            asyncio.create_task(consumer(i, queue, resources, results, results_lock))
            for i in range(settings.num_consumers)
        ]
        await prod_task
        await queue.join()
        await asyncio.gather(*consumer_tasks)
        return results
    finally:
        # Do NOT close resources.llm_client if it came from warm —
        # it is the persistent client shared across all tasks.
        if warm is not None and warm.llm_client is not None:
            resources.llm_client = None   # detach before close() so close() is a no-op
        await resources.close()


# ---------------------------------------------------------------------------
# Task: evaluate_unit_task
# ---------------------------------------------------------------------------
@celery_app.task(
    bind=True,
    name="api.workers.tasks.evaluate_unit_task",
    queue=QUEUE_EVALUATE,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=2,
    default_retry_delay=30,
)
def evaluate_unit_task(
    self,
    session_id: str,
    task_type: str,
    audio_path: Optional[str],
    topic: str,
    text: Optional[str],
    reference_text: Optional[str],
    question_data: Optional[dict],
    followup_audio_path: Optional[str] = None,
) -> dict:
    """
    Celery task: run the full evaluation pipeline for one audio unit.

    Args:
        session_id:          UUID identifying this evaluation session.
        task_type:           Pipeline type string: 'speaking_evaluation' etc.
        audio_path:          Absolute path to a .wav file on disk (or None for writing tasks).
        topic:               Question / prompt text.
        text:                Pre-supplied transcript / essay (writing tasks).
        reference_text:      Reference for pronunciation scoring.
        question_data:       Full question dict for technical evaluation
                             (validated TechnicalQuestionData fields).
        followup_audio_path: Absolute path to the follow-up answer audio file
                             (technical tasks only, may be None).

    Returns:
        Serialized PipelineRecord dict (also stored in Redis).
    """
    _ensure_repo_paths()
    from final_integration.models_integration.models import AudioUnit  # noqa: PLC0415
    from final_integration.config_integration.config import TaskType   # noqa: PLC0415

    r = _get_sync_redis()
    _update_session_sync(r, session_id, {"status": "STARTED", "celery_task_id": self.request.id or ""})
    logger.info(
        "[task] evaluate_unit_task started session_id=%s task_id=%s",
        session_id, self.request.id,
    )

    try:
        unit = AudioUnit(
            unit_id=session_id,
            task_type=TaskType(task_type),
            audio_path=audio_path,
            text=text,
            topic=topic or "",
            reference_text=reference_text,
        )
        if question_data:
            unit.metadata["question"] = question_data
        if followup_audio_path:
            unit.metadata["followup_audio_path"] = followup_audio_path

        warm = get_warm_resources()
        t0 = time.monotonic()

        # GROUP 2 FIX: run_async() instead of asyncio.run()
        # Reuses the persistent worker event loop — httpx connection pool stays alive.
        records = run_async(_run_pipeline_with_resources([unit], warm))
        elapsed_ms = (time.monotonic() - t0) * 1000

        if not records:
            raise RuntimeError("Pipeline returned no records for unit %s" % session_id)

        record = records[0]
        result = _serialize_record(record)

        r.setex(f"result:{session_id}", RESULT_TTL, json.dumps(result))
        _update_session_sync(r, session_id, {
            "status": "SUCCESS",
            "overall_score": str(record.overall_score) if record.overall_score is not None else "",
        })

        logger.info(
            "[task] evaluate_unit_task SUCCESS session_id=%s task_id=%s latency_ms=%.1f",
            session_id, self.request.id, elapsed_ms,
        )

        # Delete audio files only on SUCCESS.
        # If deleted in finally{}, a retry would find the file gone and fail
        # with FileNotFoundError before the pipeline even starts.
        for _path in (audio_path, followup_audio_path):
            if _path and os.path.exists(_path):
                try:
                    os.remove(_path)
                except OSError:
                    pass

        return result

    except Exception as exc:
        logger.error(
            "[task] evaluate_unit_task FAILED session_id=%s task_id=%s: %s",
            session_id, self.request.id, exc, exc_info=True,
        )
        _update_session_sync(r, session_id, {
            "status": "FAILURE",
            "error": str(exc)[:500],
        })
        raise self.retry(exc=exc, countdown=30) from exc


# ---------------------------------------------------------------------------
# Task: evaluate_batch_task
# ---------------------------------------------------------------------------
@celery_app.task(
    bind=True,
    name="api.workers.tasks.evaluate_batch_task",
    queue=QUEUE_BATCH,
    acks_late=True,
    reject_on_worker_lost=True,
)
def evaluate_batch_task(
    self,
    batch_id: str,
    units_payload: list[dict],
) -> dict:
    """
    Celery task: fan-out N evaluate_unit_task calls for a batch submission.

    GROUP 5 FIX — GroupResult now stored in Redis:
        Previously the Celery GroupResult was ignored after apply_async().
        Now we store a richer batch_meta:{batch_id} hash containing:
          - group_id:    Celery GroupResult ID for native state inspection
          - task_ids:    JSON list of per-unit Celery task IDs
          - session_ids: JSON list of session UUIDs in the same order
          - dispatched_at, total

        get_batch_status() in evaluate.py can now check each unit's live
        Celery state via AsyncResult(task_id) — much lighter than waiting
        for all N session Redis hashes to be written by workers.

    Each element of units_payload must contain:
        session_id, task_type, audio_path, topic, text,
        reference_text, question_data

    Returns a summary dict with batch_id, task_ids, and group_id.
    """
    from celery import group  # noqa: PLC0415

    logger.info("[task] evaluate_batch_task started batch_id=%s units=%d",
                batch_id, len(units_payload))

    r = _get_sync_redis()

    signatures = []
    for payload in units_payload:
        sig = evaluate_unit_task.s(
            session_id=payload["session_id"],
            task_type=payload["task_type"],
            audio_path=payload.get("audio_path"),
            topic=payload.get("topic", ""),
            text=payload.get("text"),
            reference_text=payload.get("reference_text"),
            question_data=payload.get("question_data"),
        ).set(queue=QUEUE_EVALUATE)
        signatures.append(sig)

    task_group = group(signatures)
    group_result = task_group.apply_async()

    # GROUP 5 FIX: collect child task IDs AND the GroupResult ID.
    task_ids: list[str] = []
    if hasattr(group_result, "results"):
        task_ids = [child.id for child in group_result.results]

    # group_result.id is the GroupResult UUID — usable with GroupResult(id, app=celery_app)
    group_id = group_result.id if hasattr(group_result, "id") else None

    # Store richer batch metadata in Redis.
    # batch_meta:{batch_id} holds everything needed to reconstruct batch status
    # without N individual session hash reads.
    batch_meta = {
        "batch_id":      batch_id,
        "group_id":      group_id or "",
        "session_ids":   json.dumps([p["session_id"] for p in units_payload]),
        "task_ids":      json.dumps(task_ids),
        "total":         str(len(units_payload)),
        "dispatched_at": str(time.time()),
    }
    # Store as a Redis hash for O(1) field access.
    r.hset(f"batch_meta:{batch_id}", mapping=batch_meta)
    r.expire(f"batch_meta:{batch_id}", SESSION_TTL)

    logger.info(
        "[task] evaluate_batch_task dispatched %d sub-tasks batch=%s group=%s",
        len(signatures), batch_id, group_id,
    )

    return {
        "batch_id":   batch_id,
        "group_id":   group_id,
        "task_count": len(signatures),
        "task_ids":   task_ids,
    }