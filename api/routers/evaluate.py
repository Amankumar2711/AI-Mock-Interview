"""
evaluate.py — /evaluate/* API router.

Endpoints
---------
POST   /evaluate/submit              Submit a new evaluation job
GET    /evaluate/status/{session_id} Poll job status (Redis cache → Celery)
GET    /evaluate/result/{session_id} Fetch final scored result (Redis or 404)
DELETE /evaluate/cancel/{session_id} Revoke task + mark CANCELLED
POST   /evaluate/batch               Fan-out N units → N tasks
GET    /evaluate/batch/{batch_id}    Aggregate batch status


"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Annotated, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from redis.asyncio import Redis

from api.config import (
    QUEUE_EVALUATE, SESSION_TTL, UPLOAD_DIR,
    AUDIO_MIN_DURATION_S, AUDIO_MAX_DURATION_S,
    AUDIO_MAX_SIZE_MB, AUDIO_ALLOWED_EXTENSIONS,
)
from api.dependencies import get_redis
from api.models import (
    BatchStatusResponse,
    BatchSubmitResponse,
    BatchUnitStatus,
    JobStatus,
    ResultResponse,
    StatusResponse,
    SubmitResponse,
    TechnicalQuestionData,
)
from api.session_store import (
    create_batch,
    create_session,
    get_batch_session_ids,
    get_result,
    get_session,
    set_result,
    update_session_status,
)
from api.workers.celery_app import celery_app
from api.workers.tasks import evaluate_unit_task

router = APIRouter(prefix="/evaluate", tags=["Evaluate"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _celery_state_to_job_status(state: str) -> JobStatus:
    mapping = {
        "PENDING":  JobStatus.QUEUED,
        "RECEIVED": JobStatus.QUEUED,
        "STARTED":  JobStatus.STARTED,
        "RETRY":    JobStatus.STARTED,
        "SUCCESS":  JobStatus.SUCCESS,
        "FAILURE":  JobStatus.FAILURE,
        "REVOKED":  JobStatus.CANCELLED,
    }
    return mapping.get(state, JobStatus.QUEUED)


async def _save_audio(file: UploadFile, session_id: str) -> str:
    """
    Validate and save an uploaded audio file to disk.

    Reads file content once, runs _validate_audio (extension, size, duration),
    then writes to disk via asyncio.to_thread so the event loop is never blocked.
    """
    import io as _io
    ext = os.path.splitext(file.filename or "audio.wav")[1].lower() or ".wav"
    path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")
    content = await file.read()

    # ── Extension check ───────────────────────────────────────────────
    if ext not in AUDIO_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported audio format '{ext}'. "
                f"Allowed: {', '.join(sorted(AUDIO_ALLOWED_EXTENSIONS))}"
            ),
        )

    # ── Size check ────────────────────────────────────────────────────
    size_mb = len(content) / (1024 * 1024)
    if size_mb > AUDIO_MAX_SIZE_MB:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Audio file too large ({size_mb:.1f} MB). "
                f"Maximum: {AUDIO_MAX_SIZE_MB:.0f} MB."
            ),
        )

    # ── Duration check ────────────────────────────────────────────────
    try:
        import soundfile as _sf
        def _get_dur():
            with _sf.SoundFile(_io.BytesIO(content)) as sf_f:
                return len(sf_f) / sf_f.samplerate
        duration = await asyncio.to_thread(_get_dur)

        if duration < AUDIO_MIN_DURATION_S:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Audio too short ({duration:.1f}s). "
                    f"Minimum: {AUDIO_MIN_DURATION_S:.0f}s."
                ),
            )
        if duration > AUDIO_MAX_DURATION_S:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Audio too long ({duration:.0f}s / {duration/60:.1f} min). "
                    f"Maximum: {AUDIO_MAX_DURATION_S/60:.0f} min."
                ),
            )
    except HTTPException:
        raise
    except Exception as _dur_err:
        # Cannot read duration — let the pipeline handle it rather than blocking
        logger.warning("_save_audio: duration check skipped for %s: %s", file.filename, _dur_err)

    def _write():
        with open(path, "wb") as f:
            f.write(content)

    await asyncio.to_thread(_write)
    return path


async def _save_audio_optional(file: Optional[UploadFile], session_id: str) -> Optional[str]:
    """Save audio only if the file is present and has a filename. Returns None otherwise."""
    if file and file.filename:
        return await _save_audio(file, session_id)
    return None


def _task_type_to_pipeline(task_type: str) -> str:
    """Convert short task_type (speaking) → pipeline enum value (speaking_evaluation)."""
    if task_type.endswith("_evaluation"):
        return task_type
    return f"{task_type}_evaluation"


# ---------------------------------------------------------------------------
# GROUP 5 helper — read batch metadata stored by evaluate_batch_task
# ---------------------------------------------------------------------------
async def _get_batch_meta(redis: Redis, batch_id: str) -> Optional[dict]:
    """
    Read batch_meta:{batch_id} hash written by evaluate_batch_task.
    Returns dict with group_id, task_ids (list), session_ids (list), total.
    Returns None if the key does not exist.
    """
    raw = await redis.hgetall(f"batch_meta:{batch_id}")
    if not raw:
        return None
    return {
        "group_id":    raw.get("group_id", ""),
        "task_ids":    json.loads(raw.get("task_ids", "[]")),
        "session_ids": json.loads(raw.get("session_ids", "[]")),
        "total":       int(raw.get("total", 0)),
        "dispatched_at": raw.get("dispatched_at", ""),
    }


async def _get_sessions_with_limit(redis: Redis, session_ids: list[str], limit: int = 20) -> list[Optional[dict]]:
    """Fetch session hashes with bounded concurrency to avoid a Redis thundering herd."""
    sem = asyncio.Semaphore(limit)

    async def _guarded(sid: str) -> Optional[dict]:
        async with sem:
            return await get_session(redis, sid)

    return await asyncio.gather(*[_guarded(sid) for sid in session_ids])


# ---------------------------------------------------------------------------
# POST /evaluate/submit
# ---------------------------------------------------------------------------
@router.post(
    "/submit",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new evaluation job",
    description=(
        "Accepts task_type, topic, optional audio files and text. "
        "For technical tasks, supply two audio files: `audio` for the main answer "
        "and `followup_audio` for the follow-up answer. "
        "Enqueues a Celery task and returns session_id + celery_task_id."
    ),
)
async def submit_evaluation(
    redis: Annotated[Redis, Depends(get_redis)],
    task_type: str = Form(..., description="speaking | writing | behavioral | technical"),
    topic: str = Form("", description="Question / prompt / topic for the response"),
    text: Optional[str] = Form(None, description="Pre-supplied text (writing tasks or transcript override)"),
    reference_text: Optional[str] = Form(None, description="Reference text for pronunciation scoring"),
    questions_json: Optional[str] = Form(
        None,
        description=(
            "JSON object with question metadata (required for technical tasks). "
            "Must match the question-bank schema: "
            "q_id, role_family, topic, level_band, level_range, est_minutes, "
            "question, model_answer, evaluation_rubric, follow_up, red_flags."
        ),
    ),
    session_id: Optional[str] = Form(None, description="Pre-created session ID (optional)"),
    audio: UploadFile = File(default=None, description="Main answer audio file (.wav/.mp3/.m4a)"),
    followup_audio: UploadFile = File(
        default=None,
        description="Follow-up answer audio file (.wav/.mp3/.m4a) — technical tasks only.",
    ),
) -> SubmitResponse:

    sid = session_id or str(uuid.uuid4())
    pipeline_type = _task_type_to_pipeline(task_type)

    # ── Parse and validate questions_json ─────────────────────────────────
    question_data: Optional[dict] = None
    if questions_json:
        try:
            raw_q = json.loads(questions_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=422, detail=f"Invalid questions_json: {e}") from e

        # For technical tasks, enforce the full question-bank schema at the
        # API boundary.  This surfaces missing/misspelled fields (e.g. a
        # typo in 'model_answer') as a 422 here rather than a silent failure
        # inside the pipeline.
        if pipeline_type == "technical_evaluation":
            try:
                validated = TechnicalQuestionData.model_validate(raw_q)
                question_data = validated.model_dump()
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"questions_json failed schema validation: {e}",
                ) from e
        else:
            question_data = raw_q

    # ── Save audio files ───────────────────────────────────────────────────
    audio_path: Optional[str] = None
    if audio and audio.filename:
        audio_path = await _save_audio(audio, sid)

    followup_audio_path: Optional[str] = await _save_audio_optional(
        followup_audio, f"{sid}_followup"
    )

    if pipeline_type != "writing_evaluation" and audio_path is None and not text:
        raise HTTPException(
            status_code=422,
            detail=f"task_type='{task_type}' requires either an audio file or text input.",
        )

    existing = await get_session(redis, sid)
    if not existing:
        await create_session(redis, session_id=sid, task_type=pipeline_type)

    celery_task_id = str(uuid.uuid4())
    evaluate_unit_task.apply_async(
        kwargs={
            "session_id":          sid,
            "task_type":           pipeline_type,
            "audio_path":          audio_path,
            "followup_audio_path": followup_audio_path,
            "topic":               topic,
            "text":                text,
            "reference_text":      reference_text,
            "question_data":       question_data,
        },
        task_id=celery_task_id,
        queue=QUEUE_EVALUATE,
    )

    await update_session_status(redis, sid, "QUEUED", celery_task_id=celery_task_id)

    logger.info(
        "evaluate/submit: session_id=%s celery_task_id=%s task_type=%s followup_audio=%s",
        sid, celery_task_id, pipeline_type, bool(followup_audio_path),
    )
    return SubmitResponse(session_id=sid, celery_task_id=celery_task_id, queue=QUEUE_EVALUATE)


# ---------------------------------------------------------------------------
# GET /evaluate/status/{session_id}
# ---------------------------------------------------------------------------
@router.get(
    "/status/{session_id}",
    response_model=StatusResponse,
    summary="Poll evaluation job status",
    description="Checks Redis cache first; falls back to Celery AsyncResult.",
)
async def get_evaluation_status(
    session_id: str,
    redis: Annotated[Redis, Depends(get_redis)],
) -> StatusResponse:

    session = await get_session(redis, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    stored_status = session.get("status", "QUEUED")
    celery_task_id = session.get("celery_task_id")

    if stored_status not in ("SUCCESS", "FAILURE", "CANCELLED") and celery_task_id:
        try:
            ar = AsyncResult(celery_task_id, app=celery_app)
            celery_state = _celery_state_to_job_status(ar.state)
            if celery_state.value != stored_status:
                stored_status = celery_state.value
        except Exception as _cel_err:
            logger.debug("Celery AsyncResult failed for task %s: %s", celery_task_id, _cel_err)

    result: Optional[dict] = None
    if stored_status == "SUCCESS":
        result = await get_result(redis, session_id)

    return StatusResponse(
        session_id=session_id,
        celery_task_id=celery_task_id,
        status=JobStatus(stored_status),
        result=result,
        error=session.get("error"),
        created_at=float(session.get("created_at", 0)),
        updated_at=float(session.get("updated_at", 0)),
        overall_score=float(session["overall_score"]) if session.get("overall_score") else None,
    )


# ---------------------------------------------------------------------------
# GET /evaluate/result/{session_id}
# ---------------------------------------------------------------------------
@router.get(
    "/result/{session_id}",
    response_model=ResultResponse,
    summary="Fetch final scored result",
    description="Returns the cached PipelineRecord. 404 if not ready or expired.",
)
async def get_evaluation_result(
    session_id: str,
    redis: Annotated[Redis, Depends(get_redis)],
) -> ResultResponse:

    result = await get_result(redis, session_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Result for session '{session_id}' not found or expired.",
        )
    return ResultResponse(session_id=session_id, **result)


# ---------------------------------------------------------------------------
# DELETE /evaluate/cancel/{session_id}
# ---------------------------------------------------------------------------
@router.delete(
    "/cancel/{session_id}",
    summary="Cancel a pending or running evaluation",
    description="Revokes the Celery task and marks the session as CANCELLED.",
)
async def cancel_evaluation(
    session_id: str,
    redis: Annotated[Redis, Depends(get_redis)],
) -> dict:

    session = await get_session(redis, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    celery_task_id = session.get("celery_task_id")
    if celery_task_id:
        try:
            celery_app.control.revoke(celery_task_id, terminate=True, signal="SIGTERM")
        except Exception as e:
            logger.warning("Failed to revoke task %s: %s", celery_task_id, e)

    await update_session_status(redis, session_id, "CANCELLED")

    logger.info("evaluate/cancel: session_id=%s celery_task_id=%s", session_id, celery_task_id)
    return {"session_id": session_id, "status": "CANCELLED", "celery_task_id": celery_task_id}


# ---------------------------------------------------------------------------
# POST /evaluate/batch
# ---------------------------------------------------------------------------
@router.post(
    "/batch",
    response_model=BatchSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch of audio files for evaluation",
    description=(
        "Accepts multiple audio files + metadata as a multipart form. "
        "Each file becomes an independent Celery task. Returns batch_id + list of session_ids. "
        "Audio files are saved concurrently — no sequential blocking on large batches."
    ),
)
async def submit_batch(
    redis: Annotated[Redis, Depends(get_redis)],
    task_type: str = Form(...),
    topic: str = Form(""),
    audios: list[UploadFile] = File(..., description="One or more audio files"),
    followup_audios: Optional[list[UploadFile]] = File(
        default=None,
        description="Optional follow-up audio files, one per main audio file",
    ),
    questions_json: Optional[str] = Form(None, description="JSON array — one object per audio file"),
) -> BatchSubmitResponse:

    if not audios:
        raise HTTPException(status_code=422, detail="At least one audio file is required.")

    if followup_audios is not None and len(followup_audios) != len(audios):
        raise HTTPException(
            status_code=422,
            detail="followup_audios must contain the same number of files as audios.",
        )

    pipeline_type = _task_type_to_pipeline(task_type)

    questions: list[dict] = []
    if questions_json:
        try:
            questions = json.loads(questions_json)
            if not isinstance(questions, list):
                raise ValueError("questions_json must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid questions_json: {e}") from e

    batch_id = f"batch_{uuid.uuid4().hex[:12]}"

    # Assign session_ids upfront so parallel coroutines can reference them.
    session_ids = [str(uuid.uuid4()) for _ in audios]

    # GROUP 3 FIX: all audio saves fire concurrently.
    audio_paths: list[Optional[str]] = await asyncio.gather(
        *[_save_audio_optional(audio_file, sid) for audio_file, sid in zip(audios, session_ids)]
    )
    followup_audio_paths: list[Optional[str]] = []
    if followup_audios is not None:
        followup_audio_paths = await asyncio.gather(
            *[
                _save_audio_optional(followup_audio, f"{sid}_followup")
                for followup_audio, sid in zip(followup_audios, session_ids)
            ]
        )

    # GROUP 3 FIX: all session creates fire concurrently.
    await asyncio.gather(
        *[
            create_session(redis, session_id=sid, task_type=pipeline_type, batch_id=batch_id)
            for sid in session_ids
        ]
    )

    # Store batch session list (used by get_batch_session_ids in session_store).
    await create_batch(redis, batch_id, session_ids)

    # Dispatch one Celery task per unit and record the task_id on the session.
    # apply_async is synchronous and fast (~1 ms/task) — no gather needed.
    dispatched_task_ids: list[str] = []
    for idx, (sid, audio_path) in enumerate(zip(session_ids, audio_paths)):
        question_data = questions[idx] if idx < len(questions) else None
        followup_audio_path = followup_audio_paths[idx] if followup_audio_paths else None
        task_id = str(uuid.uuid4())
        evaluate_unit_task.apply_async(
            kwargs={
                "session_id":          sid,
                "task_type":           pipeline_type,
                "audio_path":          audio_path,
                "followup_audio_path": followup_audio_path,
                "topic":               topic,
                "text":                None,
                "reference_text":     None,
                "question_data":       question_data,
            },
            task_id=task_id,
            queue=QUEUE_EVALUATE,
        )
        dispatched_task_ids.append(task_id)

    # GROUP 5 FIX: store task_id per session so get_batch_status() can use
    # AsyncResult for live Celery state without waiting for the worker to
    # write its own session hash.
    await asyncio.gather(
        *[
            update_session_status(redis, sid, "QUEUED", celery_task_id=task_id)
            for sid, task_id in zip(session_ids, dispatched_task_ids)
        ]
    )

    logger.info("evaluate/batch: batch_id=%s total=%d", batch_id, len(session_ids))
    return BatchSubmitResponse(
        batch_id=batch_id,
        session_ids=session_ids,
        total=len(session_ids),
    )


# ---------------------------------------------------------------------------
# GET /evaluate/batch/{batch_id}
# ---------------------------------------------------------------------------
@router.get(
    "/batch/{batch_id}",
    response_model=BatchStatusResponse,
    summary="Aggregate status for a batch",
    description=(
        "Returns per-unit states and overall completion percentage. "
        "Uses Celery AsyncResult for live state when the worker has not yet "
        "written the session Redis hash (avoids stale QUEUED reads)."
    ),
)
async def get_batch_status(
    batch_id: str,
    redis: Annotated[Redis, Depends(get_redis)],
) -> BatchStatusResponse:

    session_ids = await get_batch_session_ids(redis, batch_id)
    if session_ids is None:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")

    # GROUP 5 FIX: read batch_meta to get per-unit task_ids stored at dispatch time.
    # Falls back to empty task_ids if batch_meta doesn't exist (e.g. old batches
    # dispatched directly via /evaluate/submit loops instead of evaluate_batch_task).
    batch_meta = await _get_batch_meta(redis, batch_id)
    task_id_by_session: dict[str, str] = {}
    if batch_meta:
        meta_sids = batch_meta.get("session_ids", [])
        meta_tids = batch_meta.get("task_ids", [])
        task_id_by_session = dict(zip(meta_sids, meta_tids))

    # GROUP 3 FIX: all session hash reads fire concurrently, but with a bounded semaphore.
    sessions = await _get_sessions_with_limit(redis, session_ids)

    units: list[BatchUnitStatus] = []
    completed = 0
    failed = 0

    for sid, sess in zip(session_ids, sessions):
        if sess is None:
            units.append(BatchUnitStatus(
                session_id=sid, status=JobStatus.FAILURE, error="Session expired"
            ))
            failed += 1
            continue

        s = sess.get("status", "QUEUED")

        # GROUP 5 FIX: if status is still QUEUED/STARTED in Redis, check the
        # live Celery state via AsyncResult so we reflect actual worker progress
        # without waiting for the task to write its own session update.
        if s not in ("SUCCESS", "FAILURE", "CANCELLED"):
            task_id = task_id_by_session.get(sid) or sess.get("celery_task_id")
            if task_id:
                try:
                    ar = AsyncResult(task_id, app=celery_app)
                    celery_status = _celery_state_to_job_status(ar.state)
                    # Only override Redis status if Celery knows more.
                    if celery_status.value != "QUEUED":
                        s = celery_status.value
                except Exception:
                    pass  # Redis status is stale but better than crashing

        score = float(sess["overall_score"]) if sess.get("overall_score") else None
        error = sess.get("error")

        units.append(BatchUnitStatus(
            session_id=sid,
            status=JobStatus(s),
            overall_score=score,
            error=error,
        ))
        if s == "SUCCESS":
            completed += 1
        elif s == "FAILURE":
            failed += 1

    total = len(session_ids)
    completion_pct = round((completed / total * 100) if total else 0.0, 1)

    return BatchStatusResponse(
        batch_id=batch_id,
        total=total,
        completed=completed,
        failed=failed,
        completion_pct=completion_pct,
        units=units,
    )