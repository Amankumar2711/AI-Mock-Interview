"""
session_store.py — Async Redis-backed session CRUD for FastAPI.

Redis key schema
----------------
  session:{session_id}    → Hash  (metadata + status)
  result:{session_id}     → String (JSON-encoded PipelineRecord dict)
  batch:{batch_id}        → String (JSON-encoded list of session_ids)

All keys are created with TTLs derived from config.SESSION_TTL / RESULT_TTL.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

from redis.asyncio import Redis

from api.config import RESULT_TTL, SESSION_TTL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def _result_key(session_id: str) -> str:
    return f"result:{session_id}"


def _batch_key(batch_id: str) -> str:
    return f"batch:{batch_id}"


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

async def create_session(
    redis: Redis,
    *,
    session_id: Optional[str] = None,
    task_type: Optional[str] = None,
    user_id: Optional[str] = None,
    context: Optional[dict] = None,
    celery_task_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    ttl: Optional[int] = None,
) -> str:
    """
    Create a new session hash in Redis. Returns the session_id.
    """
    sid = session_id or str(uuid.uuid4())
    now = time.time()
    mapping: dict[str, str] = {
        "session_id": sid,
        "created_at": str(now),
        "updated_at": str(now),
        "status": "QUEUED",
    }
    if task_type:
        mapping["task_type"] = task_type
    if user_id:
        mapping["user_id"] = user_id
    if context:
        mapping["context"] = json.dumps(context)
    if celery_task_id:
        mapping["celery_task_id"] = celery_task_id
    if batch_id:
        mapping["batch_id"] = batch_id

    key = _session_key(sid)
    effective_ttl = ttl or SESSION_TTL
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, effective_ttl)
        await pipe.execute()

    logger.debug("Session created: session_id=%s", sid)
    return sid


async def get_session(redis: Redis, session_id: str) -> Optional[dict[str, Any]]:
    """Return session hash as dict, or None if not found / expired."""
    raw = await redis.hgetall(_session_key(session_id))
    if not raw:
        return None
    result: dict[str, Any] = {k: v for k, v in raw.items()}
    # Parse JSON fields
    if "context" in result:
        try:
            result["context"] = json.loads(result["context"])
        except (json.JSONDecodeError, TypeError):
            pass
    # Numeric fields
    for field in ("created_at", "updated_at"):
        if field in result:
            try:
                result[field] = float(result[field])
            except (ValueError, TypeError):
                pass
    return result


async def update_session_status(
    redis: Redis,
    session_id: str,
    status: str,
    *,
    celery_task_id: Optional[str] = None,
    overall_score: Optional[float] = None,
    error: Optional[str] = None,
    extend_ttl: bool = True,
) -> None:
    """Update status + optional fields on an existing session hash."""
    mapping: dict[str, str] = {
        "status": status,
        "updated_at": str(time.time()),
    }
    if celery_task_id is not None:
        mapping["celery_task_id"] = celery_task_id
    if overall_score is not None:
        mapping["overall_score"] = str(overall_score)
    if error is not None:
        mapping["error"] = error[:500]

    key = _session_key(session_id)
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping=mapping)
        if extend_ttl:
            pipe.expire(key, SESSION_TTL)
        await pipe.execute()


async def delete_session(redis: Redis, session_id: str) -> bool:
    """Delete session hash and associated result. Returns True if session existed."""
    async with redis.pipeline(transaction=True) as pipe:
        pipe.delete(_session_key(session_id))
        pipe.delete(_result_key(session_id))
        results = await pipe.execute()
    deleted = bool(results[0])
    if deleted:
        logger.debug("Session deleted: session_id=%s", session_id)
    return deleted


async def get_ttl_remaining(redis: Redis, session_id: str) -> int:
    """Return remaining TTL in seconds for the session key (-1 if no TTL, -2 if missing)."""
    return await redis.ttl(_session_key(session_id))


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

async def set_result(redis: Redis, session_id: str, result: dict) -> None:
    """Persist a PipelineRecord dict as JSON under result:{session_id}."""
    key = _result_key(session_id)
    payload = json.dumps(result, default=str)
    await redis.setex(key, RESULT_TTL, payload)
    logger.debug("Result cached: session_id=%s size=%d bytes", session_id, len(payload))


async def get_result(redis: Redis, session_id: str) -> Optional[dict]:
    """Return cached result dict, or None if not found / expired."""
    raw = await redis.get(_result_key(session_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to deserialize result for session %s: %s", session_id, e)
        return None


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

async def create_batch(
    redis: Redis,
    batch_id: str,
    session_ids: list[str],
    ttl: Optional[int] = None,
) -> None:
    """Store the list of session_ids that belong to a batch."""
    key = _batch_key(batch_id)
    await redis.setex(key, ttl or SESSION_TTL, json.dumps(session_ids))
    logger.debug("Batch created: batch_id=%s sessions=%d", batch_id, len(session_ids))


async def get_batch_session_ids(redis: Redis, batch_id: str) -> Optional[list[str]]:
    """Return list of session_ids for a batch, or None if not found."""
    raw = await redis.get(_batch_key(batch_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

async def ping_redis(redis: Redis) -> bool:
    """Return True if Redis is reachable."""
    try:
        return bool(await redis.ping())
    except Exception as e:
        logger.warning("Redis ping failed: %s", e)
        return False
