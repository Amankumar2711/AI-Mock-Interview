"""
Redis helper — async + sync wrappers used across LLM_eval.

Two connection pools, one per paradigm:
  _async_pool  — redis.asyncio, used by async callers (FastAPI routes, etc.)
  _sync_pool   — redis (sync), used by Celery task bodies which run in
                 regular OS threads and cannot await coroutines.


Key schema
----------
  eval:result:{request_id}          — JSON result dict (TTL = result_ttl)
  eval:status:{request_id}          — status string (TTL = result_ttl)
  eval:dedup:{task_type}:{hash}     — JSON dedup cache (TTL = result_ttl)
  eval:batch_lock:{batch_id}        — lock string (TTL = task hard limit)
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

import redis
import redis.asyncio as aioredis

from config_llm.config import settings
from utils_llm.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Async pool (redis.asyncio) — for async callers
# ---------------------------------------------------------------------------
_async_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return the shared async Redis client (lazy-init)."""
    global _async_pool
    if _async_pool is None:
        _async_pool = aioredis.from_url(
            settings.redis.url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _async_pool


# ---------------------------------------------------------------------------
# Sync pool (redis) — for Celery task bodies
# ---------------------------------------------------------------------------
_sync_pool: redis.ConnectionPool | None = None
_sync_pool_lock = threading.Lock()


def _get_sync_redis() -> redis.Redis:
    """Return a sync Redis client from the shared connection pool (lazy-init)."""
    global _sync_pool
    if _sync_pool is None:
        with _sync_pool_lock:
            if _sync_pool is None:
                _sync_pool = redis.ConnectionPool.from_url(
                    settings.redis.url,
                    max_connections=20,
                    decode_responses=True,
                )
    return redis.Redis(connection_pool=_sync_pool)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
def result_cache_key(request_id: str) -> str:
    return f"eval:result:{request_id}"


def batch_status_key(batch_id: str) -> str:
    return f"eval:batch:{batch_id}"


def _status_key(request_id: str) -> str:
    return f"eval:status:{request_id}"


def _dedup_key(task_type: str, question: str, answer: str) -> str:
    content = f"{task_type}:{question}:{answer}"
    h = hashlib.sha256(content.encode()).hexdigest()[:32]
    return f"eval:dedup:{task_type}:{h}"


def _batch_lock_key(batch_id: str) -> str:
    return f"eval:batch_lock:{batch_id}"


# ---------------------------------------------------------------------------
# Async result cache
# ---------------------------------------------------------------------------
async def cache_result(key: str, value: dict[str, Any], ttl: int | None = None) -> None:
    r = get_redis()
    ttl = ttl or settings.redis.result_ttl
    await r.set(key, json.dumps(value, default=str), ex=ttl)
    logger.debug("cached_result key=%s ttl=%s", key, ttl)


async def get_cached_result(key: str) -> dict[str, Any] | None:
    r = get_redis()
    raw = await r.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def delete_result(key: str) -> None:
    r = get_redis()
    await r.delete(key)


# ---------------------------------------------------------------------------
# Sync result cache — used by eval_workers (Celery task bodies)
# ---------------------------------------------------------------------------
def cache_result_sync(request_id: str, value: dict[str, Any], ttl: int | None = None) -> None:
    """Sync version of cache_result for use in Celery task bodies."""
    r = _get_sync_redis()
    ttl = ttl or settings.redis.result_ttl
    r.set(result_cache_key(request_id), json.dumps(value, default=str), ex=ttl)


def get_cached_result_sync(request_id: str) -> dict[str, Any] | None:
    """Sync version of get_cached_result."""
    r = _get_sync_redis()
    raw = r.get(result_cache_key(request_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Request status — sync (Celery workers write, API reads via async)
# ---------------------------------------------------------------------------
def set_request_status(request_id: str, status: Any, ttl: int | None = None) -> None:
    """
    Store the status string for a request.
    `status` may be an EvalStatus enum or a plain string — .value is used
    if available so the stored value is always a plain string.
    """
    r = _get_sync_redis()
    value = status.value if hasattr(status, "value") else str(status)
    r.set(_status_key(request_id), value, ex=ttl or settings.redis.result_ttl)


def get_request_status(request_id: str) -> str | None:
    """Return the stored status string, or None if not found."""
    r = _get_sync_redis()
    return r.get(_status_key(request_id))


# ---------------------------------------------------------------------------
# Deduplication cache — sync
# ---------------------------------------------------------------------------
def set_dedup_result(
    task_type: str,
    question: str,
    student_answer: str,
    result: dict[str, Any],
    ttl: int | None = None,
) -> None:
    """Cache a completed result so identical future requests skip LLM inference."""
    r = _get_sync_redis()
    key = _dedup_key(task_type, question, student_answer)
    r.set(key, json.dumps(result, default=str), ex=ttl or settings.redis.result_ttl)


def get_dedup_result(
    task_type: str,
    question: str,
    student_answer: str,
) -> dict[str, Any] | None:
    """Return a previously cached dedup result, or None on miss."""
    r = _get_sync_redis()
    raw = r.get(_dedup_key(task_type, question, student_answer))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Batch idempotency lock — sync
# ---------------------------------------------------------------------------
def acquire_batch_lock(batch_id: str, ttl: int = 360) -> bool:
    """
    Try to acquire an exclusive lock for `batch_id`.
    Uses SET NX EX (atomic) so only one worker processes a given batch.

    Returns True if the lock was acquired, False if already held.
    """
    r = _get_sync_redis()
    acquired = r.set(_batch_lock_key(batch_id), "1", nx=True, ex=ttl)
    return bool(acquired)


def release_batch_lock(batch_id: str) -> None:
    """Release the batch lock. Safe to call even if not held."""
    r = _get_sync_redis()
    try:
        r.delete(_batch_lock_key(batch_id))
    except Exception as exc:
        logger.warning("release_batch_lock failed batch_id=%s: %s", batch_id, exc)