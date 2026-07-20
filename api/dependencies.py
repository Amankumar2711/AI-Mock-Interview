"""
dependencies.py — FastAPI dependency-injection helpers.

Provides:
  get_redis()   → yields an async Redis client from the shared connection pool.
                  The pool is opened once at application startup (via lifespan).
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from redis.asyncio import Redis, ConnectionPool

from api.config import API_KEY, REDIS_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security — API Key
# ---------------------------------------------------------------------------
# We use a simple X-API-Key header. It integrates natively with Swagger UI.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header: str = Depends(_api_key_header)) -> None:
    """
    Validates the X-API-Key header against the configured API_KEY.
    Raises 401 Unauthorized if missing or incorrect.
    """
    if not api_key_header or api_key_header != API_KEY:
        logger.warning("Authentication failed: invalid or missing API Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

# ---------------------------------------------------------------------------
# Module-level pool — initialised in api/main.py lifespan
# ---------------------------------------------------------------------------
_redis_pool: ConnectionPool | None = None


def init_redis_pool() -> None:
    """Create the shared async Redis connection pool. Call once at startup."""
    global _redis_pool
    if _redis_pool is not None:
        return
    _redis_pool = ConnectionPool.from_url(
        REDIS_URL,
        max_connections=50,
        decode_responses=True,
    )
    logger.info("Redis connection pool initialised: url=%s", REDIS_URL)


async def close_redis_pool() -> None:
    """Drain the pool. Call once at shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("Redis connection pool closed.")


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    FastAPI dependency that yields a Redis client backed by the shared pool.

    Usage::

        @router.get("/example")
        async def example(redis: Redis = Depends(get_redis)):
            ...
    """
    if _redis_pool is None:
        raise RuntimeError(
            "Redis pool has not been initialised. "
            "Ensure init_redis_pool() is called in the application lifespan."
        )
    client = Redis(connection_pool=_redis_pool)
    try:
        yield client
    finally:
        await client.aclose()
