"""
sessions.py — /sessions/* API router.

Endpoints
---------
POST   /sessions/create            Pre-create a session with metadata
GET    /sessions/{session_id}      Retrieve session metadata + TTL remaining
DELETE /sessions/{session_id}      Delete session and cached result from Redis
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis

from api.config import SESSION_TTL
from api.dependencies import get_redis
from api.models import SessionCreateRequest, SessionCreateResponse, SessionMetadataResponse
from api.session_store import (
    create_session,
    delete_session,
    get_session,
    get_ttl_remaining,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /sessions/create
# ---------------------------------------------------------------------------
@router.post(
    "/create",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Explicitly create a session with metadata",
    description=(
        "Create a session before submitting audio. Useful when you want to "
        "attach user_id / context metadata or pre-register a session_id."
    ),
)
async def create_session_endpoint(
    body: SessionCreateRequest,
    redis: Annotated[Redis, Depends(get_redis)],
) -> SessionCreateResponse:

    sid = str(uuid.uuid4())
    effective_ttl = body.ttl or SESSION_TTL

    await create_session(
        redis,
        session_id=sid,
        task_type=body.task_type.value if body.task_type else None,
        user_id=body.user_id,
        context=body.context,
        ttl=effective_ttl,
    )

    logger.info("sessions/create: session_id=%s user_id=%s", sid, body.user_id)
    return SessionCreateResponse(
        session_id=sid,
        created_at=__import__("time").time(),
        ttl=effective_ttl,
    )


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}
# ---------------------------------------------------------------------------
@router.get(
    "/{session_id}",
    response_model=SessionMetadataResponse,
    summary="Retrieve session metadata",
    description="Returns all session fields including status, user context, and TTL remaining.",
)
async def get_session_endpoint(
    session_id: str,
    redis: Annotated[Redis, Depends(get_redis)],
) -> SessionMetadataResponse:

    session = await get_session(redis, session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
        )

    ttl_remaining = await get_ttl_remaining(redis, session_id)

    return SessionMetadataResponse(
        session_id=session_id,
        created_at=float(session.get("created_at", 0)),
        updated_at=float(session.get("updated_at", 0)),
        task_type=session.get("task_type"),
        user_id=session.get("user_id"),
        context=session.get("context"),
        status=session.get("status", "UNKNOWN"),
        celery_task_id=session.get("celery_task_id"),
        batch_id=session.get("batch_id"),
        ttl_remaining=max(ttl_remaining, 0),
    )


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------
@router.delete(
    "/{session_id}",
    summary="Delete a session and its cached result",
    description="Removes session hash and result cache from Redis.",
)
async def delete_session_endpoint(
    session_id: str,
    redis: Annotated[Redis, Depends(get_redis)],
) -> dict:

    deleted = await delete_session(redis, session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    logger.info("sessions/delete: session_id=%s", session_id)
    return {"session_id": session_id, "deleted": True}
