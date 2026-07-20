"""
ws.py — WebSocket /ws/status/{session_id}

Streams real-time status updates to the client by polling Redis on a
configurable interval. The connection closes automatically when the job
reaches a terminal state (SUCCESS, FAILURE, CANCELLED).

Message format (JSON):
  {
    "session_id": "...",
    "status": "STARTED" | "SUCCESS" | "FAILURE" | "CANCELLED" | "QUEUED",
    "overall_score": 82.5,     # null unless SUCCESS
    "error": null,             # error message if FAILURE
    "timestamp": 1718600000.0
  }

The client can also send a "ping" message to keep the connection alive.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from redis.asyncio import Redis, ConnectionPool

from api.config import API_KEY, REDIS_URL, WS_POLL_INTERVAL
from api.session_store import get_result, get_session

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"SUCCESS", "FAILURE", "CANCELLED"}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@router.websocket("/ws/status/{session_id}")
async def ws_status(websocket: WebSocket, session_id: str) -> None:
    """
    Stream status updates for a session in real time.

    Connects to Redis with a fresh client (not the shared pool, which may not
    be initialised when this is called during startup / testing).
    """
    # ── Manual API Key Validation ─────────────────────────────────────────
    api_key = websocket.headers.get("x-api-key")
    if not api_key or api_key != API_KEY:
        logger.warning("WS connection rejected: Invalid or missing API Key")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API Key")
        return

    await websocket.accept()
    logger.info("WS connected: session_id=%s", session_id)

    # Create a dedicated async Redis client for this WS connection
    pool = ConnectionPool.from_url(REDIS_URL, decode_responses=True, max_connections=5)
    redis = Redis(connection_pool=pool)

    last_status: Optional[str] = None

    try:
        while True:
            # ── Poll Redis ────────────────────────────────────────────────
            session = await get_session(redis, session_id)

            if session is None:
                # Session expired or never existed
                await websocket.send_text(json.dumps({
                    "session_id": session_id,
                    "status": "NOT_FOUND",
                    "error": f"Session '{session_id}' not found or expired.",
                    "timestamp": time.time(),
                }))
                break

            current_status = session.get("status", "QUEUED")

            # ── Send update if status changed ─────────────────────────────
            if current_status != last_status:
                message: dict = {
                    "session_id": session_id,
                    "status": current_status,
                    "overall_score": None,
                    "error": None,
                    "timestamp": time.time(),
                }

                if current_status == "SUCCESS":
                    score_raw = session.get("overall_score")
                    message["overall_score"] = float(score_raw) if score_raw else None

                if current_status == "FAILURE":
                    message["error"] = session.get("error")

                await websocket.send_text(json.dumps(message))
                logger.debug("WS update: session_id=%s status=%s", session_id, current_status)
                last_status = current_status

            # ── Terminal: close cleanly ───────────────────────────────────
            if current_status in _TERMINAL_STATUSES:
                # Send result payload on success
                if current_status == "SUCCESS":
                    result = await get_result(redis, session_id)
                    if result:
                        await websocket.send_text(json.dumps({
                            "session_id": session_id,
                            "status": "RESULT",
                            "result": result,
                            "timestamp": time.time(),
                        }))
                break

            # ── Handle incoming client messages (ping / close) ────────────
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=WS_POLL_INTERVAL,
                )
                if raw.strip().lower() in ("close", "disconnect"):
                    break
            except asyncio.TimeoutError:
                pass   # normal — just means no client message in this poll window
            except WebSocketDisconnect:
                logger.info("WS client disconnected: session_id=%s", session_id)
                return

    except WebSocketDisconnect:
        logger.info("WS disconnected: session_id=%s", session_id)
    except Exception as exc:
        logger.error("WS error: session_id=%s error=%s", session_id, exc, exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "session_id": session_id,
                "status": "ERROR",
                "error": str(exc),
                "timestamp": time.time(),
            }))
        except Exception:
            pass
    finally:
        await redis.aclose()
        await pool.aclose()
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WS closed: session_id=%s", session_id)
