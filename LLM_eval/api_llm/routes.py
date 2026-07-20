"""
FastAPI router for the LLM Evaluation service.

Endpoints:
  POST /eval/submit          – Submit one evaluation request
  POST /eval/submit/batch    – Submit multiple requests at once
  GET  /eval/result/{id}     – Poll for a result
  GET  /eval/stats           – BatchManager statistics
  GET  /health               – Health check
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from batching.orchestrator import orchestrator
from models_llm.models import EvalRequest, EvalStatus, TaskType
from utils_llm.redis_client import get_cached_result, result_cache_key

router = APIRouter(prefix="/eval", tags=["evaluation"])


# ---------------------------------------------------------------------------
# Request / Response schemas (API layer – thin wrappers around domain models)
# ---------------------------------------------------------------------------

class SubmitRequest(BaseModel):
    task_type: TaskType
    user_message: str
    metadata: dict[str, Any] = {}
    priority: int = 5


class SubmitResponse(BaseModel):
    request_id: str
    status: EvalStatus
    message: str


class BatchSubmitRequest(BaseModel):
    requests: list[SubmitRequest]


class BatchSubmitResponse(BaseModel):
    submitted: int
    request_ids: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/submit",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a single evaluation request",
)
async def submit_eval(body: SubmitRequest) -> SubmitResponse:
    req = EvalRequest(
        task_type=body.task_type,
        user_message=body.user_message,
        metadata=body.metadata,
        priority=body.priority,
    )
    await orchestrator.submit(req)
    return SubmitResponse(
        request_id=req.request_id,
        status=EvalStatus.QUEUED,
        message="Request accepted and queued for evaluation.",
    )


@router.post(
    "/submit/batch",
    response_model=BatchSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit multiple evaluation requests",
)
async def submit_batch(body: BatchSubmitRequest) -> BatchSubmitResponse:
    request_ids: list[str] = []
    for item in body.requests:
        req = EvalRequest(
            task_type=item.task_type,
            user_message=item.user_message,
            metadata=item.metadata,
            priority=item.priority,
        )
        await orchestrator.submit(req)
        request_ids.append(req.request_id)

    return BatchSubmitResponse(submitted=len(request_ids), request_ids=request_ids)


@router.get(
    "/result/{request_id}",
    summary="Poll for the result of a submitted request",
)
async def get_result(request_id: str) -> dict[str, Any]:
    cached = await get_cached_result(result_cache_key(request_id))
    if cached is None:
        # Still pending / not yet processed
        return {"request_id": request_id, "status": EvalStatus.PENDING}
    return cached


@router.get("/stats", summary="BatchManager statistics")
async def get_stats() -> dict[str, Any]:
    return {"managers": orchestrator.stats()}


@router.get("/health", summary="Health check", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}