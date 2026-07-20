"""
models.py — Pydantic request / response schemas for the production API.

All schemas use strict typing and snake_case field names.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import time


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TaskTypeEnum(str, Enum):
    speaking = "speaking"
    writing = "writing"
    behavioral = "behavioral"
    technical = "technical"


# ---------------------------------------------------------------------------
# Technical evaluation — question bank schema
# ---------------------------------------------------------------------------
class TechnicalQuestionData(BaseModel):
    """
    Validated representation of one row from the technical question-bank CSV.

    All fields map directly to CSV columns:
        q_id, role_family, topic, level_band, level_range, est_minutes,
        question, model_answer, evaluation_rubric, follow_up, red_flags

    Validated at the API boundary so pipeline code never receives
    structurally invalid question data.
    """
    q_id:               str   = Field(..., description="Unique question ID (e.g. TI-CF-001)")
    role_family:        str   = Field(..., description="Broad role family (e.g. CS Fundamentals)")
    topic:              str   = Field(..., description="Topic within the role family (e.g. Data Structures)")
    level_band:         str   = Field(..., description="Seniority band (e.g. Foundational, Working)")
    level_range:        Optional[str]   = Field("", description="Raw level range string (e.g. L1-L18)")
    est_minutes:        Optional[int]   = Field(None, description="Estimated answer time in minutes")
    question:           str   = Field(..., description="The main technical question text")
    model_answer:       str   = Field(..., description="Reference answer for a strong response")
    evaluation_rubric:  str   = Field(..., description="Specific criteria the answer must satisfy")
    follow_up:          Optional[str]   = Field("", description="Follow-up probe question (may be empty)")
    red_flags:          Optional[str]   = Field("", description="Known failure patterns for this question")


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"      # Celery internal state before STARTED


# ---------------------------------------------------------------------------
# Evaluate — Submit
# ---------------------------------------------------------------------------
class SubmitResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    celery_task_id: str = Field(..., description="Celery task ID for direct inspection")
    status: JobStatus = JobStatus.QUEUED
    queue: str = Field("evaluate", description="Celery queue name")


# ---------------------------------------------------------------------------
# Evaluate — Status
# ---------------------------------------------------------------------------
class StatusResponse(BaseModel):
    session_id: str
    celery_task_id: Optional[str] = None
    status: JobStatus
    result: Optional[dict[str, Any]] = None   # populated when status == SUCCESS
    error: Optional[str] = None               # populated when status == FAILURE
    created_at: float
    updated_at: float
    overall_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Evaluate — Result
# ---------------------------------------------------------------------------
class ResultResponse(BaseModel):
    session_id: str
    task_type: str
    transcript: Optional[str] = None
    confidence: Optional[dict[str, Any]] = None
    tone: Optional[dict[str, Any]] = None
    pronunciation: Optional[dict[str, Any]] = None
    llm_eval: Optional[dict[str, Any]] = None
    semantic_eval: Optional[dict[str, Any]] = None
    llm_score: Optional[float] = None
    overall_score: Optional[float] = None
    errors: dict[str, str] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    cached_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Evaluate — Batch
# ---------------------------------------------------------------------------
class BatchUnitInput(BaseModel):
    """Metadata for a single unit within a batch submission."""
    topic: str = ""
    text: Optional[str] = None
    reference_text: Optional[str] = None
    question_data: Optional[dict[str, Any]] = None   # for technical tasks


class BatchSubmitResponse(BaseModel):
    batch_id: str
    session_ids: list[str]
    total: int
    status: str = "QUEUED"


class BatchUnitStatus(BaseModel):
    session_id: str
    status: JobStatus
    overall_score: Optional[float] = None
    error: Optional[str] = None


class BatchStatusResponse(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    completion_pct: float
    units: list[BatchUnitStatus]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    task_type: Optional[TaskTypeEnum] = None
    context: Optional[dict[str, Any]] = None
    ttl: Optional[int] = Field(None, description="Custom TTL in seconds (overrides default)")


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: float
    ttl: int


class SessionMetadataResponse(BaseModel):
    session_id: str
    created_at: float
    updated_at: float
    task_type: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    status: str
    celery_task_id: Optional[str] = None
    batch_id: Optional[str] = None
    ttl_remaining: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class LivenessResponse(BaseModel):
    status: str = "ok"
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str   # "ready" | "degraded" | "unavailable"
    redis: bool
    celery_workers: int
    timestamp: str


class WorkerQueueInfo(BaseModel):
    name: str
    depth: int


class WorkersResponse(BaseModel):
    active_workers: int
    worker_names: list[str]
    queues: list[WorkerQueueInfo]
    reserved_tasks: int


class HardwareResponse(BaseModel):
    platform: str
    cpu_cores: int
    gpu_count: int
    gpu_names: list[str]
    vram_gb: list[float]
    worker_count: int
    batch_size: int
    whisper_model: str
    app_env: str
    redis_url: str


# ---------------------------------------------------------------------------
# WebSocket messages
# ---------------------------------------------------------------------------
class WsStatusMessage(BaseModel):
    session_id: str
    status: JobStatus
    overall_score: Optional[float] = None
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
