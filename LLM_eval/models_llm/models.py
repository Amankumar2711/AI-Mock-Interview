"""
Domain models for the LLM Evaluation System.
All inter-component communication uses these Pydantic models.


"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvalStatus(str, Enum):
    PENDING    = "pending"
    QUEUED     = "queued"
    RUNNING    = "running"      # added — worker actively processing this batch
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    RETRYING   = "retrying"
    CACHED     = "cached"       # added — dedup hit, served from cache


class TaskType(str, Enum):
    """
    LLM evaluation task types — these map 1:1 to keys in
    config.config.SYSTEM_PROMPTS and config.config.WEIGHT_MATRICES.

    NOTE: pronunciation, tone, and confidence are NOT LLM tasks — they are
    scored by separate acoustic/signal-processing pipelines (Wav2Vec2,
    Librosa/Parselmouth, SpeechBrain) and are combined with the LLM score
    later by the results aggregator via calculate_overall_score().
    """
    SPEAKING   = "speaking_evaluation"
    WRITING    = "writing_evaluation"
    BEHAVIORAL = "behavioral_evaluation"
    TECHNICAL  = "technical_evaluation"


class EvalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType
    user_message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = Field(default=5, ge=1, le=10)


# Alias — evaluation_service.py imports EvaluationRequest
EvaluationRequest = EvalRequest


class EvalResult(BaseModel):
    request_id: str
    task_type: TaskType
    status: EvalStatus
    result: dict[str, Any] | None = None
    raw_output: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def raw_text(self) -> str | None:
        """Alias for raw_output — eval_workers.py accesses resp.raw_text."""
        return self.raw_output


class EvaluationResult(BaseModel):
    """
    Extended result model used by eval_workers.py.
    Carries the extra fields that the Celery batch worker writes
    (session_id, student_id, score, raw_response, parsed_evaluation).
    evaluation_service.py also imports this name.
    """
    request_id: str
    task_type: str                          # str not TaskType — workers use raw value
    status: EvalStatus
    session_id: str = ""
    student_id: str = ""
    score: float | None = None
    raw_response: str | None = None
    parsed_evaluation: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_dump(self, **kwargs) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        # Serialize datetime fields and enum values for Redis JSON storage
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, Enum):
                d[k] = v.value
        return d


class BatchItem(BaseModel):
    request: EvalRequest
    celery_task_id: str | None = None
    enqueued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Batch(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType
    items: list[BatchItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dispatched_at: datetime | None = None
    status: EvalStatus = EvalStatus.PENDING

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def request_ids(self) -> list[str]:
        return [item.request.request_id for item in self.items]


@dataclass
class BatchTaskPayload:
    """
    Payload passed to a Celery eval task as a plain dict (json-serializable).
    eval_workers.py does: payload = BatchTaskPayload(**payload_dict)

    Fields:
        batch_id:       UUID string identifying this batch.
        task_type:      TaskType.value string (e.g. "speaking_evaluation").
        system_prompt:  The system prompt for this task type.
        requests:       List of per-request dicts, each containing:
                            request_id, user_message, original_request
    """
    batch_id: str
    task_type: str
    system_prompt: str
    requests: list[dict[str, Any]] = field(default_factory=list)


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    request_id: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str