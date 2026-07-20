"""
models.py — shared dataclasses/enums for the producer-consumer eval pipeline.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from config_integration.config import TaskType  # re-exported for convenience
__all__ = [
    "TaskType",
    "EvalStatus",
    "AudioUnit",
    "EvalRequest",
    "EvalResult",
    "LLMMessage",
    "LLMResponse",
    "PipelineRecord",
]
class EvalStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
@dataclass
class AudioUnit:
    """
    A single unit of work placed on the producer->consumer queue.
    For WRITING tasks, audio_path may be None (text-only).
    """
    unit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.SPEAKING
    audio_path: Optional[str] = None
    text: Optional[str] = None          # pre-supplied text (writing tasks)
    topic: str = ""
    reference_text: Optional[str] = None  # for pronunciation pipeline
    metadata: dict[str, Any] = field(default_factory=dict)
    enqueued_at: float = field(default_factory=time.monotonic)
@dataclass
class EvalRequest:
    task_type: TaskType
    user_message: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
@dataclass
class EvalResult:
    request_id: str
    task_type: TaskType
    status: EvalStatus
    result: Optional[dict[str, Any]] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
@dataclass
class LLMMessage:
    role: str
    content: str
@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
@dataclass
class PipelineRecord:
    """
    Final combined result for one AudioUnit, after all sub-pipelines +
    LLM eval have completed (or failed).
    llm_score    — 0-100 LLM-only score computed from the LLM_eval weight
                   matrices (calculate_llm_score).
    overall_score — 0-100 final score combining LLM + acoustic sub-scores
                   (pronunciation / tone / confidence) via calculate_overall_score.
                   For writing tasks this equals llm_score.
                   For technical tasks: llm_score*0.70 + semantic_score*0.30,
                   unless hallucination is flagged or LLM eval fails, in
                   which case overall_score = semantic_score.
    """
    unit_id: str
    task_type: TaskType
    transcript: str = ""
    confidence: Optional[dict] = None
    tone: Optional[dict] = None
    pronunciation: Optional[Any] = None
    llm_eval: Optional[EvalResult] = None
    semantic_eval: Optional[dict] = None   # technical eval semantic scoring result
    llm_score: Optional[float] = None       # computed LLM sub-score (0-100)
    overall_score: Optional[float] = None   # final combined score (0-100)
    errors: dict[str, str] = field(default_factory=dict)
    total_latency_ms: float = 0.0

