"""
llm_eval/base.py — Abstract interface that both vLLM (prod) and Ollama (dev)
backends implement.  The rest of the system talks only to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvalStatus(str, Enum):
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class EvalRequest:
    task_type:    str             # "speaking" | "writing" | "behavioral"
    request_id:   str             # e.g. session_id
    system_prompt: str
    user_message:  str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    request_id: str
    task_type:  str
    status:     EvalStatus
    result:     Optional[Dict[str, Any]] = None   # parsed LLM output
    raw_output: str = ""
    error:      str = ""
    latency_ms: float = 0.0
    prompt_tokens:     int = 0
    completion_tokens: int = 0


class LLMBackend(ABC):
    """
    Common interface for LLM inference backends.
    Implementations: VLLMBackend (prod), OllamaBackend (dev).
    """

    @abstractmethod
    async def infer(self, request: EvalRequest) -> EvalResult:
        """Send a single inference request."""

    @abstractmethod
    async def infer_batch(self, requests: List[EvalRequest]) -> List[EvalResult]:
        """
        Send multiple requests concurrently.
        Order of results matches order of requests.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release underlying HTTP connections."""

    async def __aenter__(self) -> "LLMBackend":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()