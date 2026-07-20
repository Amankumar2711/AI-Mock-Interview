"""
LLM Client – sends batched requests to the vLLM (or OpenAI-compatible) endpoint.

Supports:
  - vLLM  (self-hosted, used in both dev [quantised] and prod [full precision])
  - OpenAI-compatible cloud APIs (dev mode fallback / testing)

In both cases the wire format is identical: POST /v1/chat/completions.

"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx

from config_llm.config import BackendType, settings
from models_llm.models import EvalResult, EvalStatus, LLMMessage, LLMResponse, TaskType
from utils_llm.logger import get_logger
from core_llm.response_parser import (
    parse_behavioral_eval,
    parse_speaking_response,
    parse_technical_eval,
    parse_writing_response,
)

logger = get_logger(__name__)

# Maps each TaskType to its dedicated response parser.
# All parsers take exactly 1 arg: parse_X(response_text: str) -> dict
_PARSERS = {
    TaskType.SPEAKING:   parse_speaking_response,
    TaskType.WRITING:    parse_writing_response,
    TaskType.BEHAVIORAL: parse_behavioral_eval,
    TaskType.TECHNICAL:  parse_technical_eval,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.backend == BackendType.OPENAI:
        headers["Authorization"] = f"Bearer {settings.openai_compat.api_key}"
    return headers


def _base_url() -> str:
    if settings.backend == BackendType.OPENAI:
        return settings.openai_compat.base_url
    return settings.vllm.base_url


def _model_name() -> str:
    if settings.backend == BackendType.OPENAI:
        return settings.openai_compat.model
    params = settings.llm_params()
    return params["model"]


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Async HTTP client to the LLM inference backend.

    Thread-safety: one shared httpx.AsyncClient per process; create via
    the async context manager or call ``await LLMClient.create()``.

    For Celery task bodies (sync context), use get_llm_client() which
    returns the process-level singleton, creating it on the persistent
    worker event loop if available.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    @classmethod
    async def create(cls) -> "LLMClient":
        client = httpx.AsyncClient(
            base_url=_base_url(),
            headers=_build_headers(),
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        return cls(client)

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Public API — async
    # ------------------------------------------------------------------

    async def infer_batch(
        self,
        system_prompt: str,
        user_messages: list[tuple[str, str]],  # [(request_id, user_message), ...]
        task_type: TaskType,
    ) -> list[EvalResult]:
        """
        Send all user_messages with the shared system_prompt concurrently
        and return a list of EvalResult (one per input, order preserved).
        """
        tasks = [
            self._infer_single(request_id, system_prompt, user_message, task_type)
            for request_id, user_message in user_messages
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def complete_batch(
        self,
        system_prompt: str,
        requests: list[dict[str, Any]],
        task_type_str: str | None = None,
    ) -> list[EvalResult]:
        """
        Adapter for eval_workers.py which calls:
            llm_client.complete_batch(
                system_prompt=payload.system_prompt,
                requests=[{"request_id": ..., "user_message": ...}, ...],
            )

        Converts the dict-list format to the tuple-list that infer_batch()
        expects, resolves the TaskType enum from task_type_str, and returns
        list[EvalResult] — the same type infer_batch() returns.

        task_type_str: TaskType.value string from BatchTaskPayload.task_type.
            If not passed, defaults to SPEAKING as a safe fallback (logs warning).
            eval_workers._run_batch_sync passes payload.task_type here.
        """
        # Resolve TaskType enum from string
        if task_type_str:
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                logger.warning(
                    "complete_batch: unknown task_type_str=%s, defaulting to SPEAKING",
                    task_type_str,
                )
                task_type = TaskType.SPEAKING
        else:
            logger.warning("complete_batch: task_type_str not provided, defaulting to SPEAKING")
            task_type = TaskType.SPEAKING

        # Convert dict-list → tuple-list for infer_batch
        user_messages = [
            (r["request_id"], r["user_message"])
            for r in requests
        ]

        return await self.infer_batch(system_prompt, user_messages, task_type)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _infer_single(
        self,
        request_id: str,
        system_prompt: str,
        user_message: str,
        task_type: TaskType,
    ) -> EvalResult:
        payload = self._build_payload(system_prompt, user_message)
        t0 = time.monotonic()
        try:
            response = await self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.monotonic() - t0) * 1000
            return self._parse_response(request_id, task_type, data, latency_ms)

        except httpx.HTTPStatusError as exc:
            logger.error(
                "llm_http_error",
                request_id=request_id,
                status=exc.response.status_code,
                body=exc.response.text[:500],
            )
            return EvalResult(
                request_id=request_id,
                task_type=task_type,
                status=EvalStatus.FAILED,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("llm_unexpected_error", request_id=request_id)
            return EvalResult(
                request_id=request_id,
                task_type=task_type,
                status=EvalStatus.FAILED,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    def _build_payload(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        v = settings.vllm
        return {
            "model": _model_name(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "temperature": v.temperature,
            "max_tokens":  v.max_tokens,
            "top_p":       v.top_p,
        }

    @staticmethod
    def _parse_response(
        request_id: str,
        task_type: TaskType,
        data: dict[str, Any],
        latency_ms: float,
    ) -> EvalResult:
        choice   = data["choices"][0]
        raw_text: str = choice["message"]["content"]
        usage    = data.get("usage", {})

        from core_llm.response_parser import parse_llm_response  # noqa: PLC0415
        parser_fn = _PARSERS.get(task_type, parse_llm_response)
        # All dedicated parsers take 1 arg: parser_fn(response_text) -> dict
        # parse_llm_response (fallback) takes 2 args: (raw_text, request_id) -> tuple
        if parser_fn is parse_llm_response:
            _parsed, _score, _ = parse_llm_response(raw_text, request_id)
            parsed: dict[str, Any] = _parsed or {}
            if _score is not None:
                parsed["llm_score_raw"] = _score
        else:
            parsed: dict[str, Any] = parser_fn(raw_text)

        return EvalResult(
            request_id=request_id,
            task_type=task_type,
            status=EvalStatus.COMPLETED,
            result=parsed,
            raw_output=raw_text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Module-level singleton + get_llm_client()
# ---------------------------------------------------------------------------

_client: LLMClient | None = None
_client_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """
    Return the process-level LLMClient singleton.

    Called by eval_workers.py (Celery task bodies — sync context).
    Creates the client on the persistent worker event loop if available
    (warmup.py creates it), otherwise creates a fresh one on a temporary loop.

    Thread-safe via double-checked locking.
    """
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        # Try to use the persistent worker loop (fastest path)
        try:
            from api.workers.warmup import get_worker_loop  # noqa: PLC0415
            loop = get_worker_loop()
            if loop and loop.is_running():
                import asyncio as _asyncio  # noqa: PLC0415
                future = _asyncio.run_coroutine_threadsafe(LLMClient.create(), loop)
                _client = future.result(timeout=10)
                logger.info("get_llm_client: created on persistent worker loop")
                return _client
        except (ImportError, Exception) as e:
            logger.debug("get_llm_client: persistent loop unavailable (%s), using asyncio.run", e)

        # Fallback: create on a temporary event loop
        _client = asyncio.run(LLMClient.create())
        logger.info("get_llm_client: created via asyncio.run fallback")
        return _client


def reset_llm_client() -> None:
    """Clear the singleton. Used in tests to force re-creation."""
    global _client
    with _client_lock:
        _client = None