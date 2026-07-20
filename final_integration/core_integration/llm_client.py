"""
llm_client.py — sends requests to whichever backend is active
(Ollama for dev, vLLM for prod; OpenAI-compatible as an optional third).
All three speak identical OpenAI-style /v1/chat/completions, so this
client never branches on backend type — it just reads settings.active.
Parsing is delegated to LLM_eval's response_parser (via the re-export
shim at core_integration/response_parser.py).  The parsers return a
structured dict; llm_score_raw computation is done by the pipeline
layer using LLM_eval's calculate_llm_score().
"""
from __future__ import annotations
import asyncio
import time
from typing import Any
import httpx
from config_integration.config import BackendType, settings
from models_integration.models import EvalResult, EvalStatus, TaskType
from utils_integration.logger import get_logger
from core_integration.response_parser import (
    parse_behavioral_eval,
    parse_speaking_response,
    parse_writing_response,
    parse_technical_eval,
)
logger = get_logger(__name__)
# Maps each TaskType to its dedicated parser.
# The LLM_eval parsers accept only (response_text: str) — no request_id.
_PARSERS = {
    TaskType.SPEAKING: parse_speaking_response,
    TaskType.WRITING: parse_writing_response,
    TaskType.BEHAVIORAL: parse_behavioral_eval,
    TaskType.TECHNICAL: parse_technical_eval,
}
def _build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.backend == BackendType.OPENAI:
        headers["Authorization"] = f"Bearer {settings.openai_compat.api_key}"
    return headers
class LLMClient:
    """
    Async HTTP client to the active LLM inference backend.
    One shared httpx.AsyncClient per process — create via `LLMClient.create()`.
    A semaphore bounds concurrent in-flight requests (settings.llm_concurrency),
    so the producer-consumer pipeline can fan out many units without
    overwhelming the backend.
    """
    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client
        self._sem = asyncio.Semaphore(settings.llm_concurrency)
        logger.info(
            "llm_client_init",
        )
        print(f"[LLMClient] backend={settings.backend.value} "
              f"base_url={settings.base_url()} model={settings.llm_params()['model']}")
    @classmethod
    async def create(cls) -> "LLMClient":
        client = httpx.AsyncClient(
            base_url=settings.base_url(),
            headers=_build_headers(),
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        return cls(client)
    async def close(self) -> None:
        await self._http.aclose()
    async def __aenter__(self) -> "LLMClient":
        return self
    async def __aexit__(self, *exc) -> None:
        await self.close()
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def infer(self, system_prompt: str, user_message: str, task_type: TaskType,
                     request_id: str | None = None) -> EvalResult:
        request_id = request_id or f"req_{int(time.monotonic() * 1000)}"
        async with self._sem:
            return await self._infer_single(request_id, system_prompt, user_message, task_type)
    async def infer_batch(
        self,
        system_prompt: str,
        user_messages: list[tuple[str, str]],
        task_type: TaskType,
    ) -> list[EvalResult]:
        tasks = [
            self._infer_single(request_id, system_prompt, user_message, task_type)
            for request_id, user_message in user_messages
        ]
        return await asyncio.gather(*[self._bounded(t) for t in tasks])
    async def _bounded(self, coro):
        async with self._sem:
            return await coro
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
        # print(payload)
        t0 = time.monotonic()
        try:
            response = await self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.monotonic() - t0) * 1000
            return self._parse_response(request_id, task_type, data, latency_ms)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "llm_http_error request_id=%s status=%s body=%s",
                request_id, exc.response.status_code, exc.response.text[:500],
            )
            return EvalResult(
                request_id=request_id,
                task_type=task_type,
                status=EvalStatus.FAILED,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("llm_unexpected_error request_id=%s", request_id)
            return EvalResult(
                request_id=request_id,
                task_type=task_type,
                status=EvalStatus.FAILED,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
    def _build_payload(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        params = settings.llm_params()
        return {
            "model": params["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "top_p": params["top_p"],
            "stream": False,
        }
    @staticmethod
    def _parse_response(
        request_id: str,
        task_type: TaskType,
        data: dict[str, Any],
        latency_ms: float,
    ) -> EvalResult:
        choice = data["choices"][0]
        raw_text: str = choice["message"]["content"]
        usage = data.get("usage", {})
        # LLM_eval parsers take only (response_text: str) — no request_id arg.
        parser_fn = _PARSERS[task_type]
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
