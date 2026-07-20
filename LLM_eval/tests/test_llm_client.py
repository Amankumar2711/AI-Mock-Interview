"""Unit tests for LLMClient – mocks HTTP calls."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from models_llm.models import EvalStatus, TaskType
from core_llm.llm_client import LLMClient


def _mock_http_response(content: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "choices": [
                {"message": {"content": json.dumps(content)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


@pytest.mark.asyncio
async def test_infer_batch_success():
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(
        return_value=_mock_http_response({"score": 8, "feedback": "Good"})
    )

    client = LLMClient(mock_http)
    results = await client.infer_batch(
        system_prompt="You are an evaluator.",
        user_messages=[("req-1", "Evaluate this.")],
        task_type=TaskType.GRAMMAR,
    )

    assert len(results) == 1
    assert results[0].status == EvalStatus.COMPLETED
    assert results[0].result == {"score": 8, "feedback": "Good"}
    assert results[0].prompt_tokens == 100


@pytest.mark.asyncio
async def test_infer_batch_http_error():
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500, text="err")
        )
    )

    client = LLMClient(mock_http)
    results = await client.infer_batch(
        system_prompt="You are an evaluator.",
        user_messages=[("req-err", "bad request")],
        task_type=TaskType.FLUENCY,
    )

    assert results[0].status == EvalStatus.FAILED
    assert "HTTP 500" in results[0].error