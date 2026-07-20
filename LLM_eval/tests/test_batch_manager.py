"""Unit tests for BatchManager – no Redis / Celery / LLM required."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from models_llm.models import Batch, EvalRequest, TaskType
from batching.batch_manager import BatchManager


@pytest.fixture()
def dispatch_mock():
    return AsyncMock()


@pytest.fixture()
def manager(dispatch_mock):
    with patch("config_llm.config.settings") as mock_settings:
        mock_settings.batching.max_batch_size = 3
        mock_settings.batching.max_wait_seconds = 0.1
        mock_settings.batching.max_concurrent_batches = 2
        return BatchManager(TaskType.GRAMMAR, dispatch_fn=dispatch_mock)


def make_request(task_type: TaskType = TaskType.GRAMMAR) -> EvalRequest:
    return EvalRequest(task_type=task_type, user_message="Hello, evaluate me.")


@pytest.mark.asyncio
async def test_batch_flushes_on_size(dispatch_mock, manager):
    for _ in range(3):
        await manager.add_request(make_request())
    await asyncio.sleep(0.05)
    dispatch_mock.assert_called_once()
    batch: Batch = dispatch_mock.call_args[0][0]
    assert batch.size == 3


@pytest.mark.asyncio
async def test_batch_flushes_on_timer(dispatch_mock, manager):
    await manager.add_request(make_request())
    # Wait longer than max_wait_seconds
    await asyncio.sleep(0.3)
    dispatch_mock.assert_called_once()
    batch: Batch = dispatch_mock.call_args[0][0]
    assert batch.size == 1


@pytest.mark.asyncio
async def test_stats_tracking(dispatch_mock, manager):
    await manager.add_request(make_request())
    stats = manager.stats
    assert stats["total_requests"] == 1
    assert stats["task_type"] == TaskType.GRAMMAR.value

