"""
Unit & integration tests for the LLM Evaluation Pipeline.

Run with:
  pytest tests/ -v
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from config_llm.settings import RunMode, TaskType, settings
from models_llm.models import (
    BatchJob,
    BatchRequest,
    EvalStatus,
    EvaluationRequest,
    EvaluationResult,
)
from core_llm.prompt_builder import build_user_message
from core_llm.response_parser import parse_llm_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_request(task_type: TaskType = TaskType.COMMUNICATION_SKILLS) -> EvaluationRequest:
    return EvaluationRequest(
        session_id="session-001",
        student_id="student-001",
        task_type=task_type,
        question="Tell me about yourself.",
        student_answer="I have 3 years of experience in Python development.",
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_active_task_types_respects_num_system_prompts(self):
        assert len(settings.active_task_types) == settings.NUM_SYSTEM_PROMPTS

    def test_system_prompts_keys_match_active_types(self):
        assert set(settings.system_prompts.keys()) == set(settings.active_task_types)

    def test_redis_url_no_password(self):
        url = settings.redis_url
        assert url.startswith("redis://")
        assert settings.REDIS_HOST in url

    def test_vllm_kwargs_no_none_values(self):
        kwargs = settings.vllm_kwargs
        assert all(v is not None for v in kwargs.values())

    def test_dev_mode_uses_small_model(self):
        assert settings.RUN_MODE == RunMode.DEV  # default in tests
        assert settings.model_name == settings.DEV_MODEL_NAME


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_all_active_task_types_have_template(self):
        for task_type in settings.active_task_types:
            req = make_request(task_type)
            msg = build_user_message(req)
            assert req.question in msg
            assert req.student_answer in msg

    def test_unknown_task_type_raises(self):
        req = make_request()
        req.task_type = "nonexistent_task"  # type: ignore
        with pytest.raises((ValueError, KeyError)):
            build_user_message(req)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

class TestResponseParser:
    def test_clean_json(self):
        raw = json.dumps({"score": 85, "detailed_feedback": "Good job."})
        parsed, score, err = parse_llm_response(raw, "req-1")
        assert err is None
        assert score == 85.0
        assert parsed["detailed_feedback"] == "Good job."

    def test_fenced_json(self):
        raw = "Here is the evaluation:\n```json\n{\"score\": 72}\n```"
        parsed, score, err = parse_llm_response(raw, "req-2")
        assert err is None
        assert score == 72.0

    def test_json_with_preamble(self):
        raw = 'Sure! {"score": 60, "clarity": 6}'
        parsed, score, err = parse_llm_response(raw, "req-3")
        assert err is None
        assert score == 60.0

    def test_invalid_json_returns_error(self):
        raw = "This is not JSON at all."
        parsed, score, err = parse_llm_response(raw, "req-4")
        assert parsed is None
        assert score is None
        assert "JSON parse error" in err

    def test_empty_input_returns_error(self):
        _, _, err = parse_llm_response("", "req-5")
        assert err is not None

    def test_score_missing_gives_none(self):
        raw = json.dumps({"detailed_feedback": "No score provided."})
        parsed, score, err = parse_llm_response(raw, "req-6")
        assert err is None
        assert score is None
        assert parsed is not None


# ---------------------------------------------------------------------------
# Batch models
# ---------------------------------------------------------------------------

class TestBatchModels:
    def test_batch_job_size(self):
        req = make_request()
        batch = BatchJob(task_type=TaskType.COMMUNICATION_SKILLS)
        assert batch.size == 0
        batch.requests.append(
            BatchRequest(
                request_id="r1",
                user_message="hello",
                original_request=req,
            )
        )
        assert batch.size == 1

    def test_evaluation_result_serialisation(self):
        result = EvaluationResult(
            request_id=str(uuid.uuid4()),
            session_id="s1",
            student_id="u1",
            task_type=TaskType.TECHNICAL_KNOWLEDGE,
            status=EvalStatus.COMPLETED,
            score=88.5,
            completed_at=datetime.utcnow(),
        )
        d = result.model_dump(mode="json")
        restored = EvaluationResult(**d)
        assert restored.score == 88.5
        assert restored.status == EvalStatus.COMPLETED


# ---------------------------------------------------------------------------
# BatchManager (unit, no Redis / Celery)
# ---------------------------------------------------------------------------

class TestBatchManager:
    def test_size_trigger_dispatches(self):
        from batching.batch_manager import BatchManager

        dispatched: list[BatchJob] = []

        with patch.object(
            __import__("utils.redis_client", fromlist=["set_request_status"]),
            "set_request_status",
            lambda *a: None,
        ):
            mgr = BatchManager(dispatch_callback=dispatched.append)
            for _ in range(settings.BATCH_MAX_SIZE):
                mgr.submit(make_request())
            mgr.stop()

        assert len(dispatched) >= 1
        assert dispatched[0].size == settings.BATCH_MAX_SIZE

    def test_time_trigger_dispatches(self):
        from batching.batch_manager import BatchManager

        dispatched: list[BatchJob] = []

        with patch.object(
            __import__("utils.redis_client", fromlist=["set_request_status"]),
            "set_request_status",
            lambda *a: None,
        ):
            mgr = BatchManager(dispatch_callback=dispatched.append)
            mgr.submit(make_request())        # one item
            # Wait slightly longer than the flush window
            time.sleep(settings.BATCH_MAX_WAIT_SECONDS + 0.5)
            mgr.stop()

        assert len(dispatched) >= 1
        assert dispatched[0].size == 1


# ---------------------------------------------------------------------------
# Redis helpers (mock)
# ---------------------------------------------------------------------------

class TestRedisHelpers:
    @patch("utils.redis_client.get_redis")
    def test_cache_and_retrieve(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        result_data = {"request_id": "r1", "score": 90}
        mock_redis.get.return_value = json.dumps(result_data)

        from utils_llm.redis_client import cache_result, get_cached_result

        cache_result("r1", result_data)
        retrieved = get_cached_result("r1")
        assert retrieved == result_data

    @patch("utils.redis_client.get_redis")
    def test_cache_miss_returns_none(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        from utils_llm.redis_client import get_cached_result

        assert get_cached_result("missing-id") is None


# ---------------------------------------------------------------------------
# EvaluationService (mocked BatchManager + Redis)
# ---------------------------------------------------------------------------

class TestEvaluationService:
    @patch("core.evaluation_service.get_dedup_result", return_value=None)
    @patch("core.evaluation_service.set_request_status")
    def test_submit_returns_request_id(self, mock_status, mock_dedup):
        from batching.batch_manager import BatchManager
        from core_llm.evaluation_service import EvaluationService

        submitted: list = []
        svc = EvaluationService.__new__(EvaluationService)
        svc._batch_manager = MagicMock()
        svc._batch_manager.submit = lambda r: submitted.append(r)

        req = make_request()
        rid = svc.submit(req)
        assert rid == req.request_id
        assert len(submitted) == 1

    @patch("core.evaluation_service.get_cached_result")
    def test_get_result_not_ready(self, mock_cache):
        from core_llm.evaluation_service import EvaluationService

        mock_cache.return_value = None
        svc = EvaluationService.__new__(EvaluationService)
        svc._batch_manager = MagicMock()

        result = svc.get_result("some-id")
        assert result is None

    @patch("core.evaluation_service.get_cached_result")
    def test_get_result_returns_model(self, mock_cache):
        from core_llm.evaluation_service import EvaluationService

        sample = {
            "request_id": "r1",
            "session_id": "s1",
            "student_id": "u1",
            "task_type": TaskType.COMMUNICATION_SKILLS.value,
            "status": EvalStatus.COMPLETED.value,
            "score": 77.0,
            "cached": False,
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_cache.return_value = sample

        svc = EvaluationService.__new__(EvaluationService)
        svc._batch_manager = MagicMock()

        result = svc.get_result("r1")
        assert isinstance(result, EvaluationResult)
        assert result.score == 77.0