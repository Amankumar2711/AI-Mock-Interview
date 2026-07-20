"""
run_example.py — Local smoke test for the LLM Evaluation Service.

Tests three modes without needing the FastAPI server or Celery running:

  MODE 1 — Mock (no GPU, no API key needed)
            Stubs the HTTP call; verifies batching + parsing logic end-to-end.

  MODE 2 — OpenAI-compatible cloud (needs OPENAI_API_KEY in env / .env)
            Hits a real LLM; good for prompt tuning on a laptop.

  MODE 3 — Local vLLM (needs vLLM server running on localhost:8000)
            Full local stack test.

Usage:
  python run_example.py          # runs MODE 1 (mock) by default
  python run_example.py mock
  python run_example.py openai
  python run_example.py vllm
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ── ensure project root is on the path ───────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(__file__))

from models_llm.models import EvalRequest, EvalResult, EvalStatus, TaskType
from config_llm.config import settings, SYSTEM_PROMPTS

# ─────────────────────────────────────────────────────────────────────────────
# Sample student data
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_REQUESTS = [
    EvalRequest(
        task_type=TaskType.GRAMMAR,
        user_message=(
            "Question: Tell me about your favourite hobby.\n"
            "Transcript: I likes to play cricket with my friends. "
            "We goes to ground every sunday and plays match together. "
            "Cricket are very excited sport."
        ),
        metadata={"student_id": "stu_001", "session": "ses_abc"},
    ),
    EvalRequest(
        task_type=TaskType.FLUENCY,
        user_message=(
            "Question: Describe your typical morning routine.\n"
            "Transcript: Um... I wake up at... uh... seven o'clock and then "
            "I... I brush my teeth and... um... have breakfast. "
            "It is... you know... a normal morning for me."
        ),
        metadata={"student_id": "stu_001", "session": "ses_abc"},
    ),
    EvalRequest(
        task_type=TaskType.CONTENT,
        user_message=(
            "Question: What are the causes of climate change?\n"
            "Transcript: Climate change is mainly caused by human activities "
            "like burning fossil fuels, deforestation, and industrial emissions. "
            "These release greenhouse gases like CO2 which trap heat in the atmosphere."
        ),
        metadata={"student_id": "stu_002", "session": "ses_xyz"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: EvalResult) -> None:
    print(f"\n{'─'*60}")
    print(f"  request_id : {result.request_id}")
    print(f"  task_type  : {result.task_type.value}")
    print(f"  status     : {result.status.value}")
    print(f"  latency    : {result.latency_ms:.1f} ms")
    if result.result:
        print(f"  result     :")
        for k, v in result.result.items():
            print(f"    {k}: {v}")
    if result.error:
        print(f"  error      : {result.error}")


def _mock_llm_json(task_type: TaskType) -> str:
    """Return a plausible fake LLM JSON response for each task type."""
    payloads = {
        TaskType.GRAMMAR: {
            "score": 4,
            "strengths": ["attempts full sentences"],
            "weaknesses": ["subject-verb agreement", "article usage"],
            "feedback": "Focus on verb conjugation rules for third-person singular.",
        },
        TaskType.FLUENCY: {
            "score": 5,
            "filler_word_count": 6,
            "pace_assessment": "Hesitant with frequent pauses",
            "feedback": "Reduce filler words; practice speaking without pausing.",
        },
        TaskType.CONTENT: {
            "score": 8,
            "relevance": 9,
            "accuracy": 8,
            "depth": 7,
            "feedback": "Good coverage of key causes. Could mention feedback loops.",
        },
        TaskType.PRONUNCIATION: {
            "score": 7,
            "mispronounced_words": ["particularly", "environment"],
            "clarity": 8,
            "feedback": "Overall clear; work on multi-syllabic words.",
        },
        TaskType.CONFIDENCE: {
            "score": 5,
            "tone_assessment": "Uncertain, lacks assertiveness",
            "hesitation_count": 8,
            "feedback": "Speak with a steady pace to project more confidence.",
        },
    }
    return json.dumps(payloads.get(task_type, {"score": 7, "feedback": "Good effort."}))


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 – Mock (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

async def run_mock() -> None:
    """
    Bypasses HTTP entirely. Uses a monkey-patched LLMClient to verify
    batching, parsing, and result assembly without any external service.
    """
    import httpx
    from core_llm.llm_client import LLMClient

    print("\n  MODE: MOCK  (no LLM server required)")
    print("="*60)

    async def fake_post(url, **kwargs):
        body = kwargs.get("json", {})
        # Detect task type from system prompt content
        system_content = body["messages"][0]["content"]
        task_type = TaskType.GRAMMAR  # default
        for tt in TaskType:
            if tt.value.replace("_evaluation", "").replace("_", " ") in system_content.lower():
                task_type = tt
                break

        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"content": _mock_llm_json(task_type)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 120, "completion_tokens": 60},
            },
        )

    mock_http = MagicMock()
    mock_http.post = AsyncMock(side_effect=fake_post)
    client = LLMClient(mock_http)

    t0 = time.monotonic()
    for req in SAMPLE_REQUESTS:
        system_prompt = SYSTEM_PROMPTS[req.task_type.value]
        results = await client.infer_batch(
            system_prompt=system_prompt,
            user_messages=[(req.request_id, req.user_message)],
            task_type=req.task_type,
        )
        _print_result(results[0])

    print(f"\n  Mock run complete in {(time.monotonic()-t0)*1000:.0f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 – Real OpenAI-compatible cloud endpoint
# ─────────────────────────────────────────────────────────────────────────────

async def run_openai() -> None:
    """
    Requires:
      export OPENAI_API_KEY=sk-...
      export APP_BACKEND=openai          # optional, detected automatically
    """
    from core_llm.llm_client import LLMClient

    api_key = os.environ.get("OPENAI_API_KEY", settings.openai_compat.api_key)
    if not api_key or api_key.startswith("sk-dummy"):
        print("  OPENAI_API_KEY not set. Export it and retry.")
        sys.exit(1)

    print(f"\n☁️   MODE: OPENAI  ({settings.openai_compat.base_url})")
    print("="*60)

    # Force backend to openai for this run
    import httpx
    client = await LLMClient.create()

    t0 = time.monotonic()
    for req in SAMPLE_REQUESTS:
        system_prompt = SYSTEM_PROMPTS[req.task_type.value]
        results = await client.infer_batch(
            system_prompt=system_prompt,
            user_messages=[(req.request_id, req.user_message)],
            task_type=req.task_type,
        )
        _print_result(results[0])

    await client.close()
    print(f"\n  OpenAI run complete in {(time.monotonic()-t0)*1000:.0f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 – Local vLLM server
# ─────────────────────────────────────────────────────────────────────────────

async def run_vllm() -> None:
    """
    Requires vLLM server running locally. Start it first:

      # DEV (quantised, ~6GB VRAM):
      bash scripts/start_vllm_dev.sh

      # OR prod:
      bash scripts/start_vllm_prod.sh
    """
    import httpx
    from core_llm.llm_client import LLMClient

    vllm_url = f"http://{settings.vllm.host}:{settings.vllm.port}"
    print(f"\n🖥️   MODE: vLLM  ({vllm_url})")
    print("="*60)

    # Quick connectivity check
    try:
        async with httpx.AsyncClient(timeout=3) as probe:
            r = await probe.get(f"{vllm_url}/health")
            if r.status_code != 200:
                raise ConnectionError(f"vLLM health check returned {r.status_code}")
        print("  vLLM server reachable ✓")
    except Exception as e:
        print(f"  Cannot reach vLLM at {vllm_url}: {e}")
        print("   Run:  bash scripts/start_vllm_dev.sh")
        sys.exit(1)

    client = await LLMClient.create()
    t0 = time.monotonic()

    for req in SAMPLE_REQUESTS:
        system_prompt = SYSTEM_PROMPTS[req.task_type.value]
        results = await client.infer_batch(
            system_prompt=system_prompt,
            user_messages=[(req.request_id, req.user_message)],
            task_type=req.task_type,
        )
        _print_result(results[0])

    await client.close()
    print(f"\n  vLLM run complete in {(time.monotonic()-t0)*1000:.0f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Batching stress test (works in mock mode)
# ─────────────────────────────────────────────────────────────────────────────

async def run_batch_stress_test() -> None:
    """
    Fires 20 requests concurrently and verifies the BatchManager
    groups them correctly by task_type and flushes on size/time.
    Runs entirely in-process — no Redis, no Celery needed.
    """
    from batching.batch_manager import BatchManager
    from models_llm.models import Batch
    import random

    print("\n  BATCH STRESS TEST  (in-process, no external deps)")
    print("="*60)

    dispatched_batches: list[Batch] = []

    async def capture_dispatch(batch: Batch) -> None:
        dispatched_batches.append(batch)
        print(f"  → Batch dispatched | task={batch.task_type.value:30s} | size={batch.size}")

    # Override batch config for the test
    settings.batching.max_batch_size = 5
    settings.batching.max_wait_seconds = 0.3

    managers = {
        tt: BatchManager(task_type=tt, dispatch_fn=capture_dispatch)
        for tt in TaskType
    }

    task_types = list(TaskType)
    requests = [
        EvalRequest(
            task_type=random.choice(task_types),
            user_message=f"Evaluate student response #{i}.",
        )
        for i in range(20)
    ]

    # Submit all concurrently
    await asyncio.gather(*[
        managers[req.task_type].add_request(req)
        for req in requests
    ])

    # Wait for timer-based flushes
    await asyncio.sleep(0.5)

    total_dispatched = sum(b.size for b in dispatched_batches)
    print(f"\n  Total requests submitted : 20")
    print(f"  Total batches dispatched : {len(dispatched_batches)}")
    print(f"  Total items in batches   : {total_dispatched}")
    assert total_dispatched == 20, " Some requests were lost!"
    print("  All 20 requests accounted for across batches.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "mock":   run_mock,
    "openai": run_openai,
    "vllm":   run_vllm,
    "stress": run_batch_stress_test,
}

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "mock"

    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  LLM Evaluation Service — Local Test Runner")
    print(f"  mode: {mode}")
    print(f"{'═'*60}")

    asyncio.run(MODES[mode]())