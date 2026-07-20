"""
test_pipeline_logic.py — validates routing/concurrency without real
audio models or a live LLM backend. Monkeypatches:
  - LLMClient._infer_single -> fast fake EvalResult
  - core.pipeline._IMPORTS_OK -> False path is exercised naturally
    (sub-pipelines unavailable in this sandbox), confirming the
    "llm-only" degraded path works, and confirms TASK_PIPELINES routing
    decides which keys *would* be scheduled.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config_integration.config import TaskType, TASK_PIPELINES
from models_integration.models import AudioUnit, EvalResult, EvalStatus
import core_integration.pipeline as pipeline_mod
import core_integration.llm_client as llm_client_mod


async def fake_infer(self, system_prompt, user_message, task_type, request_id=None):
    await asyncio.sleep(0.05)
    return EvalResult(
        request_id=request_id or "x",
        task_type=task_type,
        status=EvalStatus.COMPLETED,
        result={"llm_eval": {"clarity": {"score": 8, "rationale": "clear"}},
                "llm_summary": "Solid response.", "llm_score_raw": 82.0,
                "parse_error": False},
        raw_output="{}",
        prompt_tokens=100, completion_tokens=50, latency_ms=50.0,
    )


async def main():
    llm_client_mod.LLMClient.infer = fake_infer
    # avoid real network connect in create()
    async def fake_create(cls):
        return cls.__new__(cls)
    orig_init = llm_client_mod.LLMClient.__init__
    llm_client_mod.LLMClient.__init__ = lambda self, *a, **k: None

    resources = pipeline_mod.SharedResources()
    resources.llm_client = llm_client_mod.LLMClient.__new__(llm_client_mod.LLMClient)

    # confirm routing table
    print("TASK_PIPELINES routing:")
    for t, pipes in TASK_PIPELINES.items():
        print(f"  {t.value:25s} -> {sorted(pipes)}")

    units = [
        AudioUnit(task_type=TaskType.SPEAKING, audio_path=None, text="I led a team through a tough launch."),
        AudioUnit(task_type=TaskType.WRITING, audio_path=None, text="Remote work essay text here."),
        AudioUnit(task_type=TaskType.BEHAVIORAL, audio_path=None, text="A conflict with a teammate story."),
    ]

    records = []
    for u in units:
        rec = await pipeline_mod.process_unit(u, resources)
        records.append(rec)
        print(f"\nUnit {u.unit_id} [{u.task_type.value}]")
        print(f"  llm_eval present: {rec.llm_eval is not None}, status={rec.llm_eval.status if rec.llm_eval else None}")
        print(f"  confidence present: {rec.confidence is not None}")
        print(f"  tone present: {rec.tone is not None}")
        print(f"  pronunciation present: {rec.pronunciation is not None}")
        print(f"  errors: {rec.errors}")

    assert records[0].llm_eval.status == EvalStatus.COMPLETED
    assert records[1].llm_eval.status == EvalStatus.COMPLETED
    assert records[2].llm_eval.status == EvalStatus.COMPLETED
    print("\nAll assertions passed: LLM eval ran for every task type via the unified client.")
    print("(confidence/tone/pronunciation are None here only because optional model deps")
    print(" aren't installed in this sandbox — _IMPORTS_OK=False; in a full environment")
    print(" speaking/behavioral would also populate those fields per TASK_PIPELINES.)")

    # ---- producer/consumer concurrency smoke test ----
    print("\n--- Producer/Consumer queue smoke test ---")
    queue = asyncio.Queue(maxsize=10)
    results = []
    results_lock = asyncio.Lock()

    test_units = [
        AudioUnit(task_type=TaskType.WRITING, audio_path=None, text=f"essay {i}")
        for i in range(5)
    ]

    async def fake_producer():
        for u in test_units:
            await queue.put(u)
        for _ in range(2):
            await queue.put(None)

    consumers = [
        asyncio.create_task(pipeline_mod.consumer(i, queue, resources, results, results_lock))
        for i in range(2)
    ]
    await fake_producer()
    await queue.join()
    await asyncio.gather(*consumers)

    print(f"Processed {len(results)} units via 2 concurrent consumers (expected 5).")
    assert len(results) == 5
    print("Producer-consumer smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
