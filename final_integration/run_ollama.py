"""
run_ollama.py — exercise the unified LLMClient against local Ollama.

This no longer duplicates the HTTP/parsing logic — it forces
LLM_BACKEND=ollama and drives the same LLMClient used by the full
producer-consumer pipeline, so dev testing matches production behavior
exactly except for the endpoint/model.

Usage:
    python run_ollama.py                       # all 3 task types
    python run_ollama.py --model phi3
    python run_ollama.py --task speaking
    python run_ollama.py --stress              # concurrent batch test
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Force Ollama backend before config.settings is constructed.
os.environ["LLM_BACKEND"] = "ollama"

import httpx

from config_integration.config import (
    settings,
    SYSTEM_PROMPTS,
    behavioral_user_message,
    speaking_user_message,
    writing_user_message,
)
from models_integration.models import EvalRequest, EvalStatus, TaskType
from core_integration.llm_client import LLMClient

SAMPLE_REQUESTS: list[EvalRequest] = [
    EvalRequest(
        task_type=TaskType.SPEAKING,
        user_message=speaking_user_message(
            topic="Tell us about yourself",
            duration="120",
            transcript=(
                "Sure! Right now, I'm a Digital Project Manager at a mid-sized agency, "
                "where I oversee website launches and software development sprints for "
                "tech clients. My day-to-day involves keeping cross-functional teams "
                "aligned and unblocking developer bottlenecks. Before this, I spent two "
                "years as a Project Coordinator. My proudest achievement there was "
                "migrating a legacy client portal to a cloud system, which we delivered "
                "two weeks ahead of schedule and 10% under budget. While I love my "
                "current agency work, I'm looking to transition to an in-house product "
                "team where I can deeply focus on one platform. I've been following your "
                "company's latest software updates, and I'd love to bring my agile "
                "delivery skills to this specific role."
            ),
        ),
        metadata={"candidate_id": "cand_001"},
    ),
    EvalRequest(
        task_type=TaskType.WRITING,
        user_message=writing_user_message(
            topic="Should companies allow employees to work remotely full-time?",
            word_count="231",
            transcript=(
                "Remote work has become one of the most debated topics in modern "
                "workplaces. In my opinion, companies should allow employees to work "
                "remotely full-time, provided certain conditions are met.\n\n"
                "Firstly, remote work increases productivity for many employees. "
                "Without the distractions of a commute or office chatter, people can "
                "focus better on deep work. Secondly, it widens the talent pool.\n\n"
                "However, full-time remote work also has challenges. Collaboration and "
                "spontaneous brainstorming can suffer when teams never meet in person.\n\n"
                "In conclusion, I believe a hybrid approach often works best, but for "
                "roles that do not require frequent in-person collaboration, full-time "
                "remote work should absolutely be an option."
            ),
        ),
        metadata={"candidate_id": "cand_002"},
    ),
    EvalRequest(
        task_type=TaskType.BEHAVIORAL,
        user_message=behavioral_user_message(
            topic="Tell me about a time you had a conflict with a teammate.",
            duration="95",
            transcript=(
                "During a sprint last year, I disagreed with a teammate about how we "
                "should structure our API responses. He wanted to nest everything under "
                "a single 'data' key, and I felt that would break our existing mobile "
                "clients. Instead of escalating it immediately, I scheduled a quick call "
                "with him and walked through the three mobile screens that would break. "
                "He hadn't realised the mobile team was relying on that structure, so we "
                "agreed on a versioned endpoint instead. Looking back, I think I should "
                "have looped in the mobile lead earlier."
            ),
        ),
        metadata={"candidate_id": "cand_003"},
    ),
]


async def check_connection() -> bool:
    try:
        async with httpx.AsyncClient(base_url=settings.base_url(), timeout=3.0) as c:
            r = await c.get("/models")
            return r.status_code == 200
    except Exception:
        return False


async def list_models() -> list[str]:
    try:
        async with httpx.AsyncClient(base_url=settings.base_url(), timeout=3.0) as c:
            r = await c.get("/models")
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def print_result(result, idx: int | None = None) -> None:
    label = f"[{idx}] " if idx is not None else ""
    print(f"\n{'─'*62}")
    print(f"  {label}{result.task_type.value}")
    print(f"  status   : {result.status.value}")
    print(f"  latency  : {result.latency_ms:.0f} ms")
    print(f"  tokens   : {result.prompt_tokens} prompt / {result.completion_tokens} completion")

    if result.status == EvalStatus.COMPLETED:
        r = result.result or {}
        if r.get("llm_eval"):
            print("  llm_eval :")
            for param, entry in r["llm_eval"].items():
                score = entry.get("score") if isinstance(entry, dict) else entry
                rationale = entry.get("rationale", "") if isinstance(entry, dict) else ""
                print(f"    {param:22s}: {score!s:>4}  — {rationale}")
            for key in ("star_coverage", "red_flags", "strengths", "improvements"):
                if r.get(key):
                    print(f"  {key:14s}: {r[key]}")
            print(f"  llm_summary   : {r.get('llm_summary')}")
            print(f"  llm_score_raw : {r.get('llm_score_raw')}  (weighted, 0-100)")
        else:
            print("    JSON parse failed. Raw LLM output:")
            print(f"  {result.raw_output}")
    else:
        print(f"   error : {result.error}")


async def run_all(client: LLMClient) -> None:
    print(f"\n  Running {len(SAMPLE_REQUESTS)} evaluations sequentially…\n")
    t0 = time.monotonic()
    for i, req in enumerate(SAMPLE_REQUESTS, 1):
        system_prompt = SYSTEM_PROMPTS[req.task_type.value]
        result = await client.infer(system_prompt, req.user_message, req.task_type, req.request_id)
        print_result(result, i)
    print(f"\n    All done in {(time.monotonic()-t0):.1f}s")


async def run_single_task(client: LLMClient, task_name: str) -> None:
    try:
        task_type = TaskType(f"{task_name}_evaluation")
    except ValueError:
        print(f"  Unknown task '{task_name}'")
        sys.exit(1)
    req = next(r for r in SAMPLE_REQUESTS if r.task_type == task_type)
    system_prompt = SYSTEM_PROMPTS[task_type.value]
    result = await client.infer(system_prompt, req.user_message, task_type, req.request_id)
    print_result(result)


async def run_concurrent_stress(client: LLMClient) -> None:
    print(f"\n  Concurrent batch test — {len(SAMPLE_REQUESTS)} requests at once…")
    t0 = time.monotonic()
    batched = [(r.request_id, r.user_message) for r in SAMPLE_REQUESTS]
    # group by task type since infer_batch takes one task_type per call
    results = []
    for req in SAMPLE_REQUESTS:
        system_prompt = SYSTEM_PROMPTS[req.task_type.value]
        results.append(client.infer(system_prompt, req.user_message, req.task_type, req.request_id))
    results = await asyncio.gather(*results)
    elapsed = time.monotonic() - t0
    for i, result in enumerate(results, 1):
        print_result(result, i)
    success = sum(1 for r in results if r.status == EvalStatus.COMPLETED)
    print(f"\n    {success}/{len(results)} succeeded in {elapsed:.1f}s total")


async def main(args: argparse.Namespace) -> None:
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
        settings.ollama.__dict__  # no-op; OllamaSettings reads env at construction

    print(f"\n  LLM Evaluation Service — Ollama Local Test")
    print(f"  endpoint : {settings.base_url()}")
    print(f"  model    : {settings.llm_params()['model']}")

    print("\n  Checking Ollama connection… ", end="", flush=True)
    if not await check_connection():
        print("FAILED")
        print("  Cannot reach Ollama at", settings.base_url())
        print("  Start it with: ollama serve")
        sys.exit(1)
    print("OK")

    models = await list_models()
    if models:
        print(f"  Available models: {', '.join(models)}")
    model = settings.llm_params()["model"]
    if models and model not in models:
        print(f"\n  '{model}' not found. Pull it with: ollama pull {model}")
        sys.exit(1)

    client = await LLMClient.create()
    try:
        if args.stress:
            await run_concurrent_stress(client)
        elif args.task:
            await run_single_task(client, args.task)
        else:
            await run_all(client)
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM Evaluation Service locally with Ollama")
    parser.add_argument("--model", default=None, help="Ollama model name (must be already pulled).")
    parser.add_argument("--task", choices=["speaking", "writing", "behavioral"])
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))
