"""
run_ollama.py — Test the LLM Evaluation Service locally using Ollama on Windows.

Ollama serves an OpenAI-compatible endpoint at http://localhost:11434/v1
so no code changes are needed — we just point the client there.

Prerequisites
─────────────
1. Install Ollama:  https://ollama.com/download  (Windows installer)
2. Pull a model (pick one based on your RAM):

   ollama pull mistral          # 7B  ~4 GB RAM  (recommended)
   ollama pull llama3.2         # 3B  ~2 GB RAM  (lighter)
   ollama pull phi3             # 3.8B ~2.5 GB  (very fast)
   ollama pull qwen2.5:7b       # 7B  ~4 GB RAM  (good at JSON)

3. Ollama starts automatically after install. Verify:
   curl http://localhost:11434/api/tags

4. Run this file:
   python run_ollama.py                       # runs all 3 task types
   python run_ollama.py --model phi3          # use a different model
   python run_ollama.py --task speaking       # single task only
   python run_ollama.py --stress              # batching stress test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx

from config_llm.config import (
    SYSTEM_PROMPTS,
    behavioral_user_message,
    speaking_user_message,
    writing_user_message,
)
from models_llm.models import EvalRequest, EvalResult, EvalStatus, TaskType
from core_llm.response_parser import (
    parse_behavioral_eval,
    parse_speaking_response,
    parse_writing_response
)

# ─────────────────────────────────────────────────────────────────────────────
# Ollama settings
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL   = "qwen2.5:0.5b-instruct"       # change to whatever you pulled
TEMPERATURE     = 0.0
MAX_TOKENS      = 2048

# Maps each TaskType to its dedicated parser (same mapping as core/llm_client.py)
PARSERS = {
    TaskType.SPEAKING: parse_speaking_response,
    TaskType.WRITING: parse_writing_response,
    TaskType.BEHAVIORAL: parse_behavioral_eval,
}

# ─────────────────────────────────────────────────────────────────────────────
# Sample candidate evaluation requests — one per task type
# ─────────────────────────────────────────────────────────────────────────────

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
                "focus better on deep work. Secondly, it widens the talent pool. "
                "Companies are no longer restricted to hiring people who live near "
                "their offices, which means they can access specialists from anywhere "
                "in the world.\n\n"
                "However, full-time remote work also has challenges. Collaboration and "
                "spontaneous brainstorming can suffer when teams never meet in person. "
                "New employees may also find it harder to absorb company culture "
                "remotely.\n\n"
                "In conclusion, I believe a hybrid approach often works best, but for "
                "roles that do not require frequent in-person collaboration, full-time "
                "remote work should absolutely be an option. Companies that offer this "
                "flexibility will likely attract and retain better talent in the long run."
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
                "with him and walked through the three mobile screens that would break, "
                "showing the actual JSON our app expected. He hadn't realised the mobile "
                "team was relying on that structure, so we agreed on a versioned "
                "endpoint instead — keeping the old format for v1 and using his "
                "preferred structure for v2. Looking back, I think I should have looped "
                "in the mobile lead earlier instead of trying to resolve it just between "
                "the two of us, since it affected their roadmap too."
            ),
        ),
        metadata={"candidate_id": "cand_003"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client (thin — mirrors core/llm_client.py)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Lightweight client that hits Ollama's OpenAI-compatible endpoint.
    Identical wire format to what LLMClient uses against vLLM, and uses the
    same SYSTEM_PROMPTS catalogue and per-task parsers from utils/parsers.py.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._http = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def check_connection(self) -> bool:
        try:
            r = await self._http.get("/models", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            r = await self._http.get("/models")
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    async def infer(self, request: EvalRequest) -> EvalResult:
        system_prompt = SYSTEM_PROMPTS[request.task_type.value]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_message},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }

        t0 = time.monotonic()
        try:
            response = await self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.monotonic() - t0) * 1000
            return _parse(request, data, latency_ms)

        except httpx.HTTPStatusError as exc:
            return EvalResult(
                request_id=request.request_id,
                task_type=request.task_type,
                status=EvalStatus.FAILED,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return EvalResult(
                request_id=request.request_id,
                task_type=request.task_type,
                status=EvalStatus.FAILED,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    async def infer_batch_concurrent(self, requests: list[EvalRequest]) -> list[EvalResult]:
        """Send all requests concurrently (simulates batching behaviour)."""
        return await asyncio.gather(*[self.infer(r) for r in requests])


# ─────────────────────────────────────────────────────────────────────────────
# Response parser — delegates to utils/parsers.py (same as core/llm_client.py)
# ─────────────────────────────────────────────────────────────────────────────

def _parse(request: EvalRequest, data: dict, latency_ms: float) -> EvalResult:
    choice = data["choices"][0]
    raw_text: str = choice["message"]["content"]
    usage = data.get("usage", {})

    parser_fn = PARSERS[request.task_type]
    parsed = parser_fn(raw_text)

    return EvalResult(
        request_id=request.request_id,
        task_type=request.task_type,
        status=EvalStatus.COMPLETED,
        result=parsed,
        raw_output=raw_text,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

TASK_EMOJI = {
    TaskType.SPEAKING:   "🗣️",
    TaskType.WRITING:    "✍️",
    TaskType.BEHAVIORAL: "",
}

def print_result(result: EvalResult, idx: int | None = None) -> None:
    emoji = TASK_EMOJI.get(result.task_type, "")
    label = f"[{idx}] " if idx is not None else ""
    print(f"\n{'─'*62}")
    print(f"  {emoji}  {label}{result.task_type.value}")
    print(f"  status   : {result.status.value}")
    print(f"  latency  : {result.latency_ms:.0f} ms")
    print(f"  tokens   : {result.prompt_tokens} prompt / {result.completion_tokens} completion")

    if result.status == EvalStatus.COMPLETED:
        if result.result and result.result.get("llm_eval"):
            print("  llm_eval :")
            for param, entry in result.result["llm_eval"].items():
                score = entry.get("score") if isinstance(entry, dict) else entry
                rationale = entry.get("rationale", "") if isinstance(entry, dict) else ""
                print(f"    {param:22s}: {score:>2}  — {rationale}")

            if "star_coverage" in result.result:
                print(f"  star_coverage : {result.result['star_coverage']}")
            if result.result.get("red_flags"):
                print(f"  red_flags     : {result.result['red_flags']}")
            if result.result.get("strengths"):
                print(f"  strengths     : {result.result['strengths']}")
            if result.result.get("improvements"):
                print(f"  improvements  : {result.result['improvements']}")

            print(f"  llm_summary   : {result.result.get('llm_summary')}")
            print(f"  llm_score_raw : {result.result.get('llm_score_raw')}  (computed from weight matrix)")

            print("-------------------------raw result-------------------------------")
            print(result.raw_output[:])
        else:
            print("    JSON parse failed. Raw LLM output:")
            print(f"  {result.raw_output[:]}")
    else:
        print(f"   error : {result.error}")


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_all(client: OllamaClient) -> None:
    """Run all 5 task types sequentially."""
    print(f"\n  Running {len(SAMPLE_REQUESTS)} evaluations sequentially…\n")
    total_t0 = time.monotonic()

    for i, req in enumerate(SAMPLE_REQUESTS, 1):
        print(f"  [{i}/{len(SAMPLE_REQUESTS)}] {req.task_type.value}… ", end="", flush=True)
        result = await client.infer(req)
        print(f"done ({result.latency_ms:.0f} ms)")
        print_result(result, i)

    print(f"\n{'═'*62}")
    print(f"    All done in {(time.monotonic()-total_t0):.1f}s")


async def run_single_task(client: OllamaClient, task_name: str) -> None:
    """Run only the specified task type."""
    try:
        task_type = TaskType(f"{task_name}_evaluation")
    except ValueError:
        names = [t.value.replace("_evaluation", "") for t in TaskType]
        print(f"  Unknown task '{task_name}'. Choose from: {', '.join(names)}")
        sys.exit(1)

    req = next(r for r in SAMPLE_REQUESTS if r.task_type == task_type)
    print(f"\n  Running single task: {task_type.value}…")
    result = await client.infer(req)
    print_result(result)


async def run_concurrent_stress(client: OllamaClient) -> None:
    """
    Fire all 5 requests concurrently — simulates what BatchManager does
    when it flushes a batch to the Celery worker.
    """
    print(f"\n Concurrent batch test — sending {len(SAMPLE_REQUESTS)} requests at once…")
    t0 = time.monotonic()
    results = await client.infer_batch_concurrent(SAMPLE_REQUESTS)
    elapsed = time.monotonic() - t0

    for i, result in enumerate(results, 1):
        print_result(result, i)

    success = sum(1 for r in results if r.status == EvalStatus.COMPLETED)
    print(f"\n{'═'*62}")
    print(f"    {success}/{len(results)} succeeded in {elapsed:.1f}s total")
    avg = sum(r.latency_ms for r in results) / len(results)
    print(f"  Avg latency per request: {avg:.0f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    model = args.model
    client = OllamaClient(model=model)

    print(f"\n{'═'*62}")
    print(f"  LLM Evaluation Service — Ollama Local Test")
    print(f"  endpoint : {OLLAMA_BASE_URL}")
    print(f"  model    : {model}")
    print(f"{'═'*62}")

    # ── connectivity check ───────────────────────────────────────────
    print("\n  Checking Ollama connection… ", end="", flush=True)
    ok = await client.check_connection()
    if not ok:
        print("FAILED")
        print("\n  Cannot reach Ollama at http://localhost:11434")
        print("  Make sure Ollama is running (it auto-starts on Windows after install).")
        print("  Or start it manually:  ollama serve")
        await client.close()
        sys.exit(1)
    print("OK ✓")

    # ── show available models ────────────────────────────────────────
    models = await client.list_models()
    if models:
        print(f"  Available models: {', '.join(models)}")
    if model not in models and models:
        print(f"\n  '{model}' not found. Pull it with:  ollama pull {model}")
        print(f"  Or pass a model you have:  python run_ollama.py --model {models[0]}")
        await client.close()
        sys.exit(1)

    # ── dispatch to selected runner ──────────────────────────────────
    if args.stress:
        await run_concurrent_stress(client)
    elif args.task:
        await run_single_task(client, args.task)
    else:
        await run_all(client)

    await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LLM Evaluation Service locally with Ollama"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Ollama model name (default: mistral). Must be already pulled.",
    )
    parser.add_argument(
        "--task",
        choices=["speaking", "writing", "behavioral"],
        help="Run a single task type instead of all five.",
    )
    parser.add_argument(
        "--stress", action="store_true",
        help="Send all 5 requests concurrently (simulates batch flush).",
    )
    args = parser.parse_args()
    asyncio.run(main(args))