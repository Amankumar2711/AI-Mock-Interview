"""
main.py — CLI entrypoint for the producer-consumer evaluation pipeline.
Usage
-----
  # dev (Ollama) — default if APP_ENV unset
  python main.py speaking sample.wav --topic "Tell us about yourself"
  # production (vLLM)
  APP_ENV=prod python main.py behavioral sample.wav --topic "Conflict with a teammate"
  # writing (text only, no audio)
  python main.py writing --text-file essay.txt --topic "Should companies allow remote work?"
  # technical evaluation (audio + questions JSON)
  python main.py technical answer1.wav answer2.wav --questions-json questions.json
  # multiple units at once (batched through the same queue)
  python main.py speaking a.wav b.wav c.wav --topic "Tell us about yourself"

Questions JSON format (for technical evaluation):
  [
    {
      "id": "q001",
      "domain": "python",
      "difficulty": "hard",
      "question": "Explain Python's GIL.",
      "expected_keywords": ["global interpreter lock", "thread", "concurrency"],
      "expected_answer": "The GIL is a mutex that protects ...",
      "follow_up": "How would you work around it?"
    }
  ]

Env vars
--------
  APP_ENV=dev|prod        selects ollama vs vllm (default: dev)
  LLM_BACKEND=ollama|vllm|openai   explicit override
  NUM_CONSUMERS=4         consumer pool size
  LLM_CONCURRENCY=8       max in-flight LLM requests
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from config_integration.config import TaskType
from models_integration.models import AudioUnit, EvalStatus, PipelineRecord
from core_integration.pipeline import run_evaluation_pipeline
def parse_args():
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument(
        "task_type",
        choices=["speaking", "writing", "behavioral", "technical"],
        help="Evaluation task type (determines which sub-pipelines run).",
    )
    parser.add_argument(
        "audio_paths", nargs="*", default=[],
        help="One or more audio file paths (omit for writing tasks if using --text-file).",
    )
    parser.add_argument("--topic", default="", help="Prompt/question/topic for the response.")
    parser.add_argument("--reference-text", default=None, help="Reference text for pronunciation scoring.")
    parser.add_argument("--text-file", default=None, help="Path to a text file (writing tasks).")
    parser.add_argument(
        "--questions-json", default=None,
        help=(
            "Path to a JSON file containing a list of question objects "
            "(required for technical evaluation). Each object must have: "
            "'question', 'expected_keywords', 'expected_answer'. "
            "Optional: 'id', 'domain', 'difficulty', 'follow_up'."
        ),
    )
    return parser.parse_args()
def _load_questions(path: str) -> list[dict]:
    """Load and validate the questions JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading questions JSON '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("questions JSON must be a top-level JSON array.", file=sys.stderr)
        sys.exit(1)
    for i, q in enumerate(data):
        if not isinstance(q, dict):
            print(f"Question at index {i} is not a JSON object.", file=sys.stderr)
            sys.exit(1)
        for required_field in ("question", "expected_keywords"):
            if required_field not in q:
                print(
                    f"Question at index {i} is missing required field '{required_field}'.",
                    file=sys.stderr,
                )
                sys.exit(1)
    return data
def build_units(args) -> list[AudioUnit]:
    task_type = TaskType(f"{args.task_type}_evaluation")
    units: list[AudioUnit] = []
    # ---- Writing: text-only path ----------------------------------------
    if task_type == TaskType.WRITING and not args.audio_paths:
        if not args.text_file:
            print("writing task requires either audio_paths or --text-file", file=sys.stderr)
            sys.exit(1)
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        units.append(AudioUnit(task_type=task_type, audio_path=None, text=text, topic=args.topic))
        return units
    # ---- Technical: audio + questions JSON -------------------------------
    if task_type == TaskType.TECHNICAL:
        if not args.audio_paths:
            print("technical task requires at least one audio path.", file=sys.stderr)
            sys.exit(1)
        if not args.questions_json:
            print(
                "technical task requires --questions-json pointing to a JSON file.",
                file=sys.stderr,
            )
            sys.exit(1)
        questions = _load_questions(args.questions_json)
        if len(questions) < len(args.audio_paths):
            print(
                f"Warning: {len(args.audio_paths)} audio files but only "
                f"{len(questions)} questions — extra audio files will be ignored.",
                file=sys.stderr,
            )
        for idx, path in enumerate(args.audio_paths):
            if idx >= len(questions):
                print(
                    f"Warning: no question for audio file '{path}' (index {idx}) — skipping.",
                    file=sys.stderr,
                )
                break
            q = questions[idx]
            unit = AudioUnit(
                task_type=task_type,
                audio_path=path,
                topic=q.get("question", ""),
            )
            # Store the full question object so pipeline.py can access all fields.
            unit.metadata["question"] = q
            units.append(unit)
        return units
    # ---- Speaking / Behavioral: audio paths ------------------------------
    if not args.audio_paths:
        print("at least one audio path is required for this task type", file=sys.stderr)
        sys.exit(1)
    for path in args.audio_paths:
        units.append(
            AudioUnit(
                task_type=task_type,
                audio_path=path,
                topic=args.topic,
                reference_text=args.reference_text,
            )
        )
    return units
# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------
def _fmt(val, suffix="") -> str:
    """Format a nullable numeric value for display."""
    if val is None:
        return "N/A"
    return f"{val}{suffix}"
def print_scoreboard(records: list[PipelineRecord]) -> None:
    """
    Print the final evaluation report for all processed units.
    Acoustic sub-scores  — from the confidence / tone / pronunciation pipelines.
    Semantic Score       — from the semantic_eval module (technical tasks only).
    LLM Score            — weighted combination of the LLM's per-parameter scores
                           using LLM_eval's weight matrices (calculate_llm_score).
    Overall Score        — final combined score integrating LLM + acoustic/semantic
                           using LLM_eval's calculate_overall_score().
                           For technical tasks with hallucination or LLM failure:
                           overall_score == semantic_score (fallback mode).
    """
    print("\n" + "=" * 70)
    print("FINAL EVALUATION REPORT")
    print("=" * 70)
    for rec in records:
        print(f"\nUnit : {rec.unit_id}")
        print(f"Task : {rec.task_type.value}")
        # Show question context for technical tasks
        if rec.task_type.value == "technical_evaluation":
            # Retrieve question data from the transcript or record metadata
            pass
        print("-" * 70)
        # ---- Acoustic sub-scores (speaking / behavioral) ----------------
        if rec.confidence:
            conf_score = rec.confidence.get("confidence_data", {}).get("final_score")
            print(f"  Confidence Score    : {_fmt(conf_score, '/100')}")
        if rec.tone:
            tone_score = rec.tone.get("tone_score")
            print(f"  Tone Score          : {_fmt(tone_score, '/100')}")
        if rec.pronunciation:
            pron_score = getattr(rec.pronunciation, "overall_score", None)
            print(f"  Pronunciation Score : {_fmt(pron_score, '/100')}")
        # ---- Semantic sub-scores (technical) ----------------------------
        if rec.semantic_eval:
            sem = rec.semantic_eval
            print(f"  Semantic Score      : {_fmt(sem.get('combined_score'), '/100')}")
            print(f"    Similarity        : {_fmt(sem.get('similarity_score'), '/100')}")
            print(f"    Keyword Coverage  : {_fmt(sem.get('keyword_coverage'))} "
                  f"({len(sem.get('keywords_found', []))} found, "
                  f"{len(sem.get('keywords_missing', []))} missing)")
            if sem.get("keywords_missing"):
                print(f"    Missing Keywords  : {', '.join(sem['keywords_missing'])}")
            print(f"    Word Count        : {sem.get('word_count', 'N/A')}")
            if sem.get("feedback"):
                print(f"    Semantic Feedback : {sem['feedback']}")
        # ---- LLM evaluation ---------------------------------------------
        if rec.llm_eval:
            if rec.llm_eval.status == EvalStatus.COMPLETED:
                parsed = rec.llm_eval.result or {}
                # Hallucination flag (technical only)
                if rec.task_type.value == "technical_evaluation":
                    h_flag = parsed.get("hallucination_flag")
                    flag_str = "⚠ YES — LLM score overridden by semantic score" if h_flag else "No"
                    print(f"  Hallucination Flag  : {flag_str}")
                # Per-parameter breakdown
                llm_eval_dict = parsed.get("llm_eval", {})
                if llm_eval_dict:
                    print("  LLM Parameter Scores:")
                    for param, entry in llm_eval_dict.items():
                        if isinstance(entry, dict):
                            score = entry.get("score", "N/A")
                            rationale = entry.get("rationale", "")
                            print(f"    {param:<26}: {score}/10  — {rationale}")
                        else:
                            print(f"    {param:<26}: {entry}/10")
                # Task-specific extras
                if rec.task_type.value == "technical_evaluation":
                    missed = parsed.get("missed_concepts", [])
                    if missed:
                        print(f"  Missed Concepts     : {', '.join(missed)}")
                if rec.task_type.value == "behavioral_evaluation":
                    star = parsed.get("star_coverage", {})
                    if star:
                        coverage_str = ", ".join(
                            f"{k}={'✓' if v else '✗'}" for k, v in star.items()
                        )
                        print(f"  STAR Coverage       : {coverage_str}")
                    red_flags = parsed.get("red_flags", [])
                    if red_flags:
                        print(f"  Red Flags           : {'; '.join(red_flags)}")
                if rec.task_type.value == "writing_evaluation":
                    strengths = parsed.get("strengths", [])
                    if strengths:
                        print(f"  Strengths           : {'; '.join(strengths)}")
                    improvements = parsed.get("improvements", [])
                    if improvements:
                        print(f"  Improvements        : {'; '.join(improvements)}")
                summary = parsed.get("llm_summary", "")
                if summary:
                    print(f"  LLM Summary         : {summary}")
            else:
                print(f"  LLM Eval            : FAILED ({rec.llm_eval.error})")
                if rec.task_type.value == "technical_evaluation":
                    print("  Scoring Mode        : SEMANTIC ONLY (LLM eval failed)")
        elif rec.task_type.value == "technical_evaluation":
            print("  LLM Eval            : Not run")
            print("  Scoring Mode        : SEMANTIC ONLY")
        # ---- Computed scores (from LLM_eval weight matrices) ------------
        print()
        if rec.task_type.value == "technical_evaluation":
            # Indicate if we're in fallback mode
            llm_failed = rec.llm_eval is None or rec.llm_eval.status != EvalStatus.COMPLETED
            hallucinated = (
                rec.llm_eval
                and rec.llm_eval.result
                and rec.llm_eval.result.get("hallucination_flag")
            )
            if llm_failed or hallucinated:
                print(f"  LLM Score           : N/A  (fallback mode active)")
                print(f"  Overall Score       : {_fmt(rec.overall_score, '/100')}"
                      "  ← semantic score only (LLM overridden)")
            else:
                print(f"  LLM Score           : {_fmt(rec.llm_score, '/100')}"
                      "  ← weighted from LLM parameter scores")
                print(f"  Overall Score       : {_fmt(rec.overall_score, '/100')}"
                      "  ← LLM (70%) + semantic (30%)")
        else:
            print(f"  LLM Score           : {_fmt(rec.llm_score, '/100')}"
                  "  ← weighted from LLM parameter scores")
            print(f"  Overall Score       : {_fmt(rec.overall_score, '/100')}"
                  "  ← LLM + acoustic combined")
        # ---- Errors & latency -------------------------------------------
        if rec.errors:
            print(f"  Sub-pipeline Errors : {rec.errors}")
        print(f"  Total Latency       : {rec.total_latency_ms:.0f} ms")
    print("\n" + "=" * 70)
async def main():
    args = parse_args()
    units = build_units(args)
    records = await run_evaluation_pipeline(units)
    print_scoreboard(records)
if __name__ == "__main__":
    asyncio.run(main())

