"""
pipeline.py — producer-consumer orchestration for the full evaluation system.
Architecture
------------
PRODUCER (1 task):
    - Pulls raw audio (or pre-supplied text, for writing tasks) from an
      input source (CLI args, directory watch, API uploads — pluggable).
    - Standardizes audio to 16kHz mono wav.
    - Runs the master ASR pass once per unit (speaking/behavioral only).
    - Pushes an AudioUnit (+ transcript) onto an asyncio.Queue.
    - DOES NOT store numpy audio_array in unit.metadata — the array is
      passed directly through process_unit() as a local variable to avoid
      heap bloat (7.5 MB per unit) and prevent dataclasses.asdict() from
      attempting to JSON-serialize it into Redis.
CONSUMERS (N tasks, N = settings.num_consumers):
    - Pull units off the queue.
    - Route to the correct sub-pipeline set based on TASK_PIPELINES:
        speaking    -> confidence + tone + pronunciation + LLM eval
        writing     -> LLM eval only
        behavioral  -> tone + confidence + LLM eval
    - All applicable sub-pipelines + the LLM eval call run CONCURRENTLY
      per unit (asyncio.gather), since they're independent of each other
      (they all read the same standardized audio / transcript).
    - Backend for LLM eval (Ollama dev / vLLM prod) is resolved once via
      config.settings and is transparent to this layer.
    - After all sub-pipelines complete, llm_score is computed from the
      LLM_eval weight matrices (calculate_llm_score), and overall_score
      is computed by combining llm_score with the acoustic sub-scores
      (pronunciation / tone / confidence) via calculate_overall_score.
Queue sentinel `None` signals consumers to stop once production is done
and the queue has drained.


"""
from __future__ import annotations
import asyncio
import os
import sys
import tempfile
import time
import traceback
from typing import Optional
import numpy as np

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(current_dir, "Pronunciation"))
sys.path.insert(0, os.path.join(current_dir, "confidence"))
sys.path.insert(0, os.path.join(current_dir, "tone"))
sys.path.insert(0, os.path.join(current_dir, "STT"))
sys.path.insert(0, os.path.dirname(current_dir))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(__file__))
# Ensure LLM_eval sub-packages are importable.
_LLM_EVAL_DIR = os.path.join(os.path.dirname(current_dir), "LLM_eval")
if _LLM_EVAL_DIR not in sys.path:
    sys.path.insert(0, _LLM_EVAL_DIR)

from config_integration.config import TaskType, TASK_PIPELINES, SYSTEM_PROMPTS, settings, technical_user_message
from models_integration.models import AudioUnit, PipelineRecord, EvalStatus
from core_integration.llm_client import LLMClient
from utils_integration.logger import get_logger
# LLM_eval scoring functions — the authoritative weight-matrix scorers.
from LLM_eval.config_llm.config import (
    calculate_llm_score,
    calculate_overall_score,
    SPEAKING_OVERALL_WEIGHTS,
    BEHAVIORAL_OVERALL_WEIGHTS,
)

logger = get_logger(__name__)

try:
    from confidence.app.services.interview_pipeline import run_pipeline as confidence_pipeline
    from tone.app.services.tone_service import ToneAnalyzer
    from Pronunciation.pipeline import PronunciationPipeline
    from Pronunciation.config_pron.config import PipelineConfig
    from STT.asr_stage import ASRStage
    from core_main.registry import IsolatedUnsafeLoad
    _IMPORTS_OK = True
except ImportError as e:
    logger.warning("[pipeline] Optional sub-pipeline import failed (will run in degraded mode): %s", e)
    _IMPORTS_OK = False

try:
    _TECH_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "technical_eval")
    if _TECH_EVAL_DIR not in sys.path:
        sys.path.insert(0, _TECH_EVAL_DIR)
    # Import BOTH sync and async versions — async is preferred (Group 4, Fix 1)
    from technical_eval.core_tech.semantic_eval import get_semantic_score, get_semantic_score_async
    _SEMANTIC_OK = True
except ImportError as e:
    logger.warning("[pipeline] Semantic eval import failed (will skip semantic scoring): %s", e)
    _SEMANTIC_OK = False
    get_semantic_score = None
    get_semantic_score_async = None

SENTINEL = None  # signals a consumer to shut down


# ---------------------------------------------------------------------------
# Shared, lazily-initialized heavy resources (ASR model, tone analyzer, etc.)
# Loaded once and reused across all units rather than per-unit.
# ---------------------------------------------------------------------------
class SharedResources:
    def __init__(self):
        self.asr: Optional["ASRStage"] = None
        self.tone_analyzer: Optional["ToneAnalyzer"] = None
        self.pronunciation_pipeline: Optional["PronunciationPipeline"] = None
        self.pipeline_cfg = PipelineConfig() if _IMPORTS_OK else None
        self.llm_client: Optional[LLMClient] = None
        # GROUP 4 FIX 2: Lock created lazily inside init(), NOT here.
        # asyncio.Lock() must be created on the running event loop.
        # Creating it in __init__ (which may run outside any event loop,
        # e.g. during Celery worker startup) binds it to the wrong loop
        # and causes RuntimeError: "Future attached to a different loop".
        self._lock: Optional[asyncio.Lock] = None

    async def init(self):
        # Create lock lazily — guaranteed to be on the running event loop.
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self.llm_client is None:
                self.llm_client = await LLMClient.create()
            if not _IMPORTS_OK:
                return
            if self.asr is None:
                self.asr = ASRStage(self.pipeline_cfg.whisper)
                with IsolatedUnsafeLoad():
                    await asyncio.to_thread(self.asr.load)
            if self.tone_analyzer is None:
                self.tone_analyzer = ToneAnalyzer()
            if self.pronunciation_pipeline is None:
                # PronunciationPipeline no longer loads its own ASRStage (Group 1 fix).
                # Only G2P + Wav2Vec2 are loaded here — Whisper lives in self.asr.
                self.pronunciation_pipeline = PronunciationPipeline(self.pipeline_cfg)
                with IsolatedUnsafeLoad():
                    await asyncio.to_thread(self.pronunciation_pipeline.load)

    async def close(self):
        if self.llm_client:
            await self.llm_client.close()


# ---------------------------------------------------------------------------
# PRODUCER
# ---------------------------------------------------------------------------
async def producer(
    queue: "asyncio.Queue",
    units: list[AudioUnit],
    resources: SharedResources,
):
    """
    Standardizes audio + runs master ASR for audio-bearing units, then
    enqueues each unit (mutated in place with .text set to the transcript).

    GROUP 1 FIX: The numpy audio_array is NOT stored in unit.metadata.
    It is passed as a separate tuple element so process_unit() can forward
    it directly to _run_pronunciation() without it ever entering a dict
    that could be serialized by dataclasses.asdict() into Redis.

    Writing-only units (audio_path is None, text already supplied) skip
    ASR entirely and are enqueued as (unit, None, None).
    """
    import librosa
    import soundfile as sf

    for unit in units:
        try:
            if unit.audio_path is None:
                # text-only (writing) unit — nothing to transcribe
                await queue.put((unit, None, None))
                continue

            logger.info("[producer] Loading audio for unit %s (%s)", unit.unit_id, unit.audio_path)
            audio_array, _ = await asyncio.to_thread(
                librosa.load, unit.audio_path, sr=16000, mono=True
            )

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            standardized_path = tmp.name
            tmp.close()
            await asyncio.to_thread(sf.write, standardized_path, audio_array, 16000)

            # Store ONLY the path in metadata — never the numpy array.
            unit.metadata["standardized_audio_path"] = standardized_path

            asr_result = None
            if _IMPORTS_OK:
                asr_result = await asyncio.to_thread(resources.asr.process, audio_array, 16000)
                unit.text = asr_result.raw_text
                # asr_result is a lightweight dataclass — safe to store in metadata.
                unit.metadata["asr_result"] = asr_result
            else:
                unit.text = unit.text or ""

            logger.info("[producer] Unit %s transcribed, enqueueing.", unit.unit_id)
            # audio_array passed as tuple element — NOT in metadata.
            await queue.put((unit, audio_array, asr_result))

        except Exception:
            logger.exception("[producer] Failed to prepare unit %s; skipping.", unit.unit_id)

    # Signal all consumers to stop.
    for _ in range(settings.num_consumers):
        await queue.put(SENTINEL)


# ---------------------------------------------------------------------------
# CONSUMER — per-unit sub-pipeline fan-out
# ---------------------------------------------------------------------------
async def _run_confidence(unit: AudioUnit, resources: SharedResources) -> Optional[dict]:
    path = unit.metadata.get("standardized_audio_path")
    if not path:
        return None
    try:
        return await asyncio.to_thread(confidence_pipeline, path, unit.text)
    except Exception as e:
        logger.exception("[consumer] confidence pipeline failed for unit %s", unit.unit_id)
        return None


async def _run_tone(unit: AudioUnit, resources: SharedResources) -> Optional[dict]:
    path = unit.metadata.get("standardized_audio_path")
    if not path:
        return None
    try:
        return await resources.tone_analyzer.analyze(path)
    except Exception as e:
        logger.exception("[consumer] tone pipeline failed for unit %s", unit.unit_id)
        return None


async def _run_pronunciation(
    unit: AudioUnit,
    resources: SharedResources,
    audio_array: Optional[np.ndarray],
    asr_result,
) -> Optional[object]:
    """
    Run pronunciation pipeline.

    audio_array and asr_result are passed as explicit arguments — NOT
    read from unit.metadata — to prevent numpy arrays from entering the
    metadata dict that gets serialized by dataclasses.asdict() into Redis.
    """
    if audio_array is None or asr_result is None:
        return None
    try:
        final_reference = unit.reference_text or unit.text

        def run_eval():
            return resources.pronunciation_pipeline.evaluate(
                audio=audio_array,
                sample_rate=16000,
                asr_result=asr_result,
                reference_text=final_reference,
                session_id=unit.unit_id,
            )

        return await asyncio.to_thread(run_eval)
    except Exception as e:
        logger.exception("[consumer] pronunciation pipeline failed for unit %s", unit.unit_id)
        return None


async def _run_semantic_eval(unit: AudioUnit, resources: SharedResources) -> Optional[dict]:
    """
    Runs semantic scoring for a technical evaluation unit.

    GROUP 4 FIX 1: Calls get_semantic_score_async() directly instead of
    wrapping the blocking sync version in asyncio.to_thread().
    get_semantic_score_async() uses asyncio.gather() to fetch both
    embeddings (model_answer + candidate answer) concurrently,
    cutting embedding latency roughly in half.

    Scoring change: compares candidate answer against model_answer
    (from the question bank) using pure embedding similarity.
    The key score is `similarity_score` (0-100), which is also what
    _semantic_score_from_record() and calculate_overall_technical_score()
    consume.  The old keyword/vocab/length sub-scores are still returned
    in the result dict for diagnostics but are NOT used in the overall score.
    """
    if not _SEMANTIC_OK or get_semantic_score_async is None:
        logger.warning("[consumer] semantic eval skipped — module not available")
        return None

    question_data = unit.metadata.get("question", {})
    question_text  = question_data.get("question", unit.topic)
    model_answer   = question_data.get("model_answer", "")   # primary comparison target
    answer = unit.text or ""

    if not answer.strip():
        logger.warning("[consumer] semantic eval skipped — empty transcript for unit %s", unit.unit_id)
        return None

    try:
        # Pass model_answer so the embedding comparison is candidate vs. ideal
        # reference answer, not candidate vs. question text or keyword list.
        return await get_semantic_score_async(
            question_text,
            answer,
            expected_keywords=None,   # no keyword list in new schema
            domain="general",          # domain vocab scoring not used for technical
            model_answer=model_answer,
        )
    except Exception as e:
        logger.exception("[consumer] semantic eval failed for unit %s", unit.unit_id)
        return None


async def _run_llm_eval(unit: AudioUnit, resources: SharedResources):
    if unit.task_type == TaskType.TECHNICAL:
        question_data = unit.metadata.get("question", {})

        # ── Transcribe follow-up audio if present ──────────────────────────
        follow_up_text = unit.metadata.get("follow_up_text", "")
        followup_audio_path = unit.metadata.get("followup_audio_path")
        if followup_audio_path and not follow_up_text and _IMPORTS_OK and resources.asr is not None:
            try:
                import librosa as _librosa  # noqa: PLC0415
                fu_array, _ = await asyncio.to_thread(
                    _librosa.load, followup_audio_path, sr=16000, mono=True
                )
                fu_asr = await asyncio.to_thread(resources.asr.process, fu_array, 16000)
                follow_up_text = fu_asr.raw_text or ""
                unit.metadata["follow_up_text"] = follow_up_text
                logger.debug(
                    "[consumer] follow-up audio transcribed unit=%s chars=%d",
                    unit.unit_id, len(follow_up_text),
                )
            except Exception as _fu_err:
                logger.warning(
                    "[consumer] follow-up audio transcription failed unit=%s error=%s",
                    unit.unit_id, _fu_err,
                )

        # ── Build user message with full question-bank schema ──────────────
        user_message = technical_user_message(
            role_family=question_data.get("role_family", ""),
            topic=question_data.get("topic", unit.topic),
            level_band=question_data.get("level_band", ""),
            question=question_data.get("question", unit.topic),
            model_answer=question_data.get("model_answer", ""),
            evaluation_rubric=question_data.get("evaluation_rubric", ""),
            red_flags=question_data.get("red_flags", ""),
            candidate_answer=unit.text or "",
            follow_up_question=question_data.get("follow_up", ""),
            follow_up_answer=follow_up_text,
        )
    else:
        user_message = unit.text or ""

    system_prompt = SYSTEM_PROMPTS[unit.task_type.value]
    try:
        return await resources.llm_client.infer(
            system_prompt=system_prompt,
            user_message=user_message,
            task_type=unit.task_type,
            request_id=unit.unit_id,
        )
    except Exception as e:
        logger.exception("[consumer] LLM eval failed for unit %s", unit.unit_id)
        return _make_failed_eval_result(unit, str(e))


def _make_failed_eval_result(unit: AudioUnit, error: str):
    from models_integration.models import EvalResult
    return EvalResult(
        request_id=unit.unit_id, task_type=unit.task_type,
        status=EvalStatus.FAILED, error=error,
    )


# ---------------------------------------------------------------------------
# Score extraction helpers
# ---------------------------------------------------------------------------
def _extract_confidence_score(confidence: Optional[dict]) -> Optional[float]:
    """Pull the numeric confidence score out of the confidence pipeline result."""
    if not confidence:
        return None
    raw = confidence.get("confidence_data", {}).get("final_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_tone_score(tone: Optional[dict]) -> Optional[float]:
    """Pull the numeric tone score out of the tone pipeline result."""
    if not tone:
        return None
    raw = tone.get("tone_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_pronunciation_score(pronunciation) -> Optional[float]:
    """Pull the numeric pronunciation score out of the pronunciation pipeline result."""
    if pronunciation is None:
        return None
    raw = getattr(pronunciation, "overall_score", None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Partial-weight normalization helper (Group 4, Fix 3)
# ---------------------------------------------------------------------------
def _normalized_score(scores: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    """
    Compute a weighted average using only the available (non-None) scores.

    When a sub-pipeline fails and its score is None, its weight is
    redistributed proportionally across the remaining present scores.
    This means a transient failure in one sub-pipeline (e.g. pronunciation)
    still produces a meaningful overall_score rather than returning None.

    Args:
        scores:  {score_name: value_or_None}  — values are 0-100.
        weights: {score_name: weight}          — must match keys in scores.

    Returns:
        Normalized weighted score (0-100) if at least one score is present,
        None if ALL scores are None (total pipeline failure).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    missing = []

    for name, val in scores.items():
        w = weights.get(name, 0.0)
        if val is not None:
            weighted_sum += val * w
            total_weight += w
        else:
            missing.append(name)

    if total_weight == 0.0:
        return None  # every sub-pipeline failed

    if missing:
        logger.warning(
            "overall_score_partial_normalization missing=%s "
            "available_weight=%.2f — score computed from available sub-scores only.",
            missing, total_weight,
        )

    # Normalize: divide by the weight actually used, not 1.0
    return round(weighted_sum / total_weight, 2)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def _compute_scores(record: PipelineRecord) -> None:
    """
    Compute record.llm_score and record.overall_score in-place using the
    LLM_eval weight matrices and overall-score functions.

    Technical evaluation fallback logic:
      - If LLM eval failed (status != COMPLETED) → overall_score = semantic_score
      - If hallucination_flag is True → overall_score = semantic_score
      - Otherwise → overall_score = llm_score * 0.70 + semantic_score * 0.30
      Semantic score is sourced from record.semantic_eval["combined_score"] (0-100).

    Speaking / behavioral fallback logic (Group 4, Fix 3):
      - If any acoustic sub-score is None, partial-weight normalization is
        applied using the scores that ARE available.
      - overall_score is None only if ALL sub-scores are None.
    """
    task_type = record.task_type

    def _semantic_score_from_record() -> Optional[float]:
        if not record.semantic_eval:
            return None
        raw = record.semantic_eval.get("combined_score")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    # -----------------------------------------------------------------------
    # TECHNICAL: special routing — hallucination fallback
    # -----------------------------------------------------------------------
    if task_type.value == "technical_evaluation":
        semantic_score = _semantic_score_from_record()
        llm_failed = (
            record.llm_eval is None
            or record.llm_eval.status != EvalStatus.COMPLETED
        )
        hallucinated = False
        if not llm_failed and record.llm_eval and record.llm_eval.result:
            hallucinated = bool(record.llm_eval.result.get("hallucination_flag"))

        if llm_failed or hallucinated:
            reason = "hallucination_flagged" if hallucinated else "llm_eval_failed"
            logger.warning(
                "technical_overall_score_fallback unit=%s reason=%s",
                record.unit_id, reason,
            )
            record.llm_score = None
            record.overall_score = semantic_score
            return

        parsed = record.llm_eval.result or {}
        try:
            from LLM_eval.models_llm.models import TaskType as LLMTaskType
            llm_task_type = LLMTaskType(task_type.value)
        except Exception:
            llm_task_type = task_type
        try:
            record.llm_score = calculate_llm_score(llm_task_type, parsed)
        except Exception as e:
            logger.warning("llm_score_computation_failed unit=%s error=%s", record.unit_id, e)
            record.llm_score = None
            record.overall_score = semantic_score
            return

        if semantic_score is None:
            logger.warning(
                "technical_overall_score_skipped unit=%s reason=missing_semantic_score",
                record.unit_id,
            )
            record.overall_score = None
        else:
            try:
                record.overall_score = calculate_overall_score(
                    llm_task_type, record.llm_score, semantic_score=semantic_score
                )
            except Exception as e:
                logger.warning("overall_score_computation_failed unit=%s error=%s", record.unit_id, e)
                record.overall_score = None
        return

    # -----------------------------------------------------------------------
    # All other task types
    # -----------------------------------------------------------------------
    if record.llm_eval is None or record.llm_eval.status != EvalStatus.COMPLETED:
        return

    parsed = record.llm_eval.result or {}
    try:
        from LLM_eval.models_llm.models import TaskType as LLMTaskType
        llm_task_type = LLMTaskType(task_type.value)
    except Exception:
        llm_task_type = task_type

    try:
        record.llm_score = calculate_llm_score(llm_task_type, parsed)
    except Exception as e:
        logger.warning("llm_score_computation_failed unit=%s error=%s", record.unit_id, e)
        record.llm_score = None
        return

    confidence_score = _extract_confidence_score(record.confidence)
    tone_score = _extract_tone_score(record.tone)
    pronunciation_score = _extract_pronunciation_score(record.pronunciation)

    try:
        if task_type.value == "writing_evaluation":
            record.overall_score = calculate_overall_score(
                llm_task_type, record.llm_score
            )

        elif task_type.value == "speaking_evaluation":
            # GROUP 4 FIX 3: partial-weight normalization instead of returning None.
            # If pronunciation/tone/confidence pipeline failed, redistribute its
            # weight across the scores that did succeed.
            all_present = None not in (pronunciation_score, tone_score, confidence_score)

            if all_present:
                # Fast path — all scores available, use the standard function.
                record.overall_score = calculate_overall_score(
                    llm_task_type, record.llm_score,
                    pronunciation_score=pronunciation_score,
                    tone_score=tone_score,
                    confidence_score=confidence_score,
                )
            else:
                # Slow path — at least one score missing, normalize available weights.
                record.overall_score = _normalized_score(
                    scores={
                        "llm_score":           record.llm_score,
                        "pronunciation_score": pronunciation_score,
                        "confidence_score":    confidence_score,
                        "tone_score":          tone_score,
                    },
                    weights=SPEAKING_OVERALL_WEIGHTS,
                )

        elif task_type.value == "behavioral_evaluation":
            # GROUP 4 FIX 3: same partial normalization for behavioral.
            all_present = None not in (tone_score, confidence_score)

            if all_present:
                record.overall_score = calculate_overall_score(
                    llm_task_type, record.llm_score,
                    tone_score=tone_score,
                    confidence_score=confidence_score,
                )
            else:
                record.overall_score = _normalized_score(
                    scores={
                        "llm_score":        record.llm_score,
                        "confidence_score": confidence_score,
                        "tone_score":       tone_score,
                    },
                    weights=BEHAVIORAL_OVERALL_WEIGHTS,
                )

    except Exception as e:
        logger.warning("overall_score_computation_failed unit=%s error=%s", record.unit_id, e)
        record.overall_score = None


# ---------------------------------------------------------------------------
# Core unit processor
# ---------------------------------------------------------------------------
async def process_unit(
    unit: AudioUnit,
    resources: SharedResources,
    audio_array: Optional[np.ndarray],
    asr_result,
) -> PipelineRecord:
    """
    Runs every sub-pipeline applicable to unit.task_type concurrently,
    then assembles a single PipelineRecord and computes the final scores.

    audio_array and asr_result are explicit parameters — NOT read from
    unit.metadata — to prevent numpy arrays from ever entering the metadata
    dict that gets serialized by dataclasses.asdict() into Redis.

    GROUP 4 FIX 4: Temp file cleanup is in a finally block so the
    standardized .wav is always deleted even when asyncio.gather raises
    an unhandled exception before the cleanup code would normally run.
    """
    t0 = time.monotonic()
    active = TASK_PIPELINES[unit.task_type]
    record = PipelineRecord(unit_id=unit.unit_id, task_type=unit.task_type, transcript=unit.text or "")
    std_path = unit.metadata.get("standardized_audio_path")

    try:
        coros = {}
        if "confidence" in active and _IMPORTS_OK:
            coros["confidence"] = _run_confidence(unit, resources)
        if "tone" in active and _IMPORTS_OK:
            coros["tone"] = _run_tone(unit, resources)
        if "pronunciation" in active and _IMPORTS_OK:
            # audio_array and asr_result passed explicitly — not from metadata.
            coros["pronunciation"] = _run_pronunciation(unit, resources, audio_array, asr_result)
        if "llm" in active:
            coros["llm"] = _run_llm_eval(unit, resources)
        if "semantic" in active:
            coros["semantic"] = _run_semantic_eval(unit, resources)

        keys = list(coros.keys())
        results = await asyncio.gather(*coros.values(), return_exceptions=True)

        for key, res in zip(keys, results):
            if isinstance(res, Exception):
                record.errors[key] = str(res)
                continue
            if key == "confidence":
                record.confidence = res
            elif key == "tone":
                record.tone = res
            elif key == "pronunciation":
                record.pronunciation = res
            elif key == "llm":
                record.llm_eval = res
            elif key == "semantic":
                record.semantic_eval = res

        _compute_scores(record)

    finally:
        # GROUP 4 FIX 4: always clean up the standardized .wav, even on exception.
        if std_path and os.path.exists(std_path):
            try:
                os.remove(std_path)
            except Exception:
                pass
        # audio_array goes out of scope here — GC collects it.
        # No explicit del needed since it was never stored in metadata.

    record.total_latency_ms = (time.monotonic() - t0) * 1000
    return record


async def consumer(
    consumer_id: int,
    queue: "asyncio.Queue",
    resources: SharedResources,
    results: list[PipelineRecord],
    results_lock: asyncio.Lock,
):
    while True:
        item = await queue.get()
        if item is SENTINEL:
            queue.task_done()
            break

        # Unpack the tuple produced by producer().
        # audio_array is a local variable — never stored in metadata.
        unit, audio_array, asr_result = item

        try:
            logger.info(
                "[consumer-%s] Processing unit %s (task_type=%s)",
                consumer_id,
                unit.unit_id,
                unit.task_type.value,
            )
            record = await process_unit(unit, resources, audio_array, asr_result)
            async with results_lock:
                results.append(record)
            logger.info(
                "[consumer-%s] Completed unit %s in %.0fms",
                consumer_id,
                unit.unit_id,
                record.total_latency_ms,
            )
        except Exception:
            logger.exception("[consumer-%s] Unhandled error on unit %s", consumer_id, unit.unit_id)
        finally:
            queue.task_done()
            # Release audio_array immediately so GC can reclaim the
            # 7.5 MB numpy buffer rather than waiting for the next cycle.
            audio_array = None


# ---------------------------------------------------------------------------
# Orchestration entrypoint
# ---------------------------------------------------------------------------
async def run_evaluation_pipeline(units: list[AudioUnit]) -> list[PipelineRecord]:
    """
    Spins up the producer + N consumers, runs them to completion, and
    returns all PipelineRecords. Safe to call repeatedly from an API layer.
    """
    logger.info("[pipeline] backend=%s consumers=%s", settings.backend.value, settings.num_consumers)
    resources = SharedResources()
    await resources.init()

    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.queue_max_size)
    results: list[PipelineRecord] = []
    results_lock = asyncio.Lock()

    try:
        producer_task = asyncio.create_task(producer(queue, units, resources))
        consumer_tasks = [
            asyncio.create_task(consumer(i, queue, resources, results, results_lock))
            for i in range(settings.num_consumers)
        ]
        await producer_task
        await queue.join()
        await asyncio.gather(*consumer_tasks)
        return results
    finally:
        await resources.close()