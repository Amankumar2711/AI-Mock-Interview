"""
Central configuration for the LLM Evaluation System.
Controls dev/prod modes, batching, system prompts, scoring weights,
and infrastructure settings.

NOTE: Only the "technical_evaluation" prompt, its user-message builder,
and its weight matrices were changed in this revision. speaking_evaluation,
writing_evaluation, and behavioral_evaluation are untouched.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from models_llm.models import TaskType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RunMode(str, Enum):
    DEV = "dev"          # low-quantization / cloud-hosted small model
    PROD = "production"  # full-precision vLLM cluster


class BackendType(str, Enum):
    VLLM = "vllm"          # local / self-hosted vLLM server
    OPENAI = "openai"      # OpenAI-compatible cloud endpoint (dev)
    ANTHROPIC = "anthropic"


# ---------------------------------------------------------------------------
# System-prompt catalogue
# Extend this section to add more task types without touching business logic.
#
# IMPORTANT: keys here must match TaskType enum *values* exactly, and must
# have a corresponding entry in WEIGHT_MATRICES below.
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {

    "speaking_evaluation": '''
ROLE: Expert English speaking assessor scoring an auto-transcribed spoken response.

INPUT: TOPIC, DURATION_SECONDS, TRANSCRIPT.

TASK: Score exactly 6 parameters, each 0-10 with a ≤20-word rationale.

RUBRIC: 9-10 exceptional/near-native | 7-8 good, minor gaps | 5-6 adequate, functional | 3-4 below average, frequent errors | 1-2 poor, major deficiencies | 0 no attempt/off-topic.

PARAMETERS:
- content_relevance: addresses topic directly/consistently; penalize off-topic.
- idea_development: ideas supported with examples/reasoning, not bare assertions.
- structure_coherence: clear opening, connected body, closure; penalize abrupt/random order.
- vocabulary_range: variety/appropriateness; penalize repetition.
- grammar_accuracy: tense, agreement, articles, sentence construction. Ignore filler-sound transcription artifacts.
- task_completion: substantive engagement for full duration; penalize running out of content, excessive repetition, or <200 words for a 2-min task.

RULES:
- Be strict and consistent; never inflate scores.
- Do not penalize accent, pronunciation, or pace (scored elsewhere).
- Empty response → poor scores.
- Output ONLY valid JSON, no markdown/backticks/extra text.

OUTPUT FORMAT:
{
  "llm_eval": {
    "content_relevance":{"score":,"rationale":""},
    "idea_development":{"score":,"rationale":""},
    "structure_coherence":{"score":,"rationale":""},
    "vocabulary_range":{"score":,"rationale":""},
    "grammar_accuracy":{"score":,"rationale":""},
    "task_completion":{"score":,"rationale":""}
  },
  "llm_summary": "<2-3 sentence overall assessment>",
  "llm_score_raw":
}
''',

    "writing_evaluation": '''
ROLE: Expert English writing assessor.

INPUT: TOPIC, WORD_COUNT, RESPONSE.

TASK: Score exactly 8 parameters, each 0-10 with a ≤20-word rationale.

RUBRIC: 9-10 exceptional | 7-8 good, minor gaps | 5-6 adequate | 3-4 below average | 1-2 poor | 0 no attempt/off-topic/plagiarism indicators.

PARAMETERS:
- content_relevance: direct, full focus on topic; penalize vague padding/off-topic.
- argument_quality: claims need specific examples/reasoning/evidence; do not reward unsupported opinion. Hold to high standard (writer had time to think).
- structure_coherence: clear intro, linked body paragraphs, conclusion; penalize random ordering/no structure.
- vocabulary_range: variety, precision, register fit; penalize repetition, vague filler ("things","stuff"), register mismatch.
- grammar_accuracy: tense, agreement, articles, prepositions, sentence construction — full written standard, no excuses.
- sentence_variety: mix of simple/compound/complex sentences; penalize monotonous structure.
- mechanic_accuracy: punctuation, capitalization, spelling, paragraphing.
- task_completion: target 200-300 words. <150 words → score 0-2 regardless of quality; 150-199 → cap at 6; 200-300 → score on merit; >350 → deduct 1 for padding. Penalize word count met via repetition/filler.

RULES:
- Be strict/consistent; don't inflate scores for tone/enthusiasm; reward quality not effort.
- Deduct for errors even in strong responses.
- Empty response → poor scores.
- Output ONLY valid JSON, no markdown/backticks/extra text.

OUTPUT FORMAT:
{
  "llm_eval": {
    "content_relevance":{"score":,"rationale":""},
    "argument_quality":{"score":,"rationale":""},
    "structure_coherence":{"score":,"rationale":""},
    "vocabulary_range":{"score":,"rationale":""},
    "grammar_accuracy":{"score":,"rationale":""},
    "sentence_variety":{"score":,"rationale":""},
    "mechanic_accuracy":{"score":,"rationale":""},
    "task_completion":{"score":,"rationale":""}
  },
  "strengths": ["",""],
  "improvements": ["",""],
  "llm_summary": "<2-3 sentence overall assessment>"
}
''',

    "behavioral_evaluation": '''
ROLE: Expert behavioral interview assessor scoring an auto-transcribed spoken response via STAR framework.

INPUT: QUESTION, DURATION_SECONDS, TRANSCRIPT.

TASK: Score exactly 7 parameters (STAR + 2 deeper signals), each 0-10 with a ≤20-word rationale. Evaluate SUBSTANCE only — ignore pronunciation, fluency, accent, grammar, vocabulary (scored elsewhere). A vague fluent answer should score below a grammatically poor but specific/substantive one.

RUBRIC: 9-10 exceptional, specific/credible/insightful | 7-8 good, minor gaps | 5-6 adequate, lacks depth | 3-4 weak, vague/deflected | 1-2 poor | 0 no attempt/fabricated/off-topic.

PARAMETERS:
- situation_clarity: concrete context, stakes/scale, enough detail to picture it; penalize vague/fictional setups.
- personal_ownership: first-person actions ("I decided/built/escalated") vs. collective hiding ("we did"); critical signal — penalize if candidate's own role isn't isolable.
- action_depth: specific method/tool/decision logic only the doer would know vs. generic filler verbs ("managed the process").
- result_impact: clear specific outcome (quantified, observable, or tied to a named decision) vs. vague ("it worked out") or absent.
- self_reflection: demonstrates learning/growth/honest difficulty, not generic ("learned importance of teamwork"); doesn't require failure.
- question_fit: how well the story answers THIS specific question, independent of quality; if core competency (e.g. conflict) is never addressed, cap at 2-3.
- authenticity: real/lived vs. rehearsed-template; reward specific names/timeframes/messy details/admitted mistakes; penalize suspiciously perfect outcomes, generic-fits-anywhere stories, rehearsed openers, motivational-speech language.

RULES:
- Ignore grammar/fluency/accent/vocabulary entirely; never reward eloquence over substance.
- Short, specific, first-person answer beats long, fluent, vague one.
- Empty response → poor scores.
- Output ONLY valid JSON, no markdown/backticks/extra text.

OUTPUT FORMAT:
{
  "llm_eval": {
    "situation_clarity":{"score":,"rationale":""},
    "personal_ownership":{"score":,"rationale":""},
    "action_depth":{"score":,"rationale":""},
    "result_impact":{"score":,"rationale":""},
    "self_reflection":{"score":,"rationale":""},
    "question_fit":{"score":,"rationale":""},
    "authenticity":{"score":,"rationale":""}
  },
  "star_coverage": {"situation":,"task":,"action":,"result":},
  "red_flags": [""],
  "llm_summary": "<2-3 sentence overall assessment focused on substance/fit>"
}
''',

    "technical_evaluation": '''
ROLE: Strict technical interview assessor using a structured question-bank record.

INPUT: ROLE_FAMILY, TOPIC, LEVEL_BAND, QUESTION, MODEL_ANSWER, EVALUATION_RUBRIC, RED_FLAGS, CANDIDATE_ANSWER, FOLLOW_UP_QUESTION (may be empty), FOLLOW_UP_ANSWER (may be empty).

TASK: Score exactly 6 parameters, each 0-10 with a ≤20-word rationale. Calibrate strictness to LEVEL_BAND — judge against what's reasonable for that band, not MODEL_ANSWER's full sophistication.

RUBRIC: 9-10 exceptional/expert for band | 7-8 good, minor gaps | 5-6 adequate | 3-4 below average | 1-2 poor | 0 no attempt/off-topic/incorrect.

PARAMETERS:
- conceptual_correctness: facts/claims vs. MODEL_ANSWER; penalize factual errors heavily even if well-articulated.
- rubric_alignment: check each EVALUATION_RUBRIC criterion as met/partial/missed (name-dropping without explanation ≠ satisfied); score proportional to criteria genuinely met.
- depth_of_reasoning: explains WHY/HOW (cause-effect, tradeoffs, edge cases) vs. correct-but-unreasoned facts, benchmarked against MODEL_ANSWER's depth for this LEVEL_BAND.
- completeness: addresses every part/implicit sub-question/constraint of QUESTION.
- follow_up_competence: if FOLLOW_UP_QUESTION & FOLLOW_UP_ANSWER both present, judge engagement/extension of reasoning vs. restating/contradicting/ignoring. If either is empty, score 0 and note none was asked (backend renormalizes weights — do not penalize).
- red_flags_triggered: inverted — start at 10, deduct per RED_FLAGS pattern actually exhibited in CANDIDATE_ANSWER or FOLLOW_UP_ANSWER; clean answer = 10; any match should drop sharply (curated, question-specific signatures, not generic mistakes).

ADDITIONAL SIGNALS (reported, not scored):
- hallucination_flag: true only for confident, stated falsehoods (not hedged uncertainty or incompleteness).
- missed_rubric_items: EVALUATION_RUBRIC criteria entirely absent (exclude partial/weak attempts).
- red_flags_matched: RED_FLAGS entries actually exhibited (verbatim/close paraphrase); empty if none.

RULES:
- Strict and consistent; don't inflate for confident delivery.
- Compare against MODEL_ANSWER/EVALUATION_RUBRIC, not general knowledge alone.
- Weight RED_FLAGS heavily as curated signals, not style notes.
- Don't penalize missing follow-up.
- Ignore tone/pronunciation/accent/fluency/pace — technical substance only.
- Empty response → poor scores.
- Output ONLY valid JSON, no markdown/backticks/extra text.

OUTPUT FORMAT:
{
  "llm_eval": {
    "conceptual_correctness":{"score":,"rationale":""},
    "rubric_alignment":{"score":,"rationale":""},
    "depth_of_reasoning":{"score":,"rationale":""},
    "completeness":{"score":,"rationale":""},
    "follow_up_competence":{"score":,"rationale":""},
    "red_flags_triggered":{"score":,"rationale":""}
  },
  "missed_rubric_items": [],
  "red_flags_matched": [],
  "hallucination_flag": false,
  "llm_summary": "<2-3 sentence overall assessment>",
  "llm_score_raw":
}
''',
}


def speaking_user_message(topic: str, duration: str, transcript: str) -> str:
    return f"""
<TOPIC_START>
{topic}
<TOPIC_END>

<DURATION_SECONDS_START>
{duration}
<DURATION_SECONDS_END>

<TRANSCRIPT_START>
{transcript}
<TRANSCRIPT_END>
"""


def writing_user_message(topic: str, word_count: str, transcript: str) -> str:
    return f"""
<TOPIC_START>
{topic}
<TOPIC_END>

<WORD_COUNT_START>
{word_count}
<WORD_COUNT_END>

<RESPONSE_START>
{transcript}
<RESPONSE_END>
"""


def behavioral_user_message(topic: str, duration: str, transcript: str) -> str:
    return f"""
<QUESTION_START>
{topic}
<QUESTION_END>

<DURATION_SECONDS_START>
{duration}
<DURATION_SECONDS_END>

<TRANSCRIPT_START>
{transcript}
<TRANSCRIPT_END>
"""


def technical_user_message(
    role_family: str,
    topic: str,
    level_band: str,
    question: str,
    model_answer: str,
    evaluation_rubric: str,
    red_flags: str,
    candidate_answer: str,
    follow_up_question: str = "",
    follow_up_answer: str = "",
) -> str:
    return f"""
<ROLE_FAMILY_START>
{role_family}
<ROLE_FAMILY_END>

<TOPIC_START>
{topic}
<TOPIC_END>

<LEVEL_BAND_START>
{level_band}
<LEVEL_BAND_END>

<QUESTION_START>
{question}
<QUESTION_END>

<MODEL_ANSWER_START>
{model_answer}
<MODEL_ANSWER_END>

<EVALUATION_RUBRIC_START>
{evaluation_rubric}
<EVALUATION_RUBRIC_END>

<RED_FLAGS_START>
{red_flags}
<RED_FLAGS_END>

<CANDIDATE_ANSWER_START>
{candidate_answer}
<CANDIDATE_ANSWER_END>

<FOLLOW_UP_QUESTION_START>
{follow_up_question}
<FOLLOW_UP_QUESTION_END>

<FOLLOW_UP_ANSWER_START>
{follow_up_answer}
<FOLLOW_UP_ANSWER_END>
"""


# Number of distinct system prompts (= number of separate batch queues)
NUM_SYSTEM_PROMPTS: int = len(SYSTEM_PROMPTS)


# ---------------------------------------------------------------------------
# Weight matrices & scoring functions
#
# Two layers per task type:
#   1. "llm_sub_weights"  — how the LLM's individual 0-10 parameter scores
#                           combine into a single llm_score (0-100).
#   2. "overall_weights"  — how llm_score combines with the other pipeline
#                           scores (pronunciation/tone/confidence/semantic)
#                           into the final round score. Writing has no
#                           overall_weights because its overall score IS
#                           the llm_score.
#
# These weights are deliberately data, not hardcoded into logic, so they
# can be swapped per role/question-set without redeploying code.
#
# NOTE: Only the TECHNICAL_* weights below were changed in this revision.
# SPEAKING_*, WRITING_*, BEHAVIORAL_* are untouched.
# ---------------------------------------------------------------------------

# --- Speaking -----------------------------------------------------------
SPEAKING_LLM_WEIGHTS: dict[str, float] = {
    "content_relevance":   0.27,
    "idea_development":    0.23,
    "structure_coherence": 0.17,
    "vocabulary_range":    0.17,
    "grammar_accuracy":    0.11,
    "task_completion":     0.05,
}

SPEAKING_OVERALL_WEIGHTS: dict[str, float] = {
    "llm_score":           0.35,
    "pronunciation_score": 0.25,
    "confidence_score":    0.25,
    "tone_score":          0.15,
}

# --- Writing --------------------------------------------------------------
WRITING_LLM_WEIGHTS: dict[str, float] = {
    "content_relevance":   0.22,
    "argument_quality":    0.20,
    "structure_coherence": 0.15,
    "vocabulary_range":    0.15,
    "grammar_accuracy":    0.12,
    "sentence_variety":    0.08,
    "mechanic_accuracy":   0.05,
    "task_completion":     0.03,
}
# Writing has no separate acoustic pipeline — overall_writing_score == llm_score.
WRITING_OVERALL_WEIGHTS: dict[str, float] | None = None

# --- Behavioral -------------------------------------------------------------
BEHAVIORAL_LLM_WEIGHTS: dict[str, float] = {
    "personal_ownership": 0.24,
    "action_depth":       0.20,
    "question_fit":       0.18,
    "result_impact":      0.14,
    "self_reflection":    0.11,
    "situation_clarity":  0.08,
    "authenticity":       0.05,
}

BEHAVIORAL_OVERALL_WEIGHTS: dict[str, float] = {
    "llm_score":        0.55,
    "confidence_score": 0.30,
    "tone_score":       0.15,
}

# Hard point deductions (on the 0-100 llm_score scale) applied when the
# corresponding STAR element is entirely absent (star_coverage[x] == False).
BEHAVIORAL_STAR_PENALTY: dict[str, float] = {
    "situation": 2.0,
    "task":      3.0,
    "action":    6.0,
    "result":    8.0,
}

# --- Technical (REVISED) ----------------------------------------------------
# New parameter set matches the richer CSV question-bank schema:
#   - concept_coverage      -> rubric_alignment   (now driven by an explicit
#                              per-question evaluation_rubric instead of a
#                              flat expected_concepts keyword list)
#   - structure_clarity     -> dropped (rubric_alignment + level-band
#                              calibration already cover answer quality;
#                              structure is a much weaker technical signal
#                              than it is for speaking/writing)
#   - domain_relevance      -> dropped (subsumed by role_family/topic
#                              context now embedded directly in the rubric
#                              and model_answer)
#   - follow_up_competence  -> NEW (candidates now answer a follow_up
#                              question; the old prompt had no slot for it)
#   - red_flags_triggered   -> NEW (the CSV's red_flags column is now a
#                              first-class, per-question checklist rather
#                              than a generic missed_concepts afterthought)
TECHNICAL_LLM_WEIGHTS: dict[str, float] = {
    "conceptual_correctness": 0.28,
    "rubric_alignment":       0.22,
    "depth_of_reasoning":     0.18,
    "completeness":           0.12,
    "follow_up_competence":   0.10,
    "red_flags_triggered":    0.10,
}

# semantic_score weight raised to 20% per spec (was 30% previously).
# llm_score absorbs the remaining 80%.
TECHNICAL_OVERALL_WEIGHTS: dict[str, float] = {
    "llm_score":      0.80,
    "semantic_score": 0.20,
}


# Master registry — keyed by TaskType.value, matching SYSTEM_PROMPTS keys.
WEIGHT_MATRICES: dict[str, dict[str, Any]] = {
    TaskType.SPEAKING.value: {
        "llm_sub_weights": SPEAKING_LLM_WEIGHTS,
        "overall_weights": SPEAKING_OVERALL_WEIGHTS,
    },
    TaskType.WRITING.value: {
        "llm_sub_weights": WRITING_LLM_WEIGHTS,
        "overall_weights": WRITING_OVERALL_WEIGHTS,
    },
    TaskType.BEHAVIORAL.value: {
        "llm_sub_weights": BEHAVIORAL_LLM_WEIGHTS,
        "overall_weights": BEHAVIORAL_OVERALL_WEIGHTS,
        "star_penalty":    BEHAVIORAL_STAR_PENALTY,
    },
    TaskType.TECHNICAL.value: {
        "llm_sub_weights": TECHNICAL_LLM_WEIGHTS,
        "overall_weights": TECHNICAL_OVERALL_WEIGHTS,
    },
}


def compute_llm_subscore(
    llm_eval: dict[str, Any],
    weights: dict[str, float],
    skip_zero_weight_for: tuple[str, ...] = (),
) -> float:
    """
    Generic weighted average of 0-10 LLM sub-scores -> 0-100 scale.

    Missing parameters (LLM omitted a key, or returned a non-numeric score)
    are treated as a score of 0 but their weight still counts toward the
    denominator, so a missing parameter pulls the overall score down rather
    than being silently ignored.

    skip_zero_weight_for: parameter names that should be EXCLUDED from the
    weighted average entirely (both numerator and denominator) when their
    reported score is exactly 0. Used for follow_up_competence — if no
    follow-up was asked, the LLM is instructed to score it 0, and we don't
    want that to drag the technical score down for candidates who simply
    weren't asked a follow-up. Their weight is redistributed proportionally
    across the remaining parameters.
    """
    if not llm_eval:
        return 0.0

    weighted_total = 0.0
    weight_sum = 0.0

    for param, weight in weights.items():
        entry = llm_eval.get(param)
        score = entry.get("score") if isinstance(entry, dict) else None
        try:
            score = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        score = max(0.0, min(10.0, score))  # clamp to valid range

        if param in skip_zero_weight_for and score == 0.0:
            continue  # drop this parameter's weight entirely, don't count as a 0

        weighted_total += score * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    # scores are 0-10 -> scale weighted average (also 0-10) to 0-100
    return round((weighted_total / weight_sum) * 10, 2)


def calculate_speaking_llm_score(llm_eval: dict[str, Any]) -> float:
    return compute_llm_subscore(llm_eval, SPEAKING_LLM_WEIGHTS)


def calculate_writing_llm_score(llm_eval: dict[str, Any]) -> float:
    return compute_llm_subscore(llm_eval, WRITING_LLM_WEIGHTS)


def calculate_behavioral_llm_score(
    llm_eval: dict[str, Any],
    star_coverage: dict[str, bool] | None = None,
) -> float:
    """
    Behavioral llm_score = weighted sub-score average, minus hard penalties
    for any STAR element the candidate never addressed at all.
    """
    base = compute_llm_subscore(llm_eval, BEHAVIORAL_LLM_WEIGHTS)

    if star_coverage:
        missing = [k for k, covered in star_coverage.items() if not covered]
        penalty = sum(BEHAVIORAL_STAR_PENALTY.get(k, 0.0) for k in missing)
        base = max(0.0, base - penalty)

    return round(base, 2)


def calculate_technical_llm_score(
    llm_eval: dict[str, Any],
    follow_up_asked: bool = True,
) -> float:
    """
    Weighted average of the 6 technical LLM sub-scores -> 0-100 scale.

    follow_up_asked: pass False when the question had no FOLLOW_UP_QUESTION
    (or the candidate gave no FOLLOW_UP_ANSWER). In that case
    follow_up_competence's weight (10%) is excluded and redistributed
    across the other 5 parameters, rather than counting as a 0 that
    unfairly drags the score down for a follow-up that was never asked.
    """
    skip = ("follow_up_competence",) if not follow_up_asked else ()
    return compute_llm_subscore(llm_eval, TECHNICAL_LLM_WEIGHTS, skip_zero_weight_for=skip)


def calculate_llm_score(task_type: TaskType, parsed: dict[str, Any]) -> float:
    """
    Dispatcher: compute the 0-100 llm_score for a parsed LLM response,
    using the correct weight matrix and (for behavioral) STAR penalties /
    (for technical) follow-up-asked handling.
    """
    llm_eval = parsed.get("llm_eval", {}) or {}

    if task_type == TaskType.SPEAKING:
        return calculate_speaking_llm_score(llm_eval)
    if task_type == TaskType.WRITING:
        return calculate_writing_llm_score(llm_eval)
    if task_type == TaskType.BEHAVIORAL:
        return calculate_behavioral_llm_score(llm_eval, parsed.get("star_coverage"))
    if task_type == TaskType.TECHNICAL:
        follow_up_asked = bool(parsed.get("_follow_up_asked", True))
        return calculate_technical_llm_score(llm_eval, follow_up_asked=follow_up_asked)

    raise ValueError(f"Unknown task type for LLM scoring: {task_type}")


def calculate_overall_speaking_score(
    llm_score: float,
    pronunciation_score: float,
    confidence_score: float,
    tone_score: float,
) -> float:
    w = SPEAKING_OVERALL_WEIGHTS
    return round(
        llm_score * w["llm_score"]
        + pronunciation_score * w["pronunciation_score"]
        + confidence_score * w["confidence_score"]
        + tone_score * w["tone_score"],
        2,
    )


def calculate_overall_behavioral_score(
    llm_score: float,
    confidence_score: float,
    tone_score: float,
) -> float:
    w = BEHAVIORAL_OVERALL_WEIGHTS
    return round(
        llm_score * w["llm_score"]
        + confidence_score * w["confidence_score"]
        + tone_score * w["tone_score"],
        2,
    )


def calculate_overall_writing_score(llm_score: float) -> float:
    # Writing has no acoustic pipeline — the LLM score IS the final score.
    return round(llm_score, 2)


def calculate_overall_technical_score(
    llm_score: float,
    semantic_score: float,
) -> float:
    """
    Overall technical score = 80% llm_score + 20% semantic_score
    (semantic_score = embedding/similarity-based comparison of
    candidate_answer against model_answer, computed outside the LLM call).
    """
    w = TECHNICAL_OVERALL_WEIGHTS
    return round(
        llm_score * w["llm_score"]
        + semantic_score * w["semantic_score"],
        2,
    )


def calculate_overall_score(
    task_type: TaskType,
    llm_score: float,
    pronunciation_score: float | None = None,
    tone_score: float | None = None,
    confidence_score: float | None = None,
    semantic_score: float | None = None,
) -> float:
    """
    Dispatcher used by the results aggregator once all pipeline scores
    (LLM + acoustic / semantic) are available for a given answer.
    """
    if task_type == TaskType.WRITING:
        return calculate_overall_writing_score(llm_score)

    if task_type == TaskType.SPEAKING:
        if None in (pronunciation_score, tone_score, confidence_score):
            raise ValueError(
                "Speaking overall score requires pronunciation_score, "
                "tone_score, and confidence_score"
            )
        return calculate_overall_speaking_score(
            llm_score, pronunciation_score, confidence_score, tone_score
        )

    if task_type == TaskType.BEHAVIORAL:
        if None in (tone_score, confidence_score):
            raise ValueError(
                "Behavioral overall score requires tone_score and confidence_score"
            )
        return calculate_overall_behavioral_score(llm_score, confidence_score, tone_score)

    if task_type == TaskType.TECHNICAL:
        if semantic_score is None:
            raise ValueError(
                "Technical overall score requires semantic_score"
            )
        return calculate_overall_technical_score(llm_score, semantic_score)

    raise ValueError(f"Unknown task type for overall scoring: {task_type}")


# ---------------------------------------------------------------------------
# Settings (reads from environment / .env file)
# ---------------------------------------------------------------------------

class BatchingConfig(BaseSettings):
    """Batching behaviour for each system-prompt queue."""

    max_batch_size: int = Field(default=32, ge=1, le=256)
    max_wait_seconds: float = Field(default=5.0, ge=0.1, le=60.0)
    max_concurrent_batches: int = Field(default=4, ge=1)

    model_config = {"env_prefix": "BATCH_"}


class RedisConfig(BaseSettings):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: str | None = Field(default=None)
    result_ttl: int = Field(default=3600)

    model_config = {"env_prefix": "REDIS_"}

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class CeleryConfig(BaseSettings):
    worker_concurrency: int = Field(default=4, ge=1)
    max_tasks_per_child: int = Field(default=100, ge=1)
    task_soft_time_limit: int = Field(default=120)
    task_time_limit: int = Field(default=180)
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True

    model_config = {"env_prefix": "CELERY_"}


class VLLMConfig(BaseSettings):
    """vLLM server / engine configuration."""

    prod_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    prod_dtype: str = Field(default="float16")
    prod_gpu_memory_utilization: float = Field(default=0.90)
    prod_max_model_len: int = Field(default=4096)
    prod_tensor_parallel_size: int = Field(default=1)

    dev_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    dev_dtype: str = Field(default="float16")
    dev_quantization: str | None = Field(default="gptq")
    dev_gpu_memory_utilization: float = Field(default=0.60)
    dev_max_model_len: int = Field(default=2048)
    dev_tensor_parallel_size: int = Field(default=1)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=512)
    top_p: float = Field(default=1.0)

    model_config = {"env_prefix": "VLLM_"}

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class OpenAICompatConfig(BaseSettings):
    api_key: str = Field(default="sk-dummy")
    base_url: str = Field(default="https://api.openai.com/v1")
    model: str = Field(default="gpt-3.5-turbo")

    model_config = {"env_prefix": "OPENAI_"}


class AppConfig(BaseSettings):
    """Top-level application config."""

    run_mode: RunMode = Field(default=RunMode.DEV)
    backend: BackendType = Field(default=BackendType.VLLM)

    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)

    system_prompts: dict[str, str] = Field(default_factory=lambda: SYSTEM_PROMPTS)
    num_system_prompts: int = Field(default=NUM_SYSTEM_PROMPTS)

    log_level: str = Field(default="INFO")

    model_config = {"env_prefix": "APP_", "env_file": ".env", "extra": "ignore"}

    @field_validator("run_mode", mode="before")
    @classmethod
    def _coerce_run_mode(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    @property
    def is_dev(self) -> bool:
        return self.run_mode == RunMode.DEV

    @property
    def is_prod(self) -> bool:
        return self.run_mode == RunMode.PROD

    def llm_params(self) -> dict[str, Any]:
        v = self.vllm
        if self.is_prod:
            return {
                "model": v.prod_model,
                "dtype": v.prod_dtype,
                "gpu_memory_utilization": v.prod_gpu_memory_utilization,
                "max_model_len": v.prod_max_model_len,
                "tensor_parallel_size": v.prod_tensor_parallel_size,
                "quantization": None,
            }
        return {
            "model": v.dev_model,
            "dtype": v.dev_dtype,
            "gpu_memory_utilization": v.dev_gpu_memory_utilization,
            "max_model_len": v.dev_max_model_len,
            "tensor_parallel_size": v.dev_tensor_parallel_size,
            "quantization": v.dev_quantization,
        }


# Singleton – import this everywhere
settings = AppConfig()