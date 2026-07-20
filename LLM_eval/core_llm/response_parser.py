"""
Parses raw LLM text output into structured evaluation dicts.
Handles common LLM formatting noise (markdown fences, leading text, etc.)
and computes llm_score deterministically from config_llm.WEIGHT_MATRICES,
instead of trusting an LLM-reported number.
"""

from __future__ import annotations

import json
import re
from typing import Any

from config_llm.config import (
    calculate_behavioral_llm_score,
    calculate_speaking_llm_score,
    calculate_writing_llm_score,
    calculate_technical_llm_score,
)
from utils_llm.logger import get_logger

logger = get_logger(__name__)

# Regex to strip ```json ... ``` or ``` ... ``` fences
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
# Regex to find the first {...} block
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _extract_json_str(text: str) -> str:
    """Best-effort extraction of a JSON object from possibly noisy text."""
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        return block_match.group(0)
    return text.strip()


def _load_json(response_text: str, request_id: str = "") -> dict[str, Any] | None:
    """Parse JSON out of a raw LLM response, returning None on failure."""
    if not response_text:
        return None
    json_str = _extract_json_str(response_text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning(
            "json_parse_error",
            request_id=request_id,
            error=str(exc),
            snippet=response_text[:200],
        )
        return None


def parse_llm_response(
    raw_text: str,
    request_id: str,
) -> tuple[dict[str, Any] | None, float | None, str | None]:
    """
    Generic fallback parser. Returns (parsed_dict, score, error_message).
    Prefer the task-specific parsers below — they compute llm_score_raw
    from the weight matrices rather than relying on a "score" key.
    """
    raw_data = _load_json(raw_text, request_id)
    if raw_data is None:
        return None, None, "JSON parse error"

    score_raw = raw_data.get("score")
    try:
        score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        score = None

    return raw_data, score, None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _try_json_loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _clean_json_text(text: str) -> str:
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single-quoted keys/strings with double quotes (best-effort)
    text = re.sub(r"'([^']*)'\s*:", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    # Remove control chars
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return text


def _to_number(val):
    try:
        f = float(val)
        if f == int(f):
            return int(f)
        return f
    except (TypeError, ValueError):
        return None


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")


def _normalize_str_list(val) -> list:
    if isinstance(val, list):
        return [str(item).strip() for item in val if item is not None]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def _extract_string_array(text: str, key: str) -> list:
    arr_match = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not arr_match:
        return []
    items_text = arr_match.group(1)
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', items_text, re.DOTALL)
    return [_unescape(i) for i in items]


def _flatten_to_llm_eval(data: dict, reserved: set[str]) -> dict:
    """
    Best-effort fallback for small/weak models that emit metrics as
    flat top-level "metric": score pairs instead of nesting them under
    "llm_eval": { "metric": {"score":.., "rationale":..} }.

    Returns a dict shaped like the expected llm_eval (each value either
    a dict with "score"/"rationale" or a bare scalar score).
    """
    flat_metrics = {
        k: v for k, v in data.items()
        if k not in reserved and isinstance(v, (int, float, str))
    }
    if not flat_metrics:
        return {}
    return {k: {"score": v, "rationale": ""} for k, v in flat_metrics.items()}


def _normalize_llm_eval(llm_eval: dict) -> dict:
    """Normalize an llm_eval dict, tolerating bare scalar metric values."""
    normalized_eval = {}
    for key, val in llm_eval.items():
        if isinstance(val, dict):
            score = _to_number(val.get("score"))
            rationale = val.get("rationale", "")
            normalized_eval[key] = {
                "score": score,
                "rationale": str(rationale).strip() if rationale is not None else ""
            }
        elif isinstance(val, (int, float, str)):
            # Bare scalar score, no rationale provided
            normalized_eval[key] = {"score": _to_number(val), "rationale": ""}
        else:
            normalized_eval[key] = {"score": None, "rationale": ""}
    return normalized_eval


def _to_bool_or_none(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "yes", "present", "1"):
            return True
        if s in ("false", "no", "absent", "0", "", "none", "n/a", "not mentioned", "missing"):
            return False
        # Any other non-empty free-text description means that
        # STAR element was actually described by the model.
        return True
    return None


# ---------------------------------------------------------------------------
# Speaking
# ---------------------------------------------------------------------------

def parse_speaking_response(response_text: str) -> dict:
    """
    Parse an LLM response containing an llm_eval JSON object and return it as a dict.
    Handles markdown code fences, extra text/preamble, trailing commas,
    single quotes, and missing/partial fields.
    """
    if not response_text or not isinstance(response_text, str):
        return _empty_speaking_result()

    text = response_text.strip()

    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Extract the outermost JSON object if there's surrounding text
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    # 3. Try strict JSON parsing first
    data = _try_json_loads(text)

    # 4. Fallback: clean common issues (trailing commas, single quotes)
    if data is None:
        cleaned = _clean_json_text(text)
        data = _try_json_loads(cleaned)

    # 5. Fallback: regex extraction if JSON parsing fully fails
    if data is None:
        data = _regex_fallback_extract_speaking(text)

    if data is None:
        data = {}

    return _normalize_speaking_result(data)


def _regex_fallback_extract_speaking(text: str) -> dict | None:
    """
    Last-resort: pull out score/rationale pairs per metric and the summary
    using regex, in case JSON is malformed beyond repair.
    """
    result = {"llm_eval": {}, "llm_summary": ""}

    # Find metric blocks: "key": { "score": N, "rationale": "..." }
    metric_pattern = re.compile(
        r'"(\w+)"\s*:\s*\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.DOTALL
    )
    for m in metric_pattern.finditer(text):
        key, score, rationale = m.groups()
        result["llm_eval"][key] = {
            "score": _to_number(score),
            "rationale": _unescape(rationale)
        }

    # Fallback within fallback: flat "metric": number pairs
    if not result["llm_eval"]:
        flat_pattern = re.compile(r'"(\w+)"\s*:\s*(\d+(?:\.\d+)?)\b')
        for m in flat_pattern.finditer(text):
            key, score = m.groups()
            if key in ("llm_score_raw",):
                continue
            result["llm_eval"][key] = {"score": _to_number(score), "rationale": ""}

    # Find summary
    summary_match = re.search(r'"llm_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if summary_match:
        result["llm_summary"] = _unescape(summary_match.group(1))

    if not result["llm_eval"] and not result["llm_summary"]:
        return None

    return result


def _empty_speaking_result() -> dict:
    return {
        "llm_eval": {},
        "llm_summary": ""
    }


def _normalize_speaking_result(data: dict) -> dict:
    """
    Ensure consistent structure: always has 'llm_eval' (dict of metric -> {score, rationale})
    and 'llm_summary' (string), even if missing in source.
    """
    if not isinstance(data, dict):
        return _empty_speaking_result()

    llm_eval = data.get("llm_eval", {})
    if not isinstance(llm_eval, dict):
        llm_eval = {}

    if not llm_eval:
        reserved = {"llm_summary", "summary", "llm_eval", "llm_score_raw"}
        # Case: metrics nested as top-level dicts with "score" key
        possible_metrics = {
            k: v for k, v in data.items()
            if k not in reserved and isinstance(v, dict) and "score" in v
        }
        if possible_metrics:
            llm_eval = possible_metrics
        else:
            # Case: flat "metric": score pairs at top level
            llm_eval = _flatten_to_llm_eval(data, reserved)

    llm_summary = data.get("llm_summary", data.get("summary", ""))
    if not isinstance(llm_summary, str):
        llm_summary = str(llm_summary) if llm_summary is not None else ""

    return {
        "llm_eval": _normalize_llm_eval(llm_eval),
        "llm_summary": llm_summary.strip()
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def parse_writing_response(response_text: str) -> dict:
    """
    Parse an extended LLM response containing llm_eval, strengths,
    improvements, llm_summary, and llm_score_raw. Returns a normalized dict.
    """
    if not response_text or not isinstance(response_text, str):
        return _empty_writing_result()

    text = response_text.strip()

    # 1. Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Extract outermost JSON object if wrapped in extra text
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    # 3. Try strict JSON
    data = _try_json_loads(text)

    # 4. Fallback: clean common issues
    if data is None:
        cleaned = _clean_json_text(text)
        data = _try_json_loads(cleaned)

    # 5. Fallback: regex extraction
    if data is None:
        data = _regex_fallback_extract_writing(text)

    if data is None:
        data = {}

    return _normalize_writing_result(data)


def _regex_fallback_extract_writing(text: str) -> dict | None:
    result = {
        "llm_eval": {},
        "strengths": [],
        "improvements": [],
        "llm_summary": "",
        "llm_score_raw": None
    }

    # Metric blocks: "key": { "score": N, "rationale": "..." }
    metric_pattern = re.compile(
        r'"(\w+)"\s*:\s*\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.DOTALL
    )
    for m in metric_pattern.finditer(text):
        key, score, rationale = m.groups()
        result["llm_eval"][key] = {
            "score": _to_number(score),
            "rationale": _unescape(rationale)
        }

    # Fallback within fallback: flat "metric": number pairs
    if not result["llm_eval"]:
        flat_pattern = re.compile(r'"(\w+)"\s*:\s*(\d+(?:\.\d+)?)\b')
        for m in flat_pattern.finditer(text):
            key, score = m.groups()
            if key in ("llm_score_raw",):
                continue
            result["llm_eval"][key] = {"score": _to_number(score), "rationale": ""}

    # Strengths array
    result["strengths"] = _extract_string_array(text, "strengths")
    # Improvements array
    result["improvements"] = _extract_string_array(text, "improvements")

    # Summary
    summary_match = re.search(r'"llm_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if summary_match:
        result["llm_summary"] = _unescape(summary_match.group(1))

    # Score raw
    score_match = re.search(r'"llm_score_raw"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if score_match:
        result["llm_score_raw"] = _to_number(score_match.group(1))

    if not any([result["llm_eval"], result["strengths"], result["improvements"],
                result["llm_summary"], result["llm_score_raw"] is not None]):
        return None

    return result


def _empty_writing_result() -> dict:
    return {
        "llm_eval": {},
        "strengths": [],
        "improvements": [],
        "llm_summary": "",
        "llm_score_raw": None
    }


def _normalize_writing_result(data: dict) -> dict:
    if not isinstance(data, dict):
        return _empty_writing_result()

    llm_eval = data.get("llm_eval", {})
    if not isinstance(llm_eval, dict):
        llm_eval = {}

    # Flat-schema fallback: small models often emit "metric": score
    # at the top level instead of nesting under "llm_eval".
    if not llm_eval:
        reserved = {
            "strengths", "improvements", "llm_summary", "summary",
            "llm_score_raw", "llm_eval"
        }
        llm_eval = _flatten_to_llm_eval(data, reserved)

    normalized_eval = _normalize_llm_eval(llm_eval)

    # strengths / improvements
    strengths = _normalize_str_list(data.get("strengths", []))
    improvements = _normalize_str_list(data.get("improvements", []))

    # llm_summary
    llm_summary = data.get("llm_summary", data.get("summary", ""))
    if not isinstance(llm_summary, str):
        llm_summary = str(llm_summary) if llm_summary is not None else ""
    llm_summary = llm_summary.strip()

    # llm_score_raw
    llm_score_raw = _to_number(data.get("llm_score_raw"))

    return {
        "llm_eval": normalized_eval,
        "strengths": strengths,
        "improvements": improvements,
        "llm_summary": llm_summary,
        "llm_score_raw": llm_score_raw
    }


# ---------------------------------------------------------------------------
# Behavioral
# ---------------------------------------------------------------------------

def parse_behavioral_eval(response_text: str) -> dict:
    """
    Parse an LLM response containing llm_eval, star_coverage, red_flags,
    llm_summary, and llm_score_raw. Returns a normalized dict.
    """
    if not response_text or not isinstance(response_text, str):
        return _empty_star_result()

    text = response_text.strip()

    # 1. Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Extract outermost JSON object if wrapped in extra text
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    # 3. Try strict JSON
    data = _try_json_loads(text)

    # 4. Fallback: clean common issues
    if data is None:
        cleaned = _clean_json_text(text)
        data = _try_json_loads(cleaned)

    # 5. Fallback: regex extraction
    if data is None:
        data = _regex_fallback_extract_star(text)

    if data is None:
        data = {}

    return _normalize_star_result(data)


def _regex_fallback_extract_star(text: str) -> dict | None:
    result = {
        "llm_eval": {},
        "star_coverage": {},
        "red_flags": [],
        "llm_summary": "",
        "llm_score_raw": None
    }

    # Metric blocks: "key": { "score": N, "rationale": "..." }
    metric_pattern = re.compile(
        r'"(\w+)"\s*:\s*\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.DOTALL
    )
    for m in metric_pattern.finditer(text):
        key, score, rationale = m.groups()
        result["llm_eval"][key] = {
            "score": _to_number(score),
            "rationale": _unescape(rationale)
        }

    # Fallback within fallback: flat "metric": number pairs
    if not result["llm_eval"]:
        flat_pattern = re.compile(r'"(\w+)"\s*:\s*(\d+(?:\.\d+)?)\b')
        for m in flat_pattern.finditer(text):
            key, score = m.groups()
            if key in ("llm_score_raw",):
                continue
            result["llm_eval"][key] = {"score": _to_number(score), "rationale": ""}

    # star_coverage block: { "situation": true, "task": "some text", ... }
    star_block_match = re.search(r'"star_coverage"\s*:\s*\{(.*?)\}', text, re.DOTALL)
    if star_block_match:
        block = star_block_match.group(1)
        for kv in re.finditer(
            r'"(\w+)"\s*:\s*(true|false|null|"(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?)',
            block, re.IGNORECASE | re.DOTALL
        ):
            key, raw_val = kv.groups()
            result["star_coverage"][key] = _parse_loose_value(raw_val)

    # red_flags array
    result["red_flags"] = _extract_string_array(text, "red_flags")

    # Summary
    summary_match = re.search(r'"llm_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if summary_match:
        result["llm_summary"] = _unescape(summary_match.group(1))

    # Score raw
    score_match = re.search(r'"llm_score_raw"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if score_match:
        result["llm_score_raw"] = _to_number(score_match.group(1))

    if not any([result["llm_eval"], result["star_coverage"], result["red_flags"],
                result["llm_summary"], result["llm_score_raw"] is not None]):
        return None

    return result


def _parse_loose_value(raw_val: str):
    raw = raw_val.strip()
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    if raw.startswith('"') and raw.endswith('"'):
        return _unescape(raw[1:-1])
    return _to_number(raw)


def _empty_star_result() -> dict:
    return {
        "llm_eval": {},
        "star_coverage": {},
        "red_flags": [],
        "llm_summary": "",
        "llm_score_raw": None
    }


def _normalize_star_result(data: dict) -> dict:
    if not isinstance(data, dict):
        return _empty_star_result()

    llm_eval = data.get("llm_eval", {})
    if not isinstance(llm_eval, dict):
        llm_eval = {}

    # Flat-schema fallback: small models often emit "metric": score
    # at the top level instead of nesting under "llm_eval".
    if not llm_eval:
        reserved = {
            "star_coverage", "red_flags", "llm_summary", "summary",
            "llm_score_raw", "llm_eval"
        }
        llm_eval = _flatten_to_llm_eval(data, reserved)

    normalized_eval = _normalize_llm_eval(llm_eval)

    # star_coverage: normalize to bool/None for known keys, keep extras
    star_coverage_raw = data.get("star_coverage", {})
    if not isinstance(star_coverage_raw, dict):
        star_coverage_raw = {}

    expected_star_keys = ["situation", "task", "action", "result"]
    normalized_star = {}
    for key in expected_star_keys:
        normalized_star[key] = _to_bool_or_none(star_coverage_raw.get(key))
    # include any extra keys present
    for key, val in star_coverage_raw.items():
        if key not in normalized_star:
            normalized_star[key] = _to_bool_or_none(val)

    # red_flags
    red_flags = _normalize_str_list(data.get("red_flags", []))

    # llm_summary
    llm_summary = data.get("llm_summary", data.get("summary", ""))
    if not isinstance(llm_summary, str):
        llm_summary = str(llm_summary) if llm_summary is not None else ""
    llm_summary = llm_summary.strip()

    # llm_score_raw
    llm_score_raw = _to_number(data.get("llm_score_raw"))

    return {
        "llm_eval": normalized_eval,
        "star_coverage": normalized_star,
        "red_flags": red_flags,
        "llm_summary": llm_summary,
        "llm_score_raw": llm_score_raw
    }


# ---------------------------------------------------------------------------
# Technical
# ---------------------------------------------------------------------------

def parse_technical_eval(response_text: str) -> dict:
    """
    Parse an LLM response for technical evaluation containing:
      - llm_eval: 6 parameters (conceptual_correctness, rubric_alignment,
                  depth_of_reasoning, completeness, follow_up_competence,
                  red_flags_triggered)
      - missed_rubric_items: list[str]
      - red_flags_matched:   list[str]
      - hallucination_flag:  bool | None
      - llm_summary:         str
      - llm_score_raw:       float | None

    Returns a normalized dict. If parsing fails at every layer, returns
    _empty_technical_result() so callers can always detect failure via
    an empty llm_eval dict rather than an exception.
    """
    if not response_text or not isinstance(response_text, str):
        return _empty_technical_result()

    text = response_text.strip()

    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Extract outermost JSON object if wrapped in extra text
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    # 3. Try strict JSON parsing
    data = _try_json_loads(text)

    # 4. Fallback: clean common issues (trailing commas, single quotes)
    if data is None:
        cleaned = _clean_json_text(text)
        data = _try_json_loads(cleaned)

    # 5. Fallback: regex extraction
    if data is None:
        data = _regex_fallback_extract_technical(text)

    if data is None:
        data = {}

    return _normalize_technical_result(data)


def _regex_fallback_extract_technical(text: str) -> dict | None:
    """
    Last-resort regex extraction for technical eval responses.
    Pulls score/rationale pairs, missed_rubric_items, red_flags_matched,
    hallucination_flag, llm_summary, and llm_score_raw.
    """
    result = {
        "llm_eval": {},
        "missed_rubric_items": [],
        "red_flags_matched": [],
        "hallucination_flag": None,
        "llm_summary": "",
        "llm_score_raw": None,
    }

    # Metric blocks: "key": { "score": N, "rationale": "..." }
    metric_pattern = re.compile(
        r'"(\w+)"\s*:\s*\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.DOTALL
    )
    for m in metric_pattern.finditer(text):
        key, score, rationale = m.groups()
        result["llm_eval"][key] = {
            "score": _to_number(score),
            "rationale": _unescape(rationale)
        }

    # Fallback within fallback: flat "metric": number pairs
    if not result["llm_eval"]:
        flat_pattern = re.compile(r'"(\w+)"\s*:\s*(\d+(?:\.\d+)?)\b')
        for m in flat_pattern.finditer(text):
            key, score = m.groups()
            if key in ("llm_score_raw",):
                continue
            result["llm_eval"][key] = {"score": _to_number(score), "rationale": ""}

    # missed_rubric_items list
    result["missed_rubric_items"] = _extract_string_array(text, "missed_rubric_items")

    # red_flags_matched list
    result["red_flags_matched"] = _extract_string_array(text, "red_flags_matched")

    # hallucination_flag: "hallucination_flag": true|false
    flag_match = re.search(
        r'"hallucination_flag"\s*:\s*(true|false)',
        text, re.IGNORECASE
    )
    if flag_match:
        result["hallucination_flag"] = flag_match.group(1).lower() == "true"

    # llm_summary
    summary_match = re.search(
        r'"llm_summary"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text, re.DOTALL
    )
    if summary_match:
        result["llm_summary"] = _unescape(summary_match.group(1))

    # llm_score_raw
    score_match = re.search(r'"llm_score_raw"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if score_match:
        result["llm_score_raw"] = _to_number(score_match.group(1))

    has_content = any([
        result["llm_eval"],
        result["missed_rubric_items"],
        result["red_flags_matched"],
        result["hallucination_flag"] is not None,
        result["llm_summary"],
        result["llm_score_raw"] is not None,
    ])
    return result if has_content else None


def _empty_technical_result() -> dict:
    return {
        "llm_eval": {},
        "missed_rubric_items": [],
        "red_flags_matched": [],
        "hallucination_flag": None,
        "llm_summary": "",
        "llm_score_raw": None,
    }


def _normalize_technical_result(data: dict) -> dict:
    """
    Ensures the result has canonical structure:
      llm_eval            — dict of metric -> {score, rationale}
      missed_rubric_items — list[str]  rubric criteria not addressed at all
      red_flags_matched   — list[str]  RED_FLAGS entries the candidate triggered
      hallucination_flag  — bool | None (None = LLM didn't report it)
      llm_summary         — str
      llm_score_raw       — float | None (LLM self-reported; override with
                            computed score from weight matrices in the pipeline)
    """
    if not isinstance(data, dict):
        return _empty_technical_result()

    # --- llm_eval ---
    llm_eval = data.get("llm_eval", {})
    if not isinstance(llm_eval, dict):
        llm_eval = {}

    if not llm_eval:
        reserved = {
            "missed_rubric_items", "red_flags_matched", "hallucination_flag",
            "llm_summary", "summary", "llm_score_raw", "llm_eval",
            # legacy key — tolerate old responses that still use missed_concepts
            "missed_concepts",
        }
        # Check top-level dicts with "score" key
        possible_metrics = {
            k: v for k, v in data.items()
            if k not in reserved and isinstance(v, dict) and "score" in v
        }
        if possible_metrics:
            llm_eval = possible_metrics
        else:
            llm_eval = _flatten_to_llm_eval(data, reserved)

    normalized_eval = _normalize_llm_eval(llm_eval)

    # --- missed_rubric_items ---
    # Also accept legacy "missed_concepts" key from older prompt versions
    missed_rubric_items = _normalize_str_list(
        data.get("missed_rubric_items") or data.get("missed_concepts", [])
    )

    # --- red_flags_matched ---
    red_flags_matched = _normalize_str_list(data.get("red_flags_matched", []))

    # --- hallucination_flag ---
    raw_flag = data.get("hallucination_flag")
    if isinstance(raw_flag, bool):
        hallucination_flag = raw_flag
    elif isinstance(raw_flag, str):
        hallucination_flag = raw_flag.strip().lower() in ("true", "yes", "1")
    elif isinstance(raw_flag, (int, float)):
        hallucination_flag = bool(raw_flag)
    else:
        hallucination_flag = None

    # --- llm_summary ---
    llm_summary = data.get("llm_summary", data.get("summary", ""))
    if not isinstance(llm_summary, str):
        llm_summary = str(llm_summary) if llm_summary is not None else ""
    llm_summary = llm_summary.strip()

    # --- llm_score_raw ---
    llm_score_raw = _to_number(data.get("llm_score_raw"))

    return {
        "llm_eval": normalized_eval,
        "missed_rubric_items": missed_rubric_items,
        "red_flags_matched": red_flags_matched,
        "hallucination_flag": hallucination_flag,
        "llm_summary": llm_summary,
        "llm_score_raw": llm_score_raw,
    }