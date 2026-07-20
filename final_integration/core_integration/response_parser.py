"""
response_parser.py — thin re-export shim.
All parsing logic lives in LLM_eval/core_llm/response_parser.py.
This file simply re-exports the three task-specific parsers so that
llm_client.py (and any other code inside final_integration) can import
them from their local package path without knowing about LLM_eval's
directory layout.
The LLM_eval parsers are strictly superior to the old hand-rolled
versions that used to live here:
  • Full regex fallback chains for malformed/partial JSON
  • Proper llm_eval dict normalisation (nested dict OR bare scalar)
  • star_coverage bool normalisation for behavioral responses
  • strengths / improvements / red_flags list handling
Scoring (llm_score_raw / overall_score) is now computed by the caller
(pipeline.py) using calculate_llm_score() / calculate_overall_score()
from config_llm.config — not inside the parser itself.
"""
from __future__ import annotations
import os
import sys
# Ensure LLM_eval is on sys.path so its sub-packages resolve correctly.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LLM_EVAL_DIR = os.path.join(_ROOT, "LLM_eval")
for _p in (_LLM_EVAL_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from core_llm.response_parser import (  # noqa: E402
    parse_behavioral_eval,
    parse_speaking_response,
    parse_writing_response,
    parse_technical_eval,
)
__all__ = [
    "parse_speaking_response",
    "parse_writing_response",
    "parse_behavioral_eval",
    "parse_technical_eval",
]