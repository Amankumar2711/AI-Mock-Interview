"""
config.py — central configuration, including the dev/prod backend switch.
Backend selection:
    APP_ENV=dev   -> Ollama   (OpenAI-compatible local endpoint)
    APP_ENV=prod  -> vLLM     (OpenAI-compatible self-hosted endpoint)
Both backends speak the same /v1/chat/completions wire format, so the
rest of the pipeline (LLMClient) never branches on backend type — it
just reads `settings.base_url` / `settings.model_name`.

Model selection is driven by api.config (hardware-aware):
  dev   -> Ollama, qwen2.5:0.5b-instruct
  prod  -> vLLM,  Qwen/Qwen2.5-7B-Instruct-AWQ (VRAM>10 GB) or 0.5B fallback
           Quantization: AWQ

System prompts, user-message builders, weight matrices, and all LLM
scoring functions are now imported directly from LLM_eval/config_llm/config.py,
which is the single source of truth for those concerns.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
# ---------------------------------------------------------------------------
# Ensure LLM_eval sub-packages are importable.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LLM_EVAL_DIR = os.path.join(_ROOT, "LLM_eval")
for _p in (_LLM_EVAL_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ---------------------------------------------------------------------------
# Pipeline-level enums
# ---------------------------------------------------------------------------
class BackendType(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"  # optional cloud fallback for testing
class TaskType(str, Enum):
    SPEAKING = "speaking_evaluation"
    WRITING = "writing_evaluation"
    BEHAVIORAL = "behavioral_evaluation"
    TECHNICAL = "technical_evaluation"
# Which sub-pipelines run for each task type, per the spec:
#   speaking    -> confidence + tone + pronunciation + LLM eval
#   writing     -> LLM eval only
#   behavioral  -> tone + confidence + LLM eval
#   technical   -> LLM eval + semantic eval (no acoustic pipeline)
TASK_PIPELINES: dict[TaskType, set[str]] = {
    TaskType.SPEAKING: {"confidence", "tone", "pronunciation", "llm"},
    TaskType.WRITING: {"llm"},
    TaskType.BEHAVIORAL: {"tone", "confidence", "llm"},
    TaskType.TECHNICAL: {"llm", "semantic"},
}
# ---------------------------------------------------------------------------
# Backend connection settings
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pull hardware-selected model names from api.config when available.
# api.config is the single source of truth for VRAM-aware defaults;
# we only fall back to bare env vars if api.config cannot be imported
# (e.g. when running final_integration standalone without the full API stack).
# ---------------------------------------------------------------------------
try:
    from api.config import (  # noqa: PLC0415
        LLM_MODEL_OLLAMA as _HW_OLLAMA_MODEL,
        LLM_MODEL_VLLM   as _HW_VLLM_MODEL,
        LLM_QUANTIZATION as _HW_QUANTIZATION,
    )
except ImportError:
    # Standalone mode — fall back to env vars only.
    _HW_OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",    "qwen2.5:0.5b-instruct")
    _HW_VLLM_MODEL    = os.getenv("VLLM_MODEL",      "Qwen/Qwen2.5-7B-Instruct-AWQ")
    _HW_QUANTIZATION  = os.getenv("LLM_QUANTIZATION", "awq")

@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434/v1"
    # Model resolved from hardware detection; env var still takes final precedence.
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", _HW_OLLAMA_MODEL))
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
@dataclass(frozen=True)
class VLLMSettings:
    # Build base_url from VLLM_HOST + VLLM_PORT env vars (set to 'vllm:8001' by
    # docker-compose), so the LLMClient targets the container name on the internal
    # Docker network rather than localhost (which is the container itself).
    base_url: str = os.getenv(
        "VLLM_BASE_URL",
        f"http://{os.getenv('VLLM_HOST', 'localhost')}:{os.getenv('VLLM_PORT', '8001')}/v1",
    )
    # Model resolved from hardware detection; env var still takes final precedence.
    model: str = field(default_factory=lambda: os.getenv("VLLM_MODEL", _HW_VLLM_MODEL))
    temperature: float = float(os.getenv("VLLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("VLLM_MAX_TOKENS", "2048"))
    top_p: float = float(os.getenv("VLLM_TOP_P", "1.0"))
    # Quantization: awq; set by api.config.
    # This field is informational — it documents what vLLM was launched with.
    # Pass --quantization $LLM_QUANTIZATION to the vLLM server at startup.
    quantization: str = field(default_factory=lambda: os.getenv("LLM_QUANTIZATION", _HW_QUANTIZATION))
@dataclass(frozen=True)
class OpenAICompatSettings:
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
@dataclass(frozen=True)
class WhisperSettings:
    model_size: str = os.getenv("WHISPER_MODEL", "base")
    device: str = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
@dataclass
class Settings:
    """
    Resolved at import time from APP_ENV. dev -> ollama, prod -> vllm.
    Override explicitly with LLM_BACKEND=ollama|vllm|openai.
    """
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "dev").lower())
    backend: BackendType = field(init=False)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    vllm: VLLMSettings = field(default_factory=VLLMSettings)
    openai_compat: OpenAICompatSettings = field(default_factory=OpenAICompatSettings)
    whisper: WhisperSettings = field(default_factory=WhisperSettings)
    # producer/consumer tuning
    queue_max_size: int = int(os.getenv("QUEUE_MAX_SIZE", "100"))
    num_consumers: int = int(os.getenv("NUM_CONSUMERS", "4"))
    llm_concurrency: int = int(os.getenv("LLM_CONCURRENCY", "8"))
    def __post_init__(self):
        override = os.getenv("LLM_BACKEND")
        if override:
            self.backend = BackendType(override.lower())
        elif self.app_env == "prod":
            self.backend = BackendType.VLLM
        else:
            self.backend = BackendType.OLLAMA
    @property
    def active(self):
        """The settings object for whichever backend is active."""
        return {
            BackendType.OLLAMA: self.ollama,
            BackendType.VLLM: self.vllm,
            BackendType.OPENAI: self.openai_compat,
        }[self.backend]
    def base_url(self) -> str:
        return self.active.base_url
    def llm_params(self) -> dict:
        a = self.active
        return {
            "model": a.model,
            "temperature": a.temperature,
            "max_tokens": a.max_tokens,
            "top_p": a.top_p,
        }
settings = Settings()
# ---------------------------------------------------------------------------
# System prompts and user-message builders — imported from LLM_eval.
#
# LLM_eval/config_llm/config.py is the single source of truth for:
#   • SYSTEM_PROMPTS  (full production prompts with rubrics + schemas)
#   • speaking/writing/behavioral_user_message() builders
#   • WEIGHT_MATRICES and all calculate_*_score() functions
# ---------------------------------------------------------------------------
from LLM_eval.config_llm.config import (  # noqa: E402  (after sys.path setup above)
    SYSTEM_PROMPTS,
    speaking_user_message,
    writing_user_message,
    behavioral_user_message,
    technical_user_message,
    calculate_llm_score,
    calculate_overall_score,
    calculate_speaking_llm_score,
    calculate_writing_llm_score,
    calculate_behavioral_llm_score,
    calculate_technical_llm_score,
    calculate_overall_speaking_score,
    calculate_overall_behavioral_score,
    calculate_overall_writing_score,
    calculate_overall_technical_score,
    WEIGHT_MATRICES,
    SPEAKING_LLM_WEIGHTS,
    WRITING_LLM_WEIGHTS,
    BEHAVIORAL_LLM_WEIGHTS,
    TECHNICAL_LLM_WEIGHTS,
    SPEAKING_OVERALL_WEIGHTS,
    BEHAVIORAL_OVERALL_WEIGHTS,
    BEHAVIORAL_STAR_PENALTY,
    TECHNICAL_OVERALL_WEIGHTS,
)
__all__ = [
    # pipeline-level
    "BackendType",
    "TaskType",
    "TASK_PIPELINES",
    "Settings",
    "settings",
    # prompts & builders (from LLM_eval)
    "SYSTEM_PROMPTS",
    "speaking_user_message",
    "writing_user_message",
    "behavioral_user_message",
    "technical_user_message",
    # scoring functions (from LLM_eval)
    "calculate_llm_score",
    "calculate_overall_score",
    "calculate_speaking_llm_score",
    "calculate_writing_llm_score",
    "calculate_behavioral_llm_score",
    "calculate_technical_llm_score",
    "calculate_overall_speaking_score",
    "calculate_overall_behavioral_score",
    "calculate_overall_writing_score",
    "calculate_overall_technical_score",
    # weight data (from LLM_eval)
    "WEIGHT_MATRICES",
    "SPEAKING_LLM_WEIGHTS",
    "WRITING_LLM_WEIGHTS",
    "BEHAVIORAL_LLM_WEIGHTS",
    "TECHNICAL_LLM_WEIGHTS",
    "SPEAKING_OVERALL_WEIGHTS",
    "BEHAVIORAL_OVERALL_WEIGHTS",
    "BEHAVIORAL_STAR_PENALTY",
    "TECHNICAL_OVERALL_WEIGHTS",
]