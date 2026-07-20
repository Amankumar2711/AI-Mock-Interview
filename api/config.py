"""
config.py — Hardware-aware configuration for the production API layer.

Executed once at import time. Detects GPU count / VRAM / CPU cores and
derives:
  - Celery worker concurrency
  - Per-worker batch size
  - Whisper model tier selection
  - LLM model name and quantization format (VRAM + env-aware)
  - All Redis / Celery / FastAPI settings from env vars with sensible defaults

LLM model selection logic
--------------------------
  APP_ENV=dev  → Ollama, model=qwen2.5:0.5b-instruct (always, regardless of VRAM)
  APP_ENV=prod → vLLM,  model=Qwen/Qwen2.5-7B-Instruct-AWQ when VRAM > 10 GB,
                        otherwise Qwen/Qwen2.5-0.5B-Instruct

Quantization (prod/vLLM only)
------------------------------
  The env var LLM_QUANTIZATION always takes precedence if set explicitly.

All values are collected into the HARDWARE dict which is exposed on
GET /health/hardware.

ADDED — Startup env var validation:
    _validate_config() runs at import time in production (APP_ENV != "dev").
    It checks that all required env vars are set to non-default values and
    that numeric settings are within sane bounds. Raises RuntimeError on
    startup rather than failing silently mid-request.
"""
from __future__ import annotations

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root + final_integration path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_INTEGRATION_DIR: str = os.path.join(PROJECT_ROOT, "final_integration")

# Attempt to explicitly load the .env file from the project root.
# This ensures variables are loaded even if the app is started via uvicorn
# without the --env-file flag (e.g. from start_api.bat).
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.isfile(_env_path):
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

for _p in [PROJECT_ROOT, FINAL_INTEGRATION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Hardware detection — GPU / VRAM
# ---------------------------------------------------------------------------
GPU_COUNT: int = 0
VRAM_GB: list[float] = []
GPU_NAMES: list[str] = []

try:
    import torch  # type: ignore

    GPU_COUNT = torch.cuda.device_count()
    for _i in range(GPU_COUNT):
        _props = torch.cuda.get_device_properties(_i)
        VRAM_GB.append(round(_props.total_memory / (1024 ** 3), 2))
        GPU_NAMES.append(_props.name)
    if GPU_COUNT:
        logger.info(
            "GPU detected: count=%d names=%s vram=%s GB",
            GPU_COUNT, GPU_NAMES, VRAM_GB,
        )
    else:
        logger.info("torch available but no CUDA GPUs — CPU-only mode")
except Exception as _gpu_err:
    logger.info("GPU detection skipped (torch unavailable): %s", _gpu_err)

CPU_CORES: int = os.cpu_count() or 4
PLATFORM: str = platform.platform()
_max_vram: float = max(VRAM_GB) if VRAM_GB else 0.0

# ---------------------------------------------------------------------------
# Derived worker / batch / model settings
# ---------------------------------------------------------------------------
#
# WORKER_COUNT auto-detection strategy
# ------------------------------------
# In production, each Celery worker loads full copies of heavy ML models
# (Whisper, Wav2Vec2, SpeechBrain emotion model) into memory. On a typical
# deployment those models consume roughly 3-4 GB per worker process.
#
# Priority order (highest to lowest):
#   1. Explicit env var  WORKER_COUNT  — always respected, set this in prod.
#   2. GPU mode          GPU_COUNT * 2 — GPU workers are VRAM-bound, not RAM-bound.
#   3. RAM-based formula floor(available_RAM_GB / 4) — safe default for CPU-only.
#      Rule of thumb: 1 worker per 4 GB of system RAM leaves headroom for the
#      OS, Redis, audio buffers, and model weights of sibling workers.
#      e.g.  8 GB instance → 2 workers
#           16 GB instance → 4 workers
#           32 GB instance → 8 workers  (hard-capped at 8)
#
# The _BOUNDS check in _validate_config() will catch values outside [1, 64]
# and raise a RuntimeError at startup rather than OOMing mid-deployment.

def _detect_available_ram_gb() -> float:
    """
    Return the total system RAM in GB using only stdlib modules.
    Falls back to 8 GB if detection fails on an unsupported platform.
    """
    try:
        # Linux / container (most common deployment target)
        with open("/proc/meminfo", "r") as _f:
            for _line in _f:
                if _line.startswith("MemTotal:"):
                    _kb = int(_line.split()[1])
                    return round(_kb / (1024 ** 2), 2)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    try:
        # Windows
        import ctypes
        class _MEMSTATUS(ctypes.Structure):  # noqa: N801
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        _ms = _MEMSTATUS()
        _ms.dwLength = ctypes.sizeof(_MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(_ms))
        return round(_ms.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        pass
    # macOS / BSD: try sysctl
    try:
        import subprocess
        _out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2)
        return round(int(_out.strip()) / (1024 ** 3), 2)
    except Exception:
        pass
    # Unknown platform — safe conservative fallback
    logger.warning("RAM detection failed on this platform; assuming 8 GB for WORKER_COUNT formula.")
    return 8.0


_TOTAL_RAM_GB: float = _detect_available_ram_gb()
logger.info("System RAM detected: %.1f GB", _TOTAL_RAM_GB)

if GPU_COUNT > 0:
    # GPU workers are VRAM-bound, not system-RAM-bound.
    # Each GPU can run one model replica; 2 workers per GPU allows overlap
    # between I/O (audio loading) and compute (inference).
    _auto_worker_count: int = min(GPU_COUNT * 2, 8)
    WORKER_COUNT: int = int(os.getenv("WORKER_COUNT", str(_auto_worker_count)))
else:
    # CPU-only mode: use RAM-based formula.
    # floor(RAM_GB / 4) — 4 GB headroom per worker for model weights + OS + buffers.
    # Minimum 1, maximum 8 to prevent runaway resource use on large instances.
    _auto_worker_count = max(1, min(int(_TOTAL_RAM_GB // 4), 8))
    WORKER_COUNT = int(os.getenv("WORKER_COUNT", str(_auto_worker_count)))

logger.info(
    "WORKER_COUNT=%d (auto=%d, ram=%.1fGB, gpu_count=%d, explicit_env=%s)",
    WORKER_COUNT, _auto_worker_count, _TOTAL_RAM_GB, GPU_COUNT,
    "yes" if os.getenv("WORKER_COUNT") else "no",
)

if _max_vram >= 16:
    _auto_batch = 8
elif _max_vram >= 8:
    _auto_batch = 4
elif _max_vram >= 4:
    _auto_batch = 2
else:
    _auto_batch = 1
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", str(_auto_batch)))

# if _max_vram >= 10:
#     _auto_whisper = "medium"
# elif _max_vram >= 6:
#     _auto_whisper = "small"
# elif _max_vram >= 2:
#     _auto_whisper = "base"
# else:
#     _auto_whisper = "tiny"
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# ---------------------------------------------------------------------------
# LLM model + quantization — VRAM-aware, env-overridable
# ---------------------------------------------------------------------------
# Dev always uses the lightweight Ollama model regardless of VRAM.
# Prod uses the 7B model only when there is enough VRAM (> 10 GB).
_APP_ENV_RAW: str = os.getenv("APP_ENV", "dev").lower()

if _APP_ENV_RAW == "prod":
    # Prod backend: vLLM
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "vllm")
    if _max_vram > 10:
        _auto_vllm_model = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    else:
        _auto_vllm_model = "Qwen/Qwen2.5-3B-Instruct"
    LLM_MODEL_VLLM:   str = os.getenv("VLLM_MODEL", _auto_vllm_model)
    LLM_MODEL_OLLAMA: str = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b-instruct")
    # Quantization: AWQ for production
    _auto_quant: str = "awq"
    LLM_QUANTIZATION: str = os.getenv("LLM_QUANTIZATION", _auto_quant)
else:
    # Dev backend: Ollama — no quantization setting needed (Ollama manages it)
    LLM_BACKEND   = os.getenv("LLM_BACKEND", "ollama")
    LLM_MODEL_OLLAMA: str = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b-instruct")
    LLM_MODEL_VLLM:   str = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
    LLM_QUANTIZATION: str = os.getenv("LLM_QUANTIZATION", "none")  # N/A for dev

# ---------------------------------------------------------------------------
# Redis / Celery
# ---------------------------------------------------------------------------
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

QUEUE_EVALUATE: str = "evaluate"
QUEUE_BATCH: str = "batch"
QUEUE_DEFAULT: str = "celery"

# ---------------------------------------------------------------------------
# Session / result TTL
# ---------------------------------------------------------------------------
SESSION_TTL: int = int(os.getenv("SESSION_TTL", "3600"))
RESULT_TTL: int = int(os.getenv("RESULT_TTL", "7200"))

# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
APP_ENV: str = os.getenv("APP_ENV", "dev").lower()
DEBUG: bool = APP_ENV == "dev"
CORS_ORIGINS: list[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

# ---------------------------------------------------------------------------
# File storage
# ---------------------------------------------------------------------------
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(PROJECT_ROOT, "uploads"))
LOG_DIR: str = os.getenv("LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── WebSocket ─────────────────────────────────────────────────────────────
WS_POLL_INTERVAL: float = float(os.getenv("WS_POLL_INTERVAL", "1.0"))

# ── Security ──────────────────────────────────────────────────────────────
# Global API Key for all non-public endpoints
API_KEY: str = os.getenv("API_KEY", "default-dev-key")

# ---------------------------------------------------------------------------
# Audio validation limits — used by evaluate.py before accepting uploads
# ---------------------------------------------------------------------------
AUDIO_MIN_DURATION_S: float = float(os.getenv("AUDIO_MIN_DURATION_S", "1.0"))   # 1 second
AUDIO_MAX_DURATION_S: float = float(os.getenv("AUDIO_MAX_DURATION_S", "600.0")) # 10 minutes
AUDIO_MAX_SIZE_MB: float = float(os.getenv("AUDIO_MAX_SIZE_MB", "50.0"))        # 50 MB
AUDIO_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    os.getenv("AUDIO_ALLOWED_EXTENSIONS", ".wav,.mp3,.m4a,.flac,.ogg,.webm").split(",")
)

# ---------------------------------------------------------------------------
# HARDWARE dict — returned by GET /health/hardware
# ---------------------------------------------------------------------------
HARDWARE: dict = {
    "platform":        PLATFORM,
    "cpu_cores":       CPU_CORES,
    "gpu_count":       GPU_COUNT,
    "gpu_names":       GPU_NAMES,
    "vram_gb":         VRAM_GB,
    "worker_count":    WORKER_COUNT,
    "batch_size":      BATCH_SIZE,
    "whisper_model":   WHISPER_MODEL,
    "app_env":         APP_ENV,
    "redis_url":       REDIS_URL,
    # LLM backend / model info
    "llm_backend":     LLM_BACKEND,
    "llm_model_ollama": LLM_MODEL_OLLAMA,
    "llm_model_vllm":  LLM_MODEL_VLLM,
    "llm_quantization": LLM_QUANTIZATION,
}

# ---------------------------------------------------------------------------
# Startup env var validation
# ---------------------------------------------------------------------------
_PROD_REQUIRED: list[tuple[str, str, str]] = [
    # (env_var_name, current_value, bad_default)
    # Raises RuntimeError in prod if value is still the dev default.
    ("REDIS_URL",             REDIS_URL,          "redis://localhost:6379/0"),
    ("CELERY_BROKER_URL",     CELERY_BROKER_URL,  "redis://localhost:6379/0"),
    ("CELERY_RESULT_BACKEND", CELERY_RESULT_BACKEND, "redis://localhost:6379/0"),
]

_BOUNDS: list[tuple[str, float, float, float]] = [
    # (name, value, min, max)
    ("SESSION_TTL",            SESSION_TTL,            60,    86400 * 7),
    ("RESULT_TTL",             RESULT_TTL,             60,    86400 * 7),
    ("WORKER_COUNT",           WORKER_COUNT,            1,    64),
    ("BATCH_SIZE",             BATCH_SIZE,              1,    32),
    ("WS_POLL_INTERVAL",       WS_POLL_INTERVAL,       0.1,   60.0),
    ("AUDIO_MIN_DURATION_S",   AUDIO_MIN_DURATION_S,   0.1,   60.0),
    ("AUDIO_MAX_DURATION_S",   AUDIO_MAX_DURATION_S,   10.0,  3600.0),
    ("AUDIO_MAX_SIZE_MB",      AUDIO_MAX_SIZE_MB,      1.0,   500.0),
]


def _validate_config() -> None:
    """
    Validate critical env vars at startup.

    In production (APP_ENV != "dev"), raises RuntimeError immediately if:
      - Required env vars still hold localhost dev defaults
      - Numeric settings are outside sane operational bounds

    In dev mode, logs warnings instead of crashing so local dev still works
    without a full .env file.
    """
    errors: list[str] = []
    warnings: list[str] = []

    is_prod = APP_ENV not in ("dev", "test", "local")

    # Check required vars are not localhost defaults in production
    for name, value, bad_default in _PROD_REQUIRED:
        if is_prod and value == bad_default:
            errors.append(
                f"  {name} is still set to the dev default '{bad_default}'. "
                f"Set it explicitly in .env or the container environment."
            )
        elif value == bad_default:
            warnings.append(f"  {name} is using dev default '{bad_default}'")

    # Check numeric bounds
    for name, value, lo, hi in _BOUNDS:
        if not (lo <= value <= hi):
            msg = (
                f"  {name}={value} is outside the allowed range [{lo}, {hi}]. "
                f"Check your .env file."
            )
            if is_prod:
                errors.append(msg)
            else:
                warnings.append(msg)

    # CORS wildcard in production is a security risk — warn always
    if is_prod and "*" in CORS_ORIGINS:
        errors.append(
            "  CORS_ORIGINS='*' is not safe for production. "
            "Set it to your actual frontend origin(s), e.g. 'https://app.example.com'."
        )

    for w in warnings:
        logger.warning("[config] %s", w)

    if errors:
        msg = (
            f"[config] Startup validation failed — {len(errors)} error(s):\n"
            + "\n".join(errors)
            + "\n\nFix these in your .env file before starting in production."
        )
        raise RuntimeError(msg)

    if not errors and not warnings:
        logger.info("[config] Startup validation passed (APP_ENV=%s)", APP_ENV)


# Run validation at import time — crashes fast in prod, warns in dev
_validate_config()