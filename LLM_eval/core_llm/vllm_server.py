"""
vLLM server launcher.

Usage:
  python -m workers.vllm_server

Reads all configuration from config/settings.py and starts a vLLM
OpenAI-compatible server process.  In dev mode a small quantised model
is used so the server fits on a laptop GPU or CPU.

The script is intentionally thin — it just constructs the right CLI flags
and execs `vllm serve`, which is the recommended way to run vLLM in production.
"""

from __future__ import annotations

import subprocess
import sys

from config_llm.settings import RunMode, settings
from utils_llm.logger import get_logger

logger = get_logger(__name__)


def build_vllm_args() -> list[str]:
    kw = settings.vllm_kwargs

    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",            kw["model"],
        "--host",             settings.VLLM_HOST,
        "--port",             str(settings.VLLM_PORT),
        "--max-model-len",    str(kw["max_model_len"]),
        "--dtype",            kw["dtype"],
        "--tensor-parallel-size", str(kw["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(kw["gpu_memory_utilization"]),
        "--served-model-name", kw["model"],
    ]

    if kw.get("quantization"):
        args += ["--quantization", kw["quantization"]]

    if settings.RUN_MODE == RunMode.PROD and kw.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")

    if settings.RUN_MODE == RunMode.DEV and kw.get("enforce_eager"):
        args.append("--enforce-eager")

    if kw.get("trust_remote_code"):
        args.append("--trust-remote-code")

    return args


def main() -> None:
    if settings.USE_CLOUD_LLM:
        logger.info("cloud_mode_enabled_skipping_vllm")
        return

    args = build_vllm_args()
    logger.info(
        "starting_vllm_server",
        mode=settings.RUN_MODE.value,
        model=settings.model_name,
        host=settings.VLLM_HOST,
        port=settings.VLLM_PORT,
    )
    logger.info("vllm_command", cmd=" ".join(args))

    try:
        subprocess.run(args, check=True)
    except KeyboardInterrupt:
        logger.info("vllm_server_stopped")
    except subprocess.CalledProcessError as exc:
        logger.error("vllm_server_error", returncode=exc.returncode)
        sys.exit(1)


if __name__ == "__main__":
    main()