"""
warmup.py — Celery worker startup handler.

Responsibilities
----------------
1. Pre-load all heavyweight models (Whisper, ToneAnalyzer,
   PronunciationPipeline) exactly once per worker OS process so task
   bodies only do inference — no per-request disk I/O or model init.

2. Create and store a PERSISTENT asyncio event loop for the worker
   process so that task bodies can reuse it across invocations.

ROOT CAUSE FIX — models loading on every request (Windows / solo pool):
    The original code registered _do_warmup() on the `worker_ready`
    Celery signal. worker_ready does NOT fire on --pool=solo (Windows
    default). So _do_warmup() never ran, _warm_resources stayed None,
    and every task hit the fallback path that loaded all models from
    scratch — Whisper, ToneAnalyzer, Wav2Vec2, SpeechBrain — on every
    single request.

    Fix 1: Register on BOTH worker_init AND worker_ready signals.
        worker_init fires on ALL pool types including solo, prefork,
        gevent, and eventlet. worker_ready is kept as a fallback for
        pool types that fire it but not worker_init.

    Fix 2: solo pool safe run_async().
        On solo pool the worker loop IS the main thread — there is no
        separate daemon thread to submit coroutines to. run_async()
        now detects this case and uses asyncio.run() directly, which
        is safe because solo pool tasks run synchronously one at a time
        with no concurrent access to the event loop.

    Fix 3: Guard against double-warmup.
        Both signals may fire on some pool types. _do_warmup() checks
        _warm_resources is None before doing anything so the second
        signal is a no-op — models never load twice.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Startup import message routed through the logger so it appears in the worker log stream.
logger.info("[warmup] warmup.py successfully imported")

# ---------------------------------------------------------------------------
# Ensure project root + final_integration are in sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FINAL_INTEGRATION = os.path.join(_HERE, "final_integration")
for _p in [_HERE, _FINAL_INTEGRATION]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Persistent event loop — one per worker OS process
# ---------------------------------------------------------------------------
_worker_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


def _run_loop(loop: asyncio.AbstractEventLoop, ready_event: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    ready_event.set()
    loop.run_forever()


def _create_worker_loop() -> asyncio.AbstractEventLoop:
    """
    Create a new event loop in a daemon background thread and return it.
    Only used on prefork/gevent/eventlet pools where background threads work.
    On solo pool, tasks run on the main thread so we use asyncio.run() directly.
    """
    global _worker_loop, _loop_thread
    loop = asyncio.new_event_loop()
    ready_event = threading.Event()
    t = threading.Thread(
        target=_run_loop, args=(loop, ready_event), daemon=True, name="celery-worker-loop"
    )
    t.start()
    ready_event.wait()  # block until the loop is actually running
    _worker_loop = loop
    _loop_thread = t
    logger.info(
        "[warmup] Persistent event loop started in thread %s PID=%d", t.name, os.getpid()
    )
    return loop


def get_worker_loop() -> Optional[asyncio.AbstractEventLoop]:
    return _worker_loop


def run_async(coro) -> object:
    """
    Run a coroutine from a sync Celery task body.

    Prefork / gevent / eventlet pool:
        Submits the coroutine to the persistent background event loop and
        blocks the calling thread until done. Keeps httpx connection pools
        alive across tasks.

    Solo pool (Windows default):
        The persistent background loop does not exist (worker_init may or
        may not create it on solo). Falls back to asyncio.run() which is
        safe because solo pool executes only one task at a time — no
        concurrent loop access.
    """
    loop = get_worker_loop()
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    # Solo pool fallback — safe because solo is single-threaded
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# WarmResources — singleton loaded once at worker startup
# ---------------------------------------------------------------------------
class WarmResources:
    def __init__(self) -> None:
        self.asr = None
        self.tone_analyzer = None
        self.pronunciation_pipeline = None
        self.semantic_model = None
        self.pipeline_cfg = None
        self.llm_client = None
        self.loaded_at: Optional[float] = None
        self.load_duration_ms: float = 0.0
        self.is_ready: bool = False


_warm_resources: Optional[WarmResources] = None
_warmup_lock = threading.Lock()   # prevents double-warmup when both signals fire


def get_warm_resources() -> Optional[WarmResources]:
    return _warm_resources


# ---------------------------------------------------------------------------
# Core warmup logic
# ---------------------------------------------------------------------------
def _do_warmup() -> None:
    """
    Load all models once at worker startup.

    Protected by _warmup_lock so double-firing signals (worker_init +
    worker_ready both firing on some pool types) never cause double loading.
    """
    global _warm_resources

    with _warmup_lock:
        if _warm_resources is not None:
            # Already ran — second signal firing, skip entirely.
            logger.info("[warmup] Already complete, skipping duplicate signal.")
            return

        # Create the persistent event loop for prefork/gevent pools.
        # On solo pool this loop is technically unused (tasks call asyncio.run
        # directly) but creating it does no harm.
        loop = _create_worker_loop()

        warm = WarmResources()
        t0 = time.monotonic()
        logger.info("[warmup] ── Starting model pre-load PID=%d ──", os.getpid())

        try:
            from final_integration.core_integration.pipeline import _IMPORTS_OK  # noqa

            if not _IMPORTS_OK:
                logger.warning(
                    "[warmup] Sub-pipeline imports unavailable. Degraded mode (LLM only)."
                )
                _create_llm_client(warm, loop)
                warm.is_ready = True
                _warm_resources = warm
                return

            # ── PipelineConfig ────────────────────────────────────────────
            from Pronunciation.config_pron.config import PipelineConfig  # noqa
            warm.pipeline_cfg = PipelineConfig()
            logger.info("[warmup] PipelineConfig initialised.")

            # ── ASR (Whisper) ─────────────────────────────────────────────
            from STT.asr_stage import ASRStage  # noqa
            from core_main.registry import IsolatedUnsafeLoad  # noqa
            logger.info("[warmup] Loading ASR (Whisper)...")
            warm.asr = ASRStage(warm.pipeline_cfg.whisper)
            with IsolatedUnsafeLoad():
                warm.asr.load()
            logger.info("[warmup] ASR loaded ✓")

            # ── ToneAnalyzer ──────────────────────────────────────────────
            logger.info("[warmup] Loading ToneAnalyzer...")
            from tone.app.services.tone_service import ToneAnalyzer  # noqa
            warm.tone_analyzer = ToneAnalyzer()
            logger.info("[warmup] ToneAnalyzer loaded ✓")

            # ── PronunciationPipeline ─────────────────────────────────────
            logger.info("[warmup] Loading PronunciationPipeline (G2P + Wav2Vec2 only)...")
            from Pronunciation.pipeline import PronunciationPipeline  # noqa
            warm.pronunciation_pipeline = PronunciationPipeline(warm.pipeline_cfg)
            with IsolatedUnsafeLoad():
                warm.pronunciation_pipeline.load()
            logger.info("[warmup] PronunciationPipeline loaded ✓")

            # ── Semantic embedding model ─────────────────────────────────
            logger.info("[warmup] Loading semantic embedding model...")
            try:
                from technical_eval.core_tech.semantic_eval import load_local_model  # noqa
                warm.semantic_model = load_local_model()
                logger.info("[warmup] Semantic embedding model loaded ✓")
            except Exception as exc:
                logger.warning("[warmup] Semantic embedding model pre-load failed (%s)", exc)

            # ── LLMClient ─────────────────────────────────────────────────
            _create_llm_client(warm, loop)

            warm.is_ready = True

        except Exception as exc:
            import traceback
            logger.error("[warmup] Pre-load FAILED: %s", exc, exc_info=True)
            logger.debug("[warmup] Pre-load traceback: %s", traceback.format_exc())
            logger.error(
                "[warmup] Pre-load FAILED: %s — worker will load on demand.", exc,
                exc_info=True,
            )
            warm.is_ready = False

        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            warm.loaded_at = time.time()
            warm.load_duration_ms = elapsed_ms
            _warm_resources = warm
            logger.info(
                "[warmup] ── Pre-load complete in %.0f ms (ready=%s) PID=%d ──",
                elapsed_ms, warm.is_ready, os.getpid(),
            )


def _create_llm_client(warm: WarmResources, loop: asyncio.AbstractEventLoop) -> None:
    try:
        from final_integration.core_integration.llm_client import LLMClient  # noqa
        future = asyncio.run_coroutine_threadsafe(LLMClient.create(), loop)
        warm.llm_client = future.result(timeout=10)
        logger.info("[warmup] LLMClient pre-created ✓")
    except Exception as exc:
        logger.warning("[warmup] LLMClient pre-creation failed (%s) — will create per-task", exc)
        warm.llm_client = None


# ---------------------------------------------------------------------------
# Signal registration
# ---------------------------------------------------------------------------
def register_warmup_signal() -> None:
    """
    Register _do_warmup on BOTH worker_init and worker_ready signals.

    worker_init  — fires on ALL pool types (solo, prefork, gevent, eventlet)
                   immediately when the worker process starts.
    worker_ready — fires on prefork/gevent after the pool is ready.
                   Does NOT fire on solo pool — this was the original bug.

    _do_warmup() is protected by _warmup_lock so double-firing is safe.
    """
    from celery.signals import worker_init, worker_ready  # noqa

    @worker_init.connect
    def _on_worker_init(sender, **kwargs):
        # Use print() as well as logger — worker_init fires before logging
        # is fully configured so logger messages may be swallowed.
        logger.info("[warmup] worker_init signal received — starting warmup.")
        _do_warmup()

    @worker_ready.connect
    def _on_worker_ready(sender, **kwargs):
        # On prefork this fires after worker_init — _do_warmup() will detect
        # _warm_resources is already set and skip cleanly.
        logger.info("[warmup] worker_ready signal received.")
        _do_warmup()

    logger.debug("[warmup] worker_init + worker_ready signals registered.")


register_warmup_signal()