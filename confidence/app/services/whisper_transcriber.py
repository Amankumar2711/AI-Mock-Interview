import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from faster_whisper import WhisperModel
from utils.logger import get_logger

logger = get_logger()

TRANSCRIPT_DIR = "transcripts"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device policy: faster-whisper (confidence module) runs on CPU only.
# This is a lightweight auxiliary transcriber; VRAM is reserved for
# WhisperX (main ASR) and the primary LLM / embedding models.
# ---------------------------------------------------------------------------

# Offline-first cache path — matches the path baked during Docker build.
_HF_MODELS_CACHE = os.environ.get("HF_MODELS_CACHE", "/app/models/hf_cache")

# Cache model globally to avoid reloading on multiple calls
_model = None

def _load_whisper_model():
    """Load faster-whisper on CPU.

    Loading strategy (production):
      1. Try loading from HF_MODELS_CACHE using the pre-baked Systran/faster-whisper-base
         snapshot that was downloaded during the Docker image build.
      2. Fall back to a normal network download if the local snapshot is missing.
    """
    # Attempt 1: load from pre-baked snapshot directory.
    # faster-whisper accepts a directory path as the model name when the
    # snapshot already exists there.
    try:
        snapshot_path = _find_local_snapshot("Systran/faster-whisper-base", _HF_MODELS_CACHE)
        if snapshot_path:
            logger.debug("faster-whisper: loading from pre-baked path %s", snapshot_path)
            model = WhisperModel(
                snapshot_path,
                device="cpu",
                compute_type="int8",
            )
            logger.info("faster-whisper: loaded from pre-baked cache (CPU).")
            return model
    except Exception as e:
        logger.warning("faster-whisper: pre-baked load failed (%s). Falling back to download.", e)

    # Attempt 2: download from HuggingFace Hub (normal fallback).
    logger.debug("faster-whisper: downloading 'base.en' model...")
    return WhisperModel("base.en", device="cpu", compute_type="int8")


def _find_local_snapshot(repo_id: str, cache_dir: str) -> str | None:
    """Return the snapshot directory for repo_id inside an HF cache, or None."""
    # HuggingFace cache layout: <cache_dir>/models--<org>--<model>/snapshots/<hash>/
    repo_folder = "models--" + repo_id.replace("/", "--")
    snapshots_dir = os.path.join(cache_dir, repo_folder, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    hashes = [
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not hashes:
        return None
    # Use the most recent snapshot hash (sorted for determinism).
    latest = sorted(hashes)[-1]
    return os.path.join(snapshots_dir, latest)


def transcribe_audio(audio_path):
    global _model

    if _model is None:
        _model = _load_whisper_model()

    logger.debug(f"Executing faster-whisper transcription on: {audio_path}")

    segments, info = _model.transcribe(audio_path, beam_size=5)

    transcript = ""
    for segment in segments:
        transcript += segment.text + " "

    transcript = transcript.strip()
    logger.info("faster-whisper transcription successful")

    filename = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
    transcript_path = os.path.join(TRANSCRIPT_DIR, filename)

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return {
        "transcript": transcript,
        "transcript_file": transcript_path
    }
