"""
emotion_service.py — Speech Emotion Recognition via SpeechBrain IEMOCAP model.


"""

import sys
import types
import os
import numpy as np
import torch
import librosa
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tone.app.utils.logger import logger

# ── SpeechBrain version-agnostic namespace shim ───────────────────────────────
#
# SpeechBrain ≥ 1.0 moved the public API from:
#   speechbrain.pretrained.interfaces   (0.5.x)
# to:
#   speechbrain.inference.interfaces    (1.0+)
#
# Additionally, HuggingFaceWav2Vec2 was reorganised in some builds.
# We apply shims defensively so the IEMOCAP foreign_class model loads
# correctly regardless of the installed version.
#
_foreign_class = None   # resolved below

# Step 1 — Wav2Vec2 alias (needed by the IEMOCAP custom_interface.py)
try:
    import speechbrain.lobes.models.huggingface_wav2vec as _hw  # 0.5.x path
    _Wav2Vec2Cls = getattr(_hw, "HuggingFaceWav2Vec2", None)
except ImportError:
    _hw = None
    _Wav2Vec2Cls = None

if _Wav2Vec2Cls is None:
    # 1.0+ path
    try:
        from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2 as _Wav2Vec2Cls  # noqa: F401
    except ImportError:
        _Wav2Vec2Cls = None
        logger.warning("[emotion_service] Could not locate SpeechBrain Wav2Vec2 class; "
                       "SER model may fail to load.")

# Inject the alias into the legacy namespace if not already present.
if _Wav2Vec2Cls is not None and "speechbrain.lobes.models.huggingface_transformers.wav2vec2" not in sys.modules:
    try:
        _m1 = types.ModuleType("speechbrain.lobes.models.huggingface_transformers")
        _m2 = types.ModuleType("speechbrain.lobes.models.huggingface_transformers.wav2vec2")
        _m2.Wav2Vec2 = _Wav2Vec2Cls
        _m1.wav2vec2 = _m2
        sys.modules["speechbrain.lobes.models.huggingface_transformers"] = _m1
        sys.modules["speechbrain.lobes.models.huggingface_transformers.wav2vec2"] = _m2
        if "speechbrain.lobes.models" in sys.modules:
            sys.modules["speechbrain.lobes.models"].huggingface_transformers = _m1
    except Exception as _e:
        logger.warning("[emotion_service] Wav2Vec2 shim failed (non-fatal): %s", _e)

# Step 2 — Resolve foreign_class from whichever SpeechBrain interface module exists.
# Try SpeechBrain ≥ 1.0 path first, then fall back to 0.5.x path.
try:
    from speechbrain.inference.interfaces import foreign_class as _foreign_class  # SB ≥ 1.0
    logger.debug("[emotion_service] Using speechbrain.inference.interfaces (SpeechBrain ≥ 1.0)")
except ImportError:
    try:
        from speechbrain.pretrained.interfaces import foreign_class as _foreign_class  # SB 0.5.x
        logger.debug("[emotion_service] Using speechbrain.pretrained.interfaces (SpeechBrain 0.5.x)")
    except ImportError:
        _foreign_class = None
        logger.warning(
            "[emotion_service] speechbrain.inference.interfaces and "
            "speechbrain.pretrained.interfaces both unavailable. "
            "SER model will be disabled."
        )

# Step 3 — Ensure speechbrain.inference.interfaces is visible for any code that
# imports it directly (e.g. IEMOCAP custom_interface.py).
try:
    import speechbrain.inference.interfaces  # noqa: F401  — already importable in SB ≥ 1.0
except ImportError:
    # SB 0.5.x — create a shim so references to speechbrain.inference.* don't crash.
    try:
        import speechbrain.pretrained.interfaces as _spi
        _inf_mod  = types.ModuleType("speechbrain.inference")
        _inf_iface = types.ModuleType("speechbrain.inference.interfaces")
        for _attr in dir(_spi):
            setattr(_inf_iface, _attr, getattr(_spi, _attr))
        sys.modules.setdefault("speechbrain.inference", _inf_mod)
        sys.modules.setdefault("speechbrain.inference.interfaces", _inf_iface)
    except Exception as _e:
        logger.warning("[emotion_service] speechbrain.inference shim failed (non-fatal): %s", _e)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_DURATION    = 3.0    # seconds per chunk
HOP_DURATION       = 1.5    # seconds between chunk starts
CONFIDENCE_GATE    = 0.55   # minimum softmax confidence to trust a label
NEG_FLAG_THRESHOLD = 0.60   # fraction of chunks that must be negative to flag

# EmotionAnalyzer expects 16kHz input — matches the producer standardization.
_EXPECTED_SR = 16000


def _find_hf_snapshot(repo_id: str, cache_dir: str) -> str | None:
    """Return the snapshot directory for repo_id inside an HF cache, or None."""
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
    return os.path.join(snapshots_dir, sorted(hashes)[-1])


def _load_emotion_model_internal():
    if _foreign_class is None:
        raise RuntimeError(
            "SpeechBrain foreign_class is unavailable — check your SpeechBrain installation."
        )

    from huggingface_hub import snapshot_download

    # ---------------------------------------------------------------------------
    # Offline-first model resolution:
    #   1. Check the pre-baked pretrained_models/emotion_model directory
    #      (populated by scripts/docker/download_extra_models.py during build).
    #   2. Fall back to downloading from HuggingFace Hub if the local copy
    #      is missing or incomplete.
    # ---------------------------------------------------------------------------
    _HF_MODELS_CACHE = os.environ.get("HF_MODELS_CACHE", "/app/models/hf_cache")

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained_models", "emotion_model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    _hyperparams = os.path.join(MODEL_DIR, "hyperparams.yaml")
    if os.path.exists(_hyperparams):
        logger.info("[emotion_service] IEMOCAP model found at %s — skipping download.", MODEL_DIR)
    else:
        # Attempt 1: restore from HF_MODELS_CACHE snapshot baked into the image.
        _snapshot_src = _find_hf_snapshot(
            "speechbrain/emotion-recognition-wav2vec2-IEMOCAP", _HF_MODELS_CACHE
        )
        if _snapshot_src:
            logger.info(
                "[emotion_service] Restoring IEMOCAP from pre-baked cache %s → %s",
                _snapshot_src, MODEL_DIR,
            )
            try:
                import shutil as _shutil
                _shutil.copytree(_snapshot_src, MODEL_DIR, dirs_exist_ok=True)
            except Exception as _e:
                logger.warning(
                    "[emotion_service] Cache restore failed (%s). Falling back to download.", _e
                )
                _snapshot_src = None  # force download

        if not _snapshot_src:
            # Attempt 2: normal network download.
            logger.info(
                "[emotion_service] Downloading SpeechBrain IEMOCAP model to %s …", MODEL_DIR
            )
            snapshot_download(
                repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
            )

    logger.info("[emotion_service] IEMOCAP model ready at %s.", MODEL_DIR)

    import shutil
    _original_symlink = getattr(os, "symlink", None)
    if os.name == "nt":
        # Windows symlink fallback: SpeechBrain's fetch() calls os.symlink(src, dst, target_is_directory=False)
        # with target_is_directory as a positional arg. shutil.copyfile only accepts 2 positional args,
        # so we wrap it in a lambda that absorbs the extra args.
        os.symlink = lambda src, dst, *args, **kwargs: shutil.copyfile(src, dst)

    try:
        return _foreign_class(
            source=MODEL_DIR,           # use local directory — no HF network call at load time
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=MODEL_DIR,          # prevent SpeechBrain from creating a tmp save dir
            run_opts={"device": "cpu"},  # CPU-only: VRAM reserved for WhisperX + LLM/embedding
        )
    finally:
        if _original_symlink is not None:
            os.symlink = _original_symlink
        elif hasattr(os, "symlink"):
            delattr(os, "symlink")


class EmotionAnalyzer:

    def __init__(self):
        from core_main.registry import ModelRegistry
        logger.info("Initializing EmotionAnalyzer – requesting SpeechBrain SER model from registry…")
        try:
            self.classifier = ModelRegistry.get_emotion_model(_load_emotion_model_internal)
            self.model_loaded = True
            logger.info("SpeechBrain SER model loaded and pinned in RAM [OK]")
        except Exception as exc:
            logger.error(f"Failed to load SpeechBrain SER model: {exc}")
            self.classifier = None
            self.model_loaded = False

    # ── public ────────────────────────────────────────────────────────────────
    def analyze(self, audio_path: str = None, *, y: np.ndarray = None, sr: int = None) -> dict:
        """
        Analyze emotional tone.

        Args:
            audio_path: Path to audio file (used only when y/sr not provided).
            y:          Pre-loaded audio as float32 numpy array at 16kHz (preferred).
            sr:         Sample rate of y. Must be 16000 when provided.

        Returns:
            dict with dominant_emotion, negative_ratio, emotion_score, etc.
        """
        try:
            if y is not None and sr is not None:
                # Fast path — use pre-loaded array.
                # Resample to 16kHz if caller passed a different rate.
                if sr != _EXPECTED_SR:
                    y = librosa.resample(y, orig_sr=sr, target_sr=_EXPECTED_SR)
                    sr = _EXPECTED_SR
                duration = len(y) / sr
            elif audio_path is not None:
                # Fallback path — load from disk (used when called standalone).
                y, sr = librosa.load(audio_path, sr=_EXPECTED_SR)
                duration = len(y) / sr
            else:
                raise ValueError("EmotionAnalyzer.analyze() requires either (y, sr) or audio_path.")

            if not self.model_loaded or self.classifier is None:
                logger.warning("SER model not loaded – returning safe fallback.")
                return self._fallback(duration)

            chunks = self._slice_chunks(y, sr)

            batch_size = 8
            out_probs = []
            confidences = []
            text_labs = []

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch = torch.tensor(np.array(batch_chunks), dtype=torch.float32)
                out_prob, _, _, text_lab = self.classifier.classify_batch(batch)

                probs = torch.exp(out_prob.squeeze(1))
                confidence = probs.max(dim=1).values.tolist()

                out_probs.append(out_prob)
                confidences.extend(confidence)
                text_labs.extend(text_lab)

            out_prob = torch.cat(out_probs, dim=0)
            confidence = confidences
            text_lab = text_labs

            label_map = {
                "neu": "neutral",
                "hap": "positive",
                "sad": "negative",
                "ang": "negative",
            }

            chunk_results = []
            for lab, conf in zip(text_lab, confidence):
                mapped = label_map.get(lab, "neutral")
                effective = mapped if conf >= CONFIDENCE_GATE else "neutral"
                chunk_results.append({
                    "raw_label": lab,
                    "mapped": mapped,
                    "effective": effective,
                    "confidence": round(float(conf), 3),
                })
                logger.debug(f"  chunk raw={lab} conf={conf:.2f} → effective={effective}")

            effective_labels = [c["effective"] for c in chunk_results]
            counts = Counter(effective_labels)
            total = len(effective_labels)

            neg_count = counts.get("negative", 0)
            neg_ratio = neg_count / total if total > 0 else 0.0
            dominant_emotion = counts.most_common(1)[0][0] if total > 0 else "neutral"
            negative_flag = neg_ratio > NEG_FLAG_THRESHOLD

            emotion_score = max(0, int(100 - neg_ratio * 100))
            if negative_flag:
                emotion_score = min(emotion_score, 60)

            logger.info(
                f"Emotion analysis: dominant={dominant_emotion}, "
                f"neg_ratio={neg_ratio:.0%}, flag={negative_flag}, chunks={total}"
            )

            return {
                "dominant_emotion": dominant_emotion,
                "negative_ratio": round(neg_ratio, 4),
                "emotion_score": emotion_score,
                "negative_flag": negative_flag,
                "chunks_analyzed": total,
                "model_loaded": True,
                "duration": round(duration, 2),
            }

        except Exception as exc:
            logger.error(f"Emotion analysis error: {exc}")
            return self._fallback(0.0)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _slice_chunks(self, y: np.ndarray, sr: int) -> list:
        window_len = int(WINDOW_DURATION * sr)
        hop_len = int(HOP_DURATION * sr)

        chunks = [
            y[start: start + window_len]
            for start in range(0, len(y) - window_len + 1, hop_len)
        ]

        if not chunks:
            padded = np.pad(y, (0, window_len - len(y)), "constant")
            chunks = [padded]

        return chunks

    @staticmethod
    def _fallback(duration: float) -> dict:
        return {
            "dominant_emotion": "unknown",
            "negative_ratio": 0.0,
            "emotion_score": 0,
            "negative_flag": False,
            "chunks_analyzed": 0,
            "model_loaded": False,
            "duration": round(duration, 2),
        }