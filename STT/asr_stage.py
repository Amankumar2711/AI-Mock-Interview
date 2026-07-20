"""
ASR stage — WhisperX transcription with word-level timestamps + confidence.

Fixes baked in for Windows/CPU:
  - Pyannote VAD disabled (avoids torchcodec/FFmpeg crash)
  - audio passed as numpy array, never via file
  - no_speech_threshold lowered so short/quiet clips aren't dropped
  - short chunks padded to 1s (Whisper encoder minimum)
  - graceful fallback when word-level alignment returns nothing

Device policy:
  - WhisperX (main ASR model + align model) runs on GPU (cuda).
  - Falls back to cpu only when cuda is unavailable.

Model loading policy (production):
  1. Try loading from HF_MODELS_CACHE with pre-baked weights (offline-first).
  2. On failure, fall back to a normal network download into the same cache dir.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

import numpy as np

from Pronunciation.config_pron.config import WhisperConfig
from Pronunciation.models_pron.models import ASRResult, WordToken

logger = logging.getLogger(__name__)

_FILLERS = re.compile(r"\b(uh|um|hmm|ah|uhh|umm|er|erm|hm|mm|mhm)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Device resolution — WhisperX is a GPU model.
# ---------------------------------------------------------------------------
import torch as _torch
_WHISPERX_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
if _WHISPERX_DEVICE == "cpu":
    logger.warning("WhisperX: CUDA not available — falling back to CPU. Transcription will be slow.")

# ---------------------------------------------------------------------------
# Offline-first cache path — matches the path baked during Docker build.
# ---------------------------------------------------------------------------
_HF_MODELS_CACHE = os.environ.get("HF_MODELS_CACHE", "/app/models/hf_cache")


def clean_text(text: str) -> str:
    text = _FILLERS.sub("", text)
    text = re.sub(r"[^\w\s'-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


class ASRStage:
    """WhisperX wrapper. Call load() once, then process(audio, sr)."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self._model = None
        self._align_model = None
        self._align_metadata = None

    def load(self) -> None:
        import whisperx

        # Always target GPU for WhisperX — override whatever the config says.
        device = _WHISPERX_DEVICE
        compute_type = "float16" if device == "cuda" else self.config.compute_type

        def _try_load_model():
            """Attempt 1: use pre-baked weights from HF_MODELS_CACHE."""
            env_backup = os.environ.get("TRANSFORMERS_CACHE")
            os.environ["TRANSFORMERS_CACHE"] = _HF_MODELS_CACHE
            try:
                model = whisperx.load_model(
                    self.config.model_size,
                    device=device,
                    compute_type=compute_type,
                    language=self.config.language,
                    download_root=_HF_MODELS_CACHE,
                    asr_options={
                        "no_speech_threshold": self.config.no_speech_threshold,
                        "log_prob_threshold": self.config.log_prob_threshold,
                    },
                )
                logger.info("WhisperX: loaded from pre-baked cache at %s", _HF_MODELS_CACHE)
                return model
            finally:
                if env_backup is None:
                    os.environ.pop("TRANSFORMERS_CACHE", None)
                else:
                    os.environ["TRANSFORMERS_CACHE"] = env_backup

        def _fallback_load_model():
            """Attempt 2: normal download fallback."""
            return whisperx.load_model(
                self.config.model_size,
                device=device,
                compute_type=compute_type,
                language=self.config.language,
                download_root=_HF_MODELS_CACHE,
                asr_options={
                    "no_speech_threshold": self.config.no_speech_threshold,
                    "log_prob_threshold": self.config.log_prob_threshold,
                },
            )

        try:
            self._model = _try_load_model()
        except Exception as e:
            logger.warning(
                "WhisperX: pre-baked load failed (%s). Falling back to network download.", e
            )
            self._model = _fallback_load_model()

        # Alignment model — same offline-first strategy.
        try:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.config.language,
                device=device,
                model_dir=_HF_MODELS_CACHE,
            )
            logger.info("WhisperX align model: loaded from pre-baked cache.")
        except Exception as e:
            logger.warning(
                "WhisperX align model: pre-baked load failed (%s). Falling back to network download.", e
            )
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.config.language,
                device=device,
            )

        logger.info("WhisperX loaded: %s on %s", self.config.model_size, device)

    def process(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        import whisperx

        device = _WHISPERX_DEVICE
        audio = audio.astype(np.float32)

        # WhisperX/faster-whisper expects 16kHz
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        # Whisper encoder needs at least ~1s of audio
        min_samples = 16000
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        raw = self._model.transcribe(
            audio,
            batch_size=self.config.batch_size,
            language=self.config.language,
        )
        language = raw.get("language", "en")
        segments = raw.get("segments", [])

        if not segments:
            logger.warning("WhisperX returned no segments")
            return ASRResult(raw_text="", words=[], language_detected=language)

        try:
            aligned = whisperx.align(
                segments, self._align_model, self._align_metadata,
                audio, device, return_char_alignments=False,
            )
        except Exception as e:
            logger.warning("Alignment failed (%s) — using segment text only", e)
            aligned = {"segments": segments}

        words = self._extract_words(aligned)
        raw_text = " ".join(w.word for w in words)

        return ASRResult(raw_text=raw_text, words=words, language_detected=language)

    def _extract_words(self, aligned: dict) -> List[WordToken]:
        words: List[WordToken] = []

        for seg in aligned.get("segments", []):
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_text = seg.get("text", "").strip()
            word_entries = seg.get("words", [])

            if not word_entries and seg_text:
                # Fallback: alignment gave no word breakdown — split text
                # evenly across the segment duration.
                tokens = clean_text(seg_text).split()
                if not tokens:
                    continue
                dur = (seg_end - seg_start) / len(tokens)
                for i, t in enumerate(tokens):
                    words.append(WordToken(
                        word=t,
                        start=seg_start + i * dur,
                        end=seg_start + (i + 1) * dur,
                        confidence=0.5,  # unknown confidence
                    ))
                continue

            for w in word_entries:
                raw_word = clean_text(w.get("word", ""))
                if not raw_word:
                    continue
                words.append(WordToken(
                    word=raw_word,
                    start=float(w.get("start", seg_start)),
                    end=float(w.get("end", seg_end)),
                    confidence=float(w.get("score", 1.0)),
                ))

        return words

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        import torch
        import torchaudio.functional as F
        t = torch.from_numpy(audio).unsqueeze(0)
        t = F.resample(t, orig_freq=orig_sr, new_freq=target_sr)
        return t.squeeze(0).numpy()
