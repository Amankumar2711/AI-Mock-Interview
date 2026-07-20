"""
Wav2Vec2 stage — extracts actual spoken phonemes from audio.

Default model `facebook/wav2vec2-lv-60-espeak-cv-ft` is a CTC model trained
on espeak IPA phoneme labels. We map its IPA output to ARPABET so it's
directly comparable with the G2P (g2p_en) expected-phoneme output.

If you have access to a model fine-tuned directly on ARPABET
(e.g. moxeeeem/wav2vec2-base-phoneme), set wav2vec2.model_name to that
and skip the IPA mapping (set use_ipa_mapping=False below).

Device policy:
  - Wav2Vec2 runs strictly on CPU.
  - VRAM is reserved for WhisperX (ASR) and the primary LLM / embedding models.

Model loading policy (production):
  1. Try loading from local model_path (dev environment pre-downloaded weights).
  2. Try loading from HF_MODELS_CACHE with local_files_only=True (pre-baked image).
  3. Fall back to a normal network download into HF_MODELS_CACHE.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# phonemizer-fork compatibility fix
# ---------------------------------------------------------------------------
# We install `phonemizer-fork` instead of `phonemizer` (the fork preserves
# the EspeakWrapper.set_data_path() API that misaki requires).
# However, HuggingFace `transformers` checks an internal flag
# (_phonemizer_available) that is set by a pip-registry lookup for the exact
# package name "phonemizer". Because the fork is registered as
# "phonemizer-fork", this flag stays False and Wav2Vec2PhonemeCTCTokenizer
# (loaded by Wav2Vec2Processor) raises a spurious ImportError at runtime.
#
# Patching the flag to True before any transformers import forces HuggingFace
# to skip the registry check. The actual `import phonemizer` inside
# transformers succeeds because phonemizer-fork installs the same top-level
# `phonemizer` Python package.
import transformers.utils.import_utils as _hf_import_utils
_hf_import_utils._phonemizer_available = True
# ---------------------------------------------------------------------------

from Pronunciation.config_pron.config import Wav2Vec2Config
from Pronunciation.models_pron.models import ActualPhonemes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache directory — read from env so it can be overridden per deployment.
# In Docker production: set HF_MODELS_CACHE to a persistent volume path.
# Default falls back to /app/models/hf_cache which is baked into the image.
# ---------------------------------------------------------------------------
_HF_MODELS_CACHE = os.environ.get("HF_MODELS_CACHE", "/app/models/hf_cache")


# Common espeak-IPA → ARPABET mapping (covers the bulk of English phonemes).
# Not exhaustive — unmapped symbols pass through uppercased so the aligner
# still treats them as distinct tokens (counted as substitutions, not crashes).
IPA_TO_ARPABET = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "g": "G",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
    "ʃ": "SH", "ʒ": "ZH", "h": "HH", "m": "M", "n": "N", "ŋ": "NG",
    "l": "L", "r": "R", "ɹ": "R", "w": "W", "j": "Y",
    "tʃ": "CH", "dʒ": "JH",
    "i": "IY", "iː": "IY", "ɪ": "IH", "e": "EH", "ɛ": "EH",
    "æ": "AE", "a": "AA", "ɑ": "AA", "ɑː": "AA", "ɒ": "AA",
    "ɔ": "AO", "ɔː": "AO", "o": "OW", "oʊ": "OW", "ʊ": "UH",
    "u": "UW", "uː": "UW", "ʌ": "AH", "ə": "AH", "ɜ": "ER", "ɜː": "ER",
    "aɪ": "AY", "aʊ": "AW", "eɪ": "EY", "ɔɪ": "OY",
}


class Wav2Vec2Stage:
    def __init__(self, config: Wav2Vec2Config, use_ipa_mapping: bool = True):
        self.config = config
        self.use_ipa_mapping = use_ipa_mapping
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        # ── Step 1: prefer the dev local path if it already exists ──────────
        if os.path.isdir(self.config.model_path):
            source = self.config.model_path
            logger.info("Wav2Vec2: loading from local path %s (CPU)", self.config.model_path)
            self._processor = Wav2Vec2Processor.from_pretrained(source)
            self._model = Wav2Vec2ForCTC.from_pretrained(source)
            self._model.eval()
            logger.info("Wav2Vec2 loaded from local path on CPU.")
            return

        # ── Step 2: try the pre-baked HF cache (offline-first) ──────────────
        source = self.config.model_name
        logger.info(
            "Wav2Vec2: attempting offline load from HF cache '%s' (CPU)", _HF_MODELS_CACHE
        )
        try:
            self._processor = Wav2Vec2Processor.from_pretrained(
                source,
                cache_dir=_HF_MODELS_CACHE,
                local_files_only=True,
            )
            self._model = Wav2Vec2ForCTC.from_pretrained(
                source,
                cache_dir=_HF_MODELS_CACHE,
                local_files_only=True,
            )
            self._model.eval()
            logger.info("Wav2Vec2 loaded from pre-baked cache on CPU.")
            return
        except Exception as e:
            logger.warning(
                "Wav2Vec2: offline load failed (%s). Falling back to network download.", e
            )

        # ── Step 3: normal network download fallback ─────────────────────────
        logger.info("Wav2Vec2: downloading from Hub '%s' into %s (CPU)", source, _HF_MODELS_CACHE)
        self._processor = Wav2Vec2Processor.from_pretrained(
            source,
            cache_dir=_HF_MODELS_CACHE,
            local_files_only=False,
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            source,
            cache_dir=_HF_MODELS_CACHE,
            local_files_only=False,
        )
        self._model.eval()
        logger.info("Wav2Vec2 loaded: %s on CPU (downloaded).", source)

    def process(self, audio: np.ndarray, sample_rate: int) -> ActualPhonemes:
        import torch

        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        # Wav2Vec2 always runs on CPU — inputs stay on CPU tensors.
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True,
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        posteriors = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        raw_tokens = self._processor.batch_decode(pred_ids)[0].split()
        phonemes = self._postprocess(raw_tokens)

        logger.debug("Actual phonemes: %s", " ".join(phonemes))
        return ActualPhonemes(phonemes=phonemes, posteriors=posteriors)

    def _postprocess(self, tokens: List[str]) -> List[str]:
        out: List[str] = []
        prev = None
        for tok in tokens:
            tok = tok.strip()
            if tok in ("", "<pad>", "<s>", "</s>", "|"):
                continue
            if tok == prev:
                continue  # collapse CTC repeats
            mapped = self._to_arpabet(tok) if self.use_ipa_mapping else tok.upper()
            out.append(mapped)
            prev = tok
        return out

    @staticmethod
    def _to_arpabet(ipa_symbol: str) -> str:
        return IPA_TO_ARPABET.get(ipa_symbol, ipa_symbol.upper())

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        import torch
        import torchaudio.functional as F
        t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        t = F.resample(t, orig_freq=orig_sr, new_freq=target_sr)
        return t.squeeze(0).numpy()