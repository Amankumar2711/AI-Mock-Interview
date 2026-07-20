"""
PronunciationPipeline -- orchestrator.

Stages: G2P (expected phonemes)
        Wav2Vec2 (actual phonemes)
                  |
        GOPScorer (weighted Levenshtein)

NOTE: ASRStage has been REMOVED from this class entirely.
The master ASR pass is done once by SharedResources (warmup.py loads it)
and the resulting ASRResult is passed directly into evaluate() by the
caller (pipeline.py _run_pronunciation). This prevents Whisper from being
loaded twice per worker process (once in WarmResources, once here).

Usage:
    from Pronunciation.pipeline import PronunciationPipeline, get_cpu_config
    import soundfile as sf

    samples, sr = sf.read("recording.wav")
    asr_result = shared_resources.asr.process(samples, sr)   # done by caller

    pipeline = PronunciationPipeline(get_cpu_config())
    pipeline.load()

    report = pipeline.evaluate(
        audio=samples,
        sample_rate=sr,
        asr_result=asr_result,        # required — pass in from SharedResources
        reference_text="Please read the following passage.",
        session_id="sess_001",
    )
    print(report.overall_score)
    print(report.error_summary)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Pronunciation.config_pron.config import PipelineConfig, get_cpu_config
from Pronunciation.models_pron.models import ASRResult, PronunciationReport
from Pronunciation.stages.g2p_stage import G2PStage
from Pronunciation.stages.wav2vec2_stage import Wav2Vec2Stage
from Pronunciation.scorer import GOPScorer

logger = logging.getLogger(__name__)


class PronunciationPipeline:
    """
    Orchestrates G2P → Wav2Vec2 → GOPScorer.

    ASR is intentionally NOT part of this class.  The caller (SharedResources /
    warmup.py) owns the single ASRStage instance for the whole worker process
    and passes the already-computed ASRResult into evaluate().  This keeps
    Whisper weights in memory exactly once per worker.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_cpu_config()

        # ASRStage deliberately removed — owned by SharedResources.
        self.g2p = G2PStage(self.config.g2p)
        self.wav2vec2 = Wav2Vec2Stage(self.config.wav2vec2)
        self.scorer = GOPScorer(self.config.scoring)

        self._loaded = False

    def load(self) -> None:
        """Load G2P and Wav2Vec2 models.  Safe to call multiple times."""
        if self._loaded:
            return
        t0 = time.perf_counter()
        # ASR is no longer loaded here — Whisper lives in SharedResources only.
        self.g2p.load()
        self.wav2vec2.load()
        self._loaded = True
        logger.info("PronunciationPipeline loaded in %.2fs (ASR excluded — owned by SharedResources)",
                    time.perf_counter() - t0)

    def evaluate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        asr_result: ASRResult,
        reference_text: str,
        session_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> PronunciationReport:
        """
        Run pronunciation evaluation.

        Args:
            audio:          Raw audio as float32 numpy array (any sample rate).
            sample_rate:    Sample rate of `audio`.
            asr_result:     ASRResult produced by the shared ASRStage.
                            This is now a REQUIRED argument — the pipeline no
                            longer falls back to running ASR internally.
            reference_text: Expected text for reference (used as fallback label).
            session_id:     Optional session identifier for logging.
            candidate_id:   Optional candidate identifier for logging.

        Returns:
            PronunciationReport with overall_score, word_scores, phoneme_scores.
        """
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # stereo -> mono

        warnings = []

        # Guard: asr_result must be provided and have words
        if asr_result is None:
            logger.error(
                "evaluate() called without asr_result for session=%s — "
                "caller must pass the SharedResources ASRResult. Returning zero score.",
                session_id,
            )
            return PronunciationReport(
                session_id=session_id, candidate_id=candidate_id,
                reference_text=reference_text, transcribed_text="",
                language_detected="en",
                overall_score=0.0, sequence_score=0.0,
                word_scores=[], phoneme_scores=[],
                warnings=["asr_result_missing"],
            )

        if not asr_result.words:
            warnings.append("no_words_transcribed")
            report = PronunciationReport(
                session_id=session_id, candidate_id=candidate_id,
                reference_text=reference_text, transcribed_text="",
                language_detected=asr_result.language_detected,
                overall_score=0.0, sequence_score=0.0,
                word_scores=[], phoneme_scores=[],
                warnings=warnings,
            )
            report.processing_time_seconds = time.perf_counter() - t0
            return report

        # 1. G2P -- expected phonemes from ASR transcript
        expected_phonemes = self.g2p.process(asr_result)

        # 2. Wav2Vec2 -- actual phonemes spoken in the audio
        actual_phonemes = self.wav2vec2.process(audio, sample_rate)

        # 3. Weighted Levenshtein scoring (replaces MFA + GOP)
        report = self.scorer.process(
            expected_phonemes=expected_phonemes,
            actual_phonemes=actual_phonemes,
            asr_result=asr_result,
            reference_text=reference_text,
            session_id=session_id,
            candidate_id=candidate_id,
        )
        report.processing_time_seconds = time.perf_counter() - t0
        report.warnings.extend(warnings)

        # Use f-string — completely immune to % characters in values or
        # Celery's log formatter mishandling %-style args on Windows.
        _o  = report.overall_score
        _s  = report.sequence_score
        _t  = report.processing_time_seconds
        # _w  = len(report.word_scores)
        _wn = len(report.warnings)
        logger.info(
            f"overall={_o:.1f} sequence={_s:.1f} time={_t:.2f}s "
            f"warnings={_wn}"
        )
        return report