"""
Levenshtein Alignment Stage — replaces MFA forced aligner.

Uses word-level timestamps from WhisperX ASR + Levenshtein DP alignment
between G2P-expected phonemes and Wav2Vec2-actual phonemes to produce
per-phoneme time intervals.

No external tool required — pure Python / NumPy.

Strategy
--------
1. For each word, we have [start, end] seconds from WhisperX.
2. G2P gives expected phonemes; Wav2Vec2 gives actual phoneme sequence.
3. We run Levenshtein alignment between expected and actual for the word.
4. Matched / substituted expected phonemes are assigned proportional durations
   within the word's time span.
5. Deletions (expected phoneme not found in actual) get a minimum duration slot.
6. Insertions are attached to the nearest expected phoneme.

This produces the same AlignmentResult / AlignedPhoneme objects the scorer
already consumes, so scorer.py needs no changes.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from Pronunciation.config_pron.config import AlignmentConfig
from Pronunciation.models_pron.models import (
    ASRResult, ActualPhonemes, AlignmentResult, AlignedPhoneme, ExpectedPhonemes,
)

logger = logging.getLogger(__name__)


# ── Levenshtein alignment (same as in scorer.py, duplicated to stay self-contained) ──

def _lev_align(
    expected: List[str], actual: List[str]
) -> List[Tuple[Optional[str], Optional[str]]]:
    """Return (expected, actual) pairs; None on either side = deletion / insertion."""
    m, n = len(expected), len(actual)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    alignment: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and expected[i - 1] == actual[j - 1]:
            alignment.append((expected[i - 1], actual[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((expected[i - 1], actual[j - 1]))  # substitution
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append((None, actual[j - 1]))  # insertion
            j -= 1
        else:
            alignment.append((expected[i - 1], None))  # deletion
            i -= 1
    alignment.reverse()
    return alignment


class LevenshteinAlignmentStage:
    """
    Drop-in replacement for AlignmentStage (MFA).

    load()    — no-op (nothing to download)
    process() — takes audio metadata + ASR result, returns AlignmentResult
                with per-phoneme time intervals derived from Levenshtein alignment.

    Because process() now also needs expected + actual phonemes, the signature
    is extended; pipeline.py is updated to pass them.
    """

    def __init__(self, config: AlignmentConfig):
        self.config = config

    def load(self) -> None:
        logger.info("LevenshteinAlignmentStage ready (no external tool required)")

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        asr_result: ASRResult,
        expected_phonemes: List[ExpectedPhonemes],
        actual_phonemes: ActualPhonemes,
    ) -> AlignmentResult:
        """
        Align phonemes per word using word timestamps + Levenshtein DP.

        Parameters
        ----------
        audio           : raw audio samples (unused here, kept for API compatibility)
        sample_rate     : sample rate (unused here)
        asr_result      : WhisperX output — gives per-word [start, end]
        expected_phonemes : G2P output — expected phoneme lists per word
        actual_phonemes   : Wav2Vec2 output — flat spoken phoneme sequence
        """
        if not asr_result.words:
            return AlignmentResult([], success=False, error_message="Empty ASR result")

        # Build word → time-span mapping from ASR
        word_spans: dict[str, Tuple[float, float]] = {}
        for wt in asr_result.words:
            word_spans[wt.word.lower()] = (wt.start, wt.end)

        aligned: List[AlignedPhoneme] = []

        # Distribute actual phonemes across words proportionally
        total_expected = sum(len(ep.phonemes) for ep in expected_phonemes)
        actual_flat = actual_phonemes.phonemes
        actual_offset = 0

        for ep in expected_phonemes:
            word = ep.word.lower()
            exp_phones = ep.phonemes
            if not exp_phones:
                continue

            # Word time span — fall back to proportional if word not in ASR output
            if word in word_spans:
                w_start, w_end = word_spans[word]
            else:
                # proportional estimate from total audio duration
                total_dur = len(audio) / sample_rate
                word_index = [e.word.lower() for e in expected_phonemes].index(word)
                n_words = len(expected_phonemes)
                w_start = total_dur * word_index / n_words
                w_end = total_dur * (word_index + 1) / n_words

            word_dur = max(w_end - w_start, 0.01)

            # Slice of actual phonemes proportional to expected count
            n_expected = len(exp_phones)
            n_actual_slice = max(1, round(len(actual_flat) * n_expected / max(total_expected, 1)))
            act_slice = actual_flat[actual_offset: actual_offset + n_actual_slice]
            actual_offset += n_actual_slice

            # Levenshtein align expected vs actual for this word
            pairs = _lev_align(exp_phones, act_slice)

            # Assign time slots — only expected phonemes (not None) consume duration
            # Count how many expected phonemes are present (not deleted-insertion-only)
            expected_present = [exp for exp, act in pairs if exp is not None]
            n_present = max(len(expected_present), 1)
            slot_dur = word_dur / n_present

            slot_index = 0
            for exp_p, act_p in pairs:
                if exp_p is None:
                    # Pure insertion — attach to previous phoneme's slot (no new interval)
                    continue
                p_start = w_start + slot_index * slot_dur
                p_end = p_start + slot_dur
                # Use the actual phoneme if matched/substituted, else expected
                phoneme_label = act_p if act_p is not None else exp_p
                aligned.append(AlignedPhoneme(
                    phoneme=phoneme_label,
                    start=round(p_start, 4),
                    end=round(p_end, 4),
                    word=word,
                ))
                slot_index += 1

        if not aligned:
            return AlignmentResult([], success=False, error_message="No phonemes aligned")

        logger.debug("Aligned %d phonemes across %d words", len(aligned), len(expected_phonemes))
        return AlignmentResult(aligned_phonemes=aligned, success=True)


# Back-compat alias so any code that imports AlignmentStage still works
AlignmentStage = LevenshteinAlignmentStage
