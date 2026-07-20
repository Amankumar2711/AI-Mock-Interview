"""
Scorer — weighted Levenshtein alignment (MFA replacement).

WHY THE OLD APPROACH FAILED
----------------------------
The previous scorer aligned each word's expected phonemes against a
*proportional slice* of the actual phoneme array (e.g. word 3 of 10 gets
actual_flat[20%:30%]). This assumes word boundaries divide the actual
phoneme stream evenly -- they don't. A well-pronounced word could get sliced
into a chunk containing the tail of the previous word + head of the next,
producing spurious substitutions regardless of how well it was actually
spoken. Plain (unweighted) Levenshtein also treats "T -> D" (one feature
off) the same as "T -> AH" (total articulation failure), which doesn't
match how humans judge pronunciation.

NEW APPROACH
------------
1. Run ONE weighted-cost Levenshtein alignment over the FULL expected vs
   FULL actual phoneme sequences. Costs come from `phoneme_costs.py`:
     - vowel<->vowel (close pair, e.g. IH/IY): cheap   -> tolerates accents
     - consonant deletion (P, K, S, ...):      expensive
     - consonant<->vowel substitution:         maximum cost
   This single global alignment is far more stable than per-word slicing
   because the DP naturally finds the best correspondence across the whole
   utterance -- it doesn't need word-boundary information to do so.

2. Map the aligned pairs back onto words using the EXPECTED phoneme
   sequence's word boundaries (known exactly from G2P -- we know which
   expected phoneme belongs to which word). Each (expected, actual) pair
   is attributed to the word that owns the expected phoneme. Pure
   insertions (no expected phoneme) are attributed to the nearest
   preceding word.

3. Score:
     phoneme_score = 100 * (1 - cost / max_cost)     clipped to [0, 100]
     word_score    = mean(phoneme_score over that word's pairs)
     overall_score = 100 * (1 - total_cost / total_max_possible_cost)
                   = a normalised whole-utterance score, robust to length
                     mismatches between expected/actual sequences.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from Pronunciation.config_pron.config import ScoringConfig
from Pronunciation.phoneme_costs import (
    PhonemeCostWeights, substitution_cost, deletion_cost, insertion_cost,
)
from Pronunciation.models_pron.models import (
    ExpectedPhonemes, ActualPhonemes, ASRResult,
    PhonemeScore, PhonemeErrorType, WordScore, PronunciationReport,
)

logger = logging.getLogger(__name__)


# -- Weighted DP alignment ---------------------------------------------------

def weighted_align(
    expected: List[str],
    actual: List[str],
    weights: PhonemeCostWeights,
) -> Tuple[List[Tuple[Optional[str], Optional[str], float]], float]:
    """
    Weighted edit-distance alignment.

    Returns:
        alignment: list of (expected_phoneme, actual_phoneme, cost) triples.
                   None on either side = deletion (expected) / insertion (actual).
        total_cost: sum of all edit costs along the optimal path.
    """
    m, n = len(expected), len(actual)
    dp = np.zeros((m + 1, n + 1), dtype=np.float64)

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + deletion_cost(expected[i - 1], weights)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + insertion_cost(actual[j - 1], weights)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + substitution_cost(expected[i - 1], actual[j - 1], weights)
            dele = dp[i - 1][j] + deletion_cost(expected[i - 1], weights)
            ins = dp[i][j - 1] + insertion_cost(actual[j - 1], weights)
            dp[i][j] = min(sub, dele, ins)

    # Backtrace
    alignment: List[Tuple[Optional[str], Optional[str], float]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = substitution_cost(expected[i - 1], actual[j - 1], weights)
            if abs(dp[i][j] - (dp[i - 1][j - 1] + sub_cost)) < 1e-9:
                alignment.append((expected[i - 1], actual[j - 1], sub_cost))
                i -= 1; j -= 1
                continue
        if i > 0:
            del_cost = deletion_cost(expected[i - 1], weights)
            if abs(dp[i][j] - (dp[i - 1][j] + del_cost)) < 1e-9:
                alignment.append((expected[i - 1], None, del_cost))
                i -= 1
                continue
        if j > 0:
            ins_cost = insertion_cost(actual[j - 1], weights)
            alignment.append((None, actual[j - 1], ins_cost))
            j -= 1
            continue
        break  # safety

    alignment.reverse()
    return alignment, float(dp[m][n])


# -- Scorer -------------------------------------------------------------------

class GOPScorer:
    """
    Despite the name (kept for API compatibility), this no longer computes
    classic GOP. It performs one global weighted-Levenshtein alignment and
    derives both per-word and overall scores from the aligned edit costs.
    """

    def __init__(self, config: ScoringConfig, cost_weights: Optional[PhonemeCostWeights] = None):
        self.config = config
        self.weights = cost_weights or PhonemeCostWeights()

    def process(
        self,
        expected_phonemes: List[ExpectedPhonemes],
        actual_phonemes: ActualPhonemes,
        asr_result: ASRResult,
        reference_text: str,
        session_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> PronunciationReport:

        expected_flat: List[str] = []
        # word_of[i] = which word (index into expected_phonemes) expected_flat[i] belongs to
        word_of: List[int] = []
        for w_idx, exp in enumerate(expected_phonemes):
            for p in exp.phonemes:
                expected_flat.append(p)
                word_of.append(w_idx)

        actual_flat = actual_phonemes.phonemes

        if not expected_flat:
            return self._empty_report(asr_result, reference_text, session_id, candidate_id)

        # -- Single global weighted alignment --------------------------------
        alignment, total_cost = weighted_align(expected_flat, actual_flat, self.weights)

        # -- Worst-case cost for normalisation --------------------------------
        # Every expected phoneme deleted = worst possible outcome.
        # Normalising against the expected side only keeps overall_score
        # independent of how verbose/short the actual transcription was.
        worst_case = sum(deletion_cost(p, self.weights) for p in expected_flat)
        worst_case = max(worst_case, 1e-6)

        overall_score = round(100.0 * (1.0 - min(total_cost / worst_case, 1.0)), 1)

        # -- Per-phoneme scores + word grouping --------------------------------
        all_phoneme_scores: List[PhonemeScore] = []
        word_pairs: List[List[PhonemeScore]] = [[] for _ in expected_phonemes]

        exp_pointer = 0
        last_word_idx = 0

        for pos, (exp_p, act_p, cost) in enumerate(alignment):
            error_type = self._classify(exp_p, act_p)

            if exp_p is not None:
                w_idx = word_of[exp_pointer]
                exp_pointer += 1
                last_word_idx = w_idx
            else:
                # pure insertion -- attribute to the word we're currently "in"
                w_idx = last_word_idx

            posterior = self._posterior(act_p, actual_phonemes, pos)

            max_possible = self.weights.max_cost
            phoneme_score_val = 100.0 * (1.0 - min(cost / max_possible, 1.0))

            score = PhonemeScore(
                expected=exp_p or "",
                actual=act_p or "",
                error_type=error_type,
                word=expected_phonemes[w_idx].word if expected_phonemes else "",
                position=pos,
                confidence=1.0 if error_type == PhonemeErrorType.CORRECT else max(0.0, posterior - 0.1),
                posterior=posterior,
                gop_score=phoneme_score_val / 100.0,
            )
            all_phoneme_scores.append(score)
            word_pairs[w_idx].append(score)

        # -- Per-word scores --------------------------------------------------
        word_scores: List[WordScore] = []
        for w_idx, exp in enumerate(expected_phonemes):
            pairs = word_pairs[w_idx]
            error_count = sum(1 for p in pairs if p.is_error)
            word_score_val = self._word_score(pairs)
            word_scores.append(WordScore(
                word=exp.word, score=word_score_val,
                phoneme_scores=pairs, error_count=error_count,
            ))

        # -- Sequence-level similarity (unweighted, for reference) ------------
        edit_distance = self._unweighted_edit_distance(expected_flat, actual_flat)
        max_len = max(len(expected_flat), len(actual_flat), 1)
        sequence_score = round(100.0 * (1.0 - edit_distance / max_len), 1)

        return PronunciationReport(
            session_id=session_id,
            candidate_id=candidate_id,
            # reference_text=reference_text,
            # transcribed_text=asr_result.raw_text,
            language_detected=asr_result.language_detected,
            overall_score=overall_score,
            sequence_score=sequence_score,
            # word_scores=word_scores,
            # phoneme_scores=all_phoneme_scores,
            # expected_phonemes_flat=expected_flat,
            # actual_phonemes_flat=actual_flat,
            phoneme_edit_distance=edit_distance,
        )

    # -- Internals ----------------------------------------------------------

    @staticmethod
    def _classify(exp: Optional[str], act: Optional[str]) -> PhonemeErrorType:
        if exp is None:
            return PhonemeErrorType.INSERTION
        if act is None:
            return PhonemeErrorType.DELETION
        if exp == act:
            return PhonemeErrorType.CORRECT
        return PhonemeErrorType.SUBSTITUTION

    @staticmethod
    def _posterior(phoneme: Optional[str], actual: ActualPhonemes, position: int) -> float:
        if phoneme is None or actual.posteriors is None or len(actual.posteriors) == 0:
            return 0.5
        frame = min(position, len(actual.posteriors) - 1)
        row = actual.posteriors[frame]
        return float(row.max()) if row.max() > 0 else 0.5

    def _word_score(self, phoneme_scores: List[PhonemeScore]) -> float:
        if not phoneme_scores:
            return 100.0
        mean = float(np.mean([p.gop_score for p in phoneme_scores])) * 100.0
        return round(float(np.clip(mean, 0, 100)), 1)

    @staticmethod
    def _unweighted_edit_distance(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    @staticmethod
    def _empty_report(asr_result, reference_text, session_id, candidate_id) -> PronunciationReport:
        return PronunciationReport(
            session_id=session_id, candidate_id=candidate_id,
            # reference_text=reference_text, transcribed_text=asr_result.raw_text,
            language_detected=asr_result.language_detected,
            overall_score=0.0, sequence_score=0.0,
            # word_scores=[], phoneme_scores=[],
            # expected_phonemes_flat=[], actual_phonemes_flat=[],
            phoneme_edit_distance=0,
        )
