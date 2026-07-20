"""Typed result objects flowing through the simplified pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np


class PhonemeErrorType(str, Enum):
    CORRECT = "correct"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


# ── ASR (WhisperX) ─────────────────────────────────────────────────────────

@dataclass
class WordToken:
    word: str
    start: float
    end: float
    confidence: float

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.4


@dataclass
class ASRResult:
    raw_text: str
    words: List[WordToken]
    language_detected: str


# ── G2P (expected phonemes) ───────────────────────────────────────────────

@dataclass
class ExpectedPhonemes:
    word: str
    phonemes: List[str]    # ARPABET


# ── Wav2Vec2 (actual phonemes) ────────────────────────────────────────────

@dataclass
class ActualPhonemes:
    phonemes: List[str]
    posteriors: Optional[np.ndarray] = None   # (T, vocab)


# ── MFA alignment ──────────────────────────────────────────────────────────

@dataclass
class AlignedPhoneme:
    phoneme: str
    start: float
    end: float
    word: str

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000


@dataclass
class AlignmentResult:
    aligned_phonemes: List[AlignedPhoneme]
    success: bool
    error_message: Optional[str] = None


# ── Scoring ────────────────────────────────────────────────────────────────

@dataclass
class PhonemeScore:
    expected: str
    actual: str
    error_type: PhonemeErrorType
    word: str
    position: int
    # duration_ms: float
    confidence: float
    posterior: float
    gop_score: float

    @property
    def is_error(self) -> bool:
        return self.error_type != PhonemeErrorType.CORRECT


@dataclass
class WordScore:
    word: str
    score: float
    phoneme_scores: List[PhonemeScore]
    error_count: int

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0


@dataclass
class PronunciationReport:
    session_id: Optional[str]
    candidate_id: Optional[str]
    # reference_text: str
    # transcribed_text: str
    language_detected: str
    sequence_score: float
    overall_score: float
    # word_scores: Optional[List[WordScore]]
    # phoneme_scores: Optional[List[PhonemeScore]]
    # expected_phonemes_flat: Optional[List[str]]
    # actual_phonemes_flat: Optional[List[str]]
    phoneme_edit_distance: int
    processing_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def error_summary(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for ws in self.word_scores:
            if ws.has_errors:
                errs = [
                    f"{ps.expected}\u2192{ps.actual}" if ps.error_type == PhonemeErrorType.SUBSTITUTION
                    else f"deleted:{ps.expected}" if ps.error_type == PhonemeErrorType.DELETION
                    else f"inserted:{ps.actual}"
                    for ps in ws.phoneme_scores if ps.is_error
                ]
                out[ws.word] = errs
        return out
