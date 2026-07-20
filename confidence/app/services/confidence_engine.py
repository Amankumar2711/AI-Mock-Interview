"""
confidence_engine.py — computes the final confidence score from text and
audio features.


"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from confidence.app.services.audio_confidence_analyzer import analyze_audio_confidence


def compute_confidence_score(text_features: dict, audio_features: dict, duration: float) -> dict:
    score = 100.0
    word_count = text_features.get("word_count", 0)
    fillers = text_features.get("fillers", 0)
    meaningful_words = word_count - fillers

    if meaningful_words == 0:
        return {
            "final_score": 26.67,
            "wpm": 0,
            "wpm_penalty": 13.33,
            "filler_penalty": 0,
            "hedge_penalty": 0,
            "silence_penalty": 60.0,
            "audio_penalties": {
                "jitter_penalty": 0,
                "shimmer_penalty": 0,
                "pause_penalty": 0,
                "jitter_raw": 0,
                "shimmer_raw": 0,
                "pause_raw": 0,
            },
        }

    audio_penalties = analyze_audio_confidence(audio_features)
    score -= audio_penalties["jitter_penalty"]
    score -= audio_penalties["shimmer_penalty"]
    score -= audio_penalties["pause_penalty"]

    # Word Per Minute scoring
    wpm = (word_count / duration) * 60 if duration > 0 else 0

    wpm_penalty = 0
    if wpm < 90:
        wpm_penalty = min(13.33, (90 - wpm) * 0.5)
    elif wpm > 150:
        wpm_penalty = min(13.33, (wpm - 150) * 0.5)
    score -= wpm_penalty

    # Filler word penalty
    filler_ratio = fillers / word_count if word_count > 0 else 1
    filler_penalty = filler_ratio * 13.33
    score -= filler_penalty

    # Hedging word penalty
    hedging = text_features.get("hedging", 0)
    hedge_ratio = hedging / word_count if word_count > 0 else 1
    hedge_penalty = min(20.0, hedge_ratio * 20)
    score -= hedge_penalty

    final_score = max(0.0, min(100.0, score))

    return {
        "final_score": round(final_score, 2),
        "wpm": round(wpm, 2),
        "wpm_penalty": round(wpm_penalty, 2),
        "filler_penalty": round(filler_penalty, 2),
        "filler_ratio": filler_ratio,
        "hedge_penalty": round(hedge_penalty, 2),
        "hedge_ratio": hedge_ratio,
        "audio_penalties": audio_penalties,
    }