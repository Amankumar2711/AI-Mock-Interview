"""
tone_service.py — orchestrates pitch, energy, and emotion analysis.


"""

import asyncio
import librosa
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tone.app.services.pitch_service import PitchAnalyzer
from tone.app.services.energy_service import EnergyAnalyzer
from tone.app.services.emotion_service import EmotionAnalyzer

# ── Scoring thresholds ────────────────────────────────────────────────────────
PITCH_MONOTONE_HZ  = 30.0
PITCH_LOW_HZ       = 50.0
PITCH_HIGH_HZ      = 100.0
ENERGY_LOW         = 0.05
ENERGY_MID         = 0.10
ENERGY_HIGH        = 0.20
NEG_FLAG_THRESHOLD = 0.60


class ToneAnalyzer:

    def __init__(self):
        self.pitch_analyzer   = PitchAnalyzer()
        self.energy_analyzer  = EnergyAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()  # SpeechBrain model loaded once here

    # ── main entry ────────────────────────────────────────────────────────────
    async def analyze(self, audio_path: str) -> dict:
        """
        Run pitch, energy, and emotion analysis on an audio file.

        Loads audio exactly ONCE at 16kHz, then passes the numpy array to
        all three sub-analyzers concurrently via asyncio.gather.
        Duration is computed from array length — no extra librosa.load.
        """
        # ── Single audio load (was 4 separate loads before this fix) ─────────
        try:
            y, sr = await asyncio.to_thread(librosa.load, audio_path, sr=16000, mono=True)
        except Exception as e:
            # If load fails entirely, return a safe zero-score result.
            raise RuntimeError(f"ToneAnalyzer: failed to load audio '{audio_path}': {e}") from e

        # Duration from array — no extra disk read.
        duration = len(y) / sr

        # ── Fan-out: all three run in parallel, sharing the same numpy array ──
        # Each sub-analyzer receives (y=..., sr=...) — no file path passed.
        pitch_result, energy_result, emotion_result = await asyncio.gather(
            asyncio.to_thread(self.pitch_analyzer.analyze,   y=y, sr=sr),
            asyncio.to_thread(self.energy_analyzer.analyze,  y=y, sr=sr),
            asyncio.to_thread(self.emotion_analyzer.analyze, y=y, sr=sr),
        )

        pv  = pitch_result["variation"]
        ev  = energy_result["energy_variation"]
        dom = emotion_result["dominant_emotion"]
        neg = emotion_result["negative_ratio"]

        # ── Flags ─────────────────────────────────────────────────────────────
        monotone              = pv < PITCH_MONOTONE_HZ
        negative_emotion_flag = neg > NEG_FLAG_THRESHOLD

        flags = []
        if monotone:
            flags.append("monotone")
        if negative_emotion_flag:
            flags.append("negative_emotion")
        flag_str = ", ".join(flags) if flags else "none"

        # ── Prosody Scorer ────────────────────────────────────────────────────
        score = 100

        # Pitch
        if pv < PITCH_MONOTONE_HZ:
            score -= 20
        elif pv < PITCH_LOW_HZ:
            score -= 10
        elif pv >= PITCH_HIGH_HZ:
            score += 5

        # Energy
        if ev < ENERGY_LOW:
            score -= 15
        elif ev < ENERGY_MID:
            score -= 8
        elif ev >= ENERGY_HIGH:
            score += 5

        # Emotion
        if negative_emotion_flag:
            score -= 20
        elif dom == "negative":
            score -= 10
        elif dom == "neutral":
            score -= 8

        score = max(0, min(100, int(score)))

        feedback = self._build_feedback(pv, ev, dom, negative_emotion_flag, monotone, score, neg)

        return {
            "duration":              round(duration, 2),
            "pitch":                 pitch_result,
            "energy":                energy_result,
            "emotion":               emotion_result,
            "tone_score":            score,
            "monotone":              monotone,
            "negative_emotion_flag": negative_emotion_flag,
            "flag":                  flag_str,
            "feedback":              feedback,
        }

    # ── feedback builder ──────────────────────────────────────────────────────
    @staticmethod
    def _build_feedback(pv, ev, dom, neg_flag, monotone, score, neg_ratio):
        fb = []

        if monotone:
            fb.append(
                f"[WARN] Pitch is very monotone (variation = {pv:.1f} Hz). "
                "Try raising and lowering your voice more to engage the listener."
            )
        elif pv < PITCH_LOW_HZ:
            fb.append(
                f"Pitch variation is below average ({pv:.1f} Hz). "
                "Add more expressive highs and lows in your speech."
            )
        elif pv >= PITCH_HIGH_HZ:
            fb.append(
                f"[OK] Excellent pitch variation ({pv:.1f} Hz) -- your voice is very expressive."
            )
        else:
            fb.append(
                f"[OK] Good pitch variation ({pv:.1f} Hz). "
                "You can push for even more expressiveness."
            )

        if ev < ENERGY_LOW:
            fb.append(
                f"[WARN] Energy dynamics are very flat (RMS variance = {ev:.4f}). "
                "Speak louder on important words and softer on others."
            )
        elif ev < ENERGY_MID:
            fb.append(
                f"Energy dynamics are moderate (RMS variance = {ev:.4f}). "
                "Add a bit more vocal contrast to keep listeners engaged."
            )
        elif ev >= ENERGY_HIGH:
            fb.append(
                f"[OK] Strong vocal energy dynamics ({ev:.4f}) -- great delivery!"
            )
        else:
            fb.append(
                f"[OK] Energy dynamics are good ({ev:.4f})."
            )

        if neg_flag:
            fb.append(
                f"[WARN] Emotional tone detected as negative in {neg_ratio:.0%} of audio segments. "
                "Try to sound warmer and more enthusiastic."
            )
        elif dom == "negative":
            fb.append(
                "Some segments sounded negative in tone. "
                "Focus on sounding more positive and engaged."
            )
        elif dom == "neutral":
            fb.append(
                "Tone is mostly neutral. A little more enthusiasm will make a stronger impression."
            )
        elif dom == "positive":
            fb.append("[OK] Positive emotional tone detected -- great energy!")

        if score >= 85:
            fb.append("[BEST] Overall: Excellent vocal delivery!")
        elif score >= 70:
            fb.append("[OK] Overall: Good vocal delivery with room for improvement.")
        elif score >= 55:
            fb.append("Overall: Average delivery -- focus on the areas above.")
        else:
            fb.append("[WARN] Overall: Significant improvement needed in tone and delivery.")

        return fb