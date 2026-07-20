"""
pitch_service.py — F0 pitch analysis.


"""

import numpy as np
import parselmouth


class PitchAnalyzer:

    def analyze(self, audio_path: str = None, *, y: np.ndarray = None, sr: int = None) -> dict:
        """
        Analyze pitch variation.

        Args:
            audio_path: Path to audio file (used only when y/sr not provided).
            y:          Pre-loaded audio as float32 numpy array (preferred).
            sr:         Sample rate of y (required when y is provided).

        Returns:
            dict with avg_pitch, min_pitch, max_pitch, variation, pitch_score.
        """
        try:
            if y is not None and sr is not None:
                # Fast path — construct parselmouth.Sound from numpy array.
                # parselmouth expects float64, 1D array.
                sound = parselmouth.Sound(
                    values=y.astype(np.float64),
                    sampling_frequency=float(sr),
                )
            elif audio_path is not None:
                # Fallback path — load from disk (used when called standalone).
                sound = parselmouth.Sound(audio_path)
            else:
                raise ValueError("PitchAnalyzer.analyze() requires either (y, sr) or audio_path.")

            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array["frequency"]
            pitch_values = pitch_values[pitch_values > 0]

            if len(pitch_values) == 0:
                return {
                    "avg_pitch": 0,
                    "min_pitch": 0,
                    "max_pitch": 0,
                    "variation": 0,
                    "pitch_score": 0,
                }

            avg_pitch = float(np.mean(pitch_values))
            min_pitch = float(np.min(pitch_values))
            max_pitch = float(np.max(pitch_values))
            variation = float(np.std(pitch_values))

            if variation < 30:
                pitch_score = 30
            elif variation < 50:
                pitch_score = 60
            elif variation < 80:
                pitch_score = 85
            else:
                pitch_score = 100

            return {
                "avg_pitch": round(avg_pitch, 2),
                "min_pitch": round(min_pitch, 2),
                "max_pitch": round(max_pitch, 2),
                "variation": round(variation, 2),
                "pitch_score": pitch_score,
            }

        except Exception as e:
            print(f"Pitch analysis error: {e}")
            return {
                "avg_pitch": 0,
                "min_pitch": 0,
                "max_pitch": 0,
                "variation": 0,
                "pitch_score": 0,
            }