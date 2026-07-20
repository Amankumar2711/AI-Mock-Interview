"""
energy_service.py — RMS energy dynamics analysis.


"""

import numpy as np
import librosa


class EnergyAnalyzer:

    def analyze(self, audio_path: str = None, *, y: np.ndarray = None, sr: int = None) -> dict:
        """
        Analyze RMS energy dynamics.

        Args:
            audio_path: Path to audio file (used only when y/sr not provided).
            y:          Pre-loaded audio as float32 numpy array (preferred).
            sr:         Sample rate of y (required when y is provided).

        Returns:
            dict with mean_energy, energy_variation, energy_score.
        """
        try:
            if y is None or sr is None:
                if audio_path is None:
                    raise ValueError("EnergyAnalyzer.analyze() requires either (y, sr) or audio_path.")
                # Fallback path — load from disk (used when called standalone).
                y, sr = librosa.load(audio_path, sr=None)

            rms = librosa.feature.rms(y=y)[0]
            mean_energy = float(np.mean(rms))
            energy_variation = float(np.std(rms))

            if mean_energy < 0.02:
                energy_score = 20
            elif mean_energy < 0.05:
                energy_score = 50
            elif mean_energy < 0.15:
                energy_score = 90
            else:
                energy_score = 70

            return {
                "mean_energy": round(mean_energy, 4),
                "energy_variation": round(energy_variation, 4),
                "energy_score": energy_score,
            }

        except Exception as e:
            print(f"Energy analysis error: {e}")
            return {
                "mean_energy": 0.0,
                "energy_variation": 0.0,
                "energy_score": 0,
            }