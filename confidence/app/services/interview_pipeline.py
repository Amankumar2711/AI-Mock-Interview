"""
interview_pipeline.py — confidence evaluation pipeline entry point.


"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import librosa

from confidence.app.services.egemaps_extractor import extract_egemaps
from confidence.app.services.text_feature_extractor import extract_text_features
from confidence.app.services.confidence_engine import compute_confidence_score
from confidence.app.services.recommendation_engine import generate_recommendation
from utils.logger import get_logger

logger = get_logger()


def get_audio_duration(file_path: str) -> float:
    """
    Return audio duration in seconds.

    Uses librosa.get_duration(path=...) which reads duration from file
    metadata without full decoding and supports all common formats:
    .wav, .mp3, .m4a, .flac, .ogg, etc.

    Falls back to 0.0 on any error so the confidence pipeline can
    still run (with a degraded WPM score) rather than crashing entirely.
    """
    try:
        return librosa.get_duration(path=file_path)
    except Exception as e:
        logger.warning(f"get_audio_duration failed for '{file_path}': {e} — defaulting to 0.0s")
        return 0.0


def run_pipeline(audio_path: str, transcript: str) -> dict:
    logger.info(f"Starting pipeline for audio: {audio_path}")

    duration = get_audio_duration(audio_path)

    logger.debug("Extracting acoustic features via OpenSMILE...")
    audio_features_result = extract_egemaps(audio_path)
    audio_features = audio_features_result["features"]

    # Cleanup egemaps temp file
    try:
        feature_file = audio_features_result.get("feature_file")
        if feature_file and os.path.exists(feature_file):
            os.remove(feature_file)
            logger.debug(f"Cleaned up extracted features file: {feature_file}")
    except Exception as e:
        logger.warning(f"Failed to clean up extracted features: {e}")

    text_features = extract_text_features(transcript)
    confidence_data = compute_confidence_score(text_features, audio_features, duration)
    recommendation = generate_recommendation(confidence_data["final_score"])

    return {
        # "transcript": transcript,
        # "audio_features": audio_features,
        # "text_features": text_features,
        "confidence_data": confidence_data,
        "recommendation": recommendation,
    }