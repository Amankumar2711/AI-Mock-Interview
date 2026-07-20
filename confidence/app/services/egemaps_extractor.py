"""
egemaps_extractor.py — extracts eGeMAPS acoustic features via OpenSMILE.


"""

import json
import os
import sys
import threading

import opensmile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.logger import get_logger

logger = get_logger()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FEATURE_DIR = os.path.join(PROJECT_ROOT, "extracted_features")
os.makedirs(FEATURE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy singleton for opensmile.Smile
# ---------------------------------------------------------------------------
_smile: opensmile.Smile | None = None
_smile_lock = threading.Lock()


def _get_smile() -> opensmile.Smile:
    """
    Return the shared opensmile.Smile instance, creating it on first call.
    Thread-safe via double-checked locking — only the first call pays the
    init cost; all subsequent calls are a single null-check (no lock).
    """
    global _smile
    if _smile is not None:
        return _smile
    with _smile_lock:
        if _smile is None:
            logger.info("Initializing opensmile.Smile (eGeMAPSv02)...")
            _smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            logger.info("opensmile.Smile initialized.")
    return _smile


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_egemaps(audio_path: str) -> dict:
    """
    Extract eGeMAPSv02 functionals from an audio file.

    Returns a dict with keys:
        feature_file — path of the JSON dump written to FEATURE_DIR
        features     — dict mapping feature name → float value
    """
    logger.debug(f"OpenSMILE analyzing {audio_path}")

    smile = _get_smile()
    features_df = smile.process_file(audio_path)
    feature_dict = features_df.iloc[0].to_dict()

    filename = os.path.splitext(os.path.basename(audio_path))[0] + "_egemaps.json"
    output_path = os.path.join(FEATURE_DIR, filename)

    with open(output_path, "w") as f:
        json.dump(feature_dict, f, indent=4, default=float)

    logger.info(f"OpenSMILE extraction successful: {output_path}")

    return {
        "feature_file": output_path,
        "features": feature_dict,
    }