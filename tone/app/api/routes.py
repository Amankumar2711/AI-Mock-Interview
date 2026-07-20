from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tone.app.services.tone_service import ToneAnalyzer
from tone.app.utils.logger import logger
import config



router = APIRouter()
tone_analyzer = ToneAnalyzer()


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "tone_prosody_pipeline"}


@router.post("/analyze/tone")
async def analyze_tone(
    audio: UploadFile = File(...),
    transcript: str = ""
):
    """
    Full tone/prosody analysis endpoint.
    Accepts audio file + optional transcript for filler detection.
    """
    os.makedirs(f"{config.AUDIO_DIR}/temp", exist_ok=True)
    temp_filename = f"{config.AUDIO_DIR}/temp/{uuid.uuid4()}.wav"

    try:
        with open(temp_filename, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        logger.info(f"Analyzing audio: {audio.filename}")

        # Core tone analysis (pitch + energy)
        tone_result = await tone_analyzer.analyze(temp_filename)


        # Use rich audio-specific feedback built in tone_service
        feedback = tone_result["feedback"]

        response = {
            "duration": tone_result["duration"],
            "pitch": tone_result["pitch"],
            "energy": tone_result["energy"],
            "emotion": tone_result["emotion"],
            "dominant_emotion": tone_result["emotion"]["dominant_emotion"],
            "tone_score": tone_result["tone_score"],
            "monotone": tone_result["monotone"],
            "negative_emotion_flag": tone_result["negative_emotion_flag"],
            "flag": tone_result["flag"],
            "feedback": feedback
        }

        logger.info(f"Analysis complete | tone_score: {tone_result['tone_score']} | flag: {tone_result['flag']}")
        return response

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_filename}: {e}")
