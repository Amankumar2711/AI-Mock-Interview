from fastapi import APIRouter

from confidence.app.services.egemaps_extractor import (
    extract_egemaps
)

router = APIRouter()

@router.post("/egemaps")
async def egemaps():

    audio_path = (
        "processed_audio/sample.wav"
    )

    result = extract_egemaps(
        audio_path
    )

    return result
