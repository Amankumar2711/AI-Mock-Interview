from fastapi import APIRouter
from confidence.app.services.audio_preprocessing import (
    preprocess_audio
)

router = APIRouter()

@router.post("/preprocess")
async def preprocess():

    path = "uploads/sample.wav"

    output = preprocess_audio(
        path
    )

    return {
        "processed_file": output
    }
