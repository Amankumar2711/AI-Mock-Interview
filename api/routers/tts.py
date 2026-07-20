import io
import soundfile as sf
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Try to import Kokoro. If it fails, make sure you ran: pip install kokoro soundfile
try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None

router = APIRouter(prefix="/tts", tags=["TTS"])

# Initialize the pipeline globally so the model is only loaded once at startup.
# 'a' stands for American English.
if KPipeline:
    pipeline = KPipeline(lang_code='a') 
else:
    pipeline = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella" # Default Kokoro voice
    speed: float = 1.0

@router.post("/generate")
async def generate_speech(request: TTSRequest):
    if not pipeline:
         raise HTTPException(status_code=500, detail="Kokoro TTS pipeline could not be loaded. Please install dependencies.")

    try:
        # Generate audio using Kokoro
        # The generator yields tuples of (graphemes, phonemes, audio)
        generator = pipeline(request.text, voice=request.voice, speed=request.speed)
        
        audio_chunks = []
        sample_rate = 24000 # Kokoro's default sample rate
        
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                audio_chunks.append(audio)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
            
        # Combine all audio chunks
        final_audio = np.concatenate(audio_chunks)
        
        # Convert the raw numpy array to a WAV file in memory
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        # Return the WAV file as a streaming response
        return StreamingResponse(buffer, media_type="audio/wav")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))