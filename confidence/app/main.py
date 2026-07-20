from fastapi import FastAPI
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.upload import router as upload_router

from api.preprocess import ( router as preprocess_router)

from api.transcribe import ( router as transcribe_router)

from api.egemaps import ( router as egemaps_router)

from api.interview import (router as interview_router )

from api.analyze import ( router as analyze_router)

app = FastAPI(
    title="AI Interview Analyzer"
)

app.include_router( upload_router)
app.include_router(preprocess_router)
app.include_router(transcribe_router)
app.include_router( egemaps_router)
app.include_router(interview_router)
app.include_router(analyze_router)



@app.get("/")
def home():
    return {
        "message":"AI Interview Analyzer Running"
    }
