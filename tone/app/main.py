from fastapi import FastAPI

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.routes import router
from utils.logger import logger

app = FastAPI(
    title="Tone/Prosody Pipeline",
    version="1.0.0"
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("Tone/Prosody API started successfully.")
