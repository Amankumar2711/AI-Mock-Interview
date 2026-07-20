"""
FastAPI application factory with lifespan hooks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_llm.routes import router
from batching.orchestrator import orchestrator
from config_llm.config import settings
from utils_llm.logger import configure_logging, get_logger

logger = get_logger(__name__)   


@asynccontextmanager
async def lifespan(app: FastAPI):  
    # ---- Startup ----
    configure_logging(log_level=settings.log_level, dev=settings.is_dev)
    logger.info(
        "app_starting",
        mode=settings.run_mode.value,
        backend=settings.backend.value,
        num_task_types=settings.num_system_prompts,
    )
    yield
    # ---- Shutdown ----
    logger.info("app_shutting_down")
    await orchestrator.flush_all()
    logger.info("app_shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Evaluation Service",
        version="1.0.0",
        description=(
            "AI-powered student evaluation API. "
            "Batches requests per task type and processes them via vLLM + Celery."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()