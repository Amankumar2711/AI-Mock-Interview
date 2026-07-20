"""
Celery application factory.

One queue per TaskType keeps requests with the same system prompt grouped
together, making vLLM prefix-caching most effective.
"""

from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue

from config_llm.config import settings
from models_llm.models import TaskType

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

celery_app = Celery(
    "llm_eval",
    broker=settings.redis.url,
    backend=settings.redis.url,
)

# ---------------------------------------------------------------------------
# Task queues – one per TaskType
# ---------------------------------------------------------------------------

default_exchange = Exchange("eval", type="direct")

task_queues = [
    Queue(
        f"eval_{task_type.value}",
        exchange=default_exchange,
        routing_key=f"eval_{task_type.value}",
    )
    for task_type in TaskType
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Broker / backend
    broker_url=settings.redis.url,
    result_backend=settings.redis.url,
    result_expires=settings.redis.result_ttl,

    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Queues
    task_queues=task_queues,
    task_default_queue="eval_grammar_evaluation",
    task_default_exchange="eval",
    task_default_routing_key="eval_grammar_evaluation",

    # Reliability
    task_acks_late=settings.celery.task_acks_late,
    task_reject_on_worker_lost=settings.celery.task_reject_on_worker_lost,
    task_soft_time_limit=settings.celery.task_soft_time_limit,
    task_time_limit=settings.celery.task_time_limit,

    # Worker
    worker_concurrency=settings.celery.worker_concurrency,
    worker_max_tasks_per_child=settings.celery.max_tasks_per_child,
    worker_prefetch_multiplier=1,  # Important for long-running LLM tasks

    # Beat (optional scheduled tasks)
    beat_schedule={},
)