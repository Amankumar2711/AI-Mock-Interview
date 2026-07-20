"""
config_llm/settings.py — compatibility shim for eval_workers.py imports.

eval_workers.py does:
    from config_llm.settings import TaskType, settings

TaskType lives in models_llm.models.
settings (AppConfig singleton) lives in config_llm.config.

This module re-exports both so eval_workers.py can use its existing
import path without changes. It also adds the flat Celery attributes
that eval_workers.py reads directly off `settings`:
    settings.CELERY_TASK_HARD_TIME_LIMIT
    settings.CELERY_MAX_RETRIES
    settings.CELERY_RETRY_BACKOFF
    settings.active_task_types

These are derived from the nested CeleryConfig already in AppConfig.
"""

from __future__ import annotations

from models_llm.models import TaskType  # noqa: F401  — re-export
from config_llm.config import settings as _base_settings, AppConfig
from models_llm.models import TaskType as _TaskType


class _SettingsWithCeleryShortcuts(AppConfig):
    """
    Thin wrapper around AppConfig that exposes flat Celery attributes
    so eval_workers.py can do settings.CELERY_MAX_RETRIES without
    going through settings.celery.max_retries.
    """

    @property
    def CELERY_TASK_HARD_TIME_LIMIT(self) -> int:
        return self.celery.task_time_limit

    @property
    def CELERY_MAX_RETRIES(self) -> int:
        return 3

    @property
    def CELERY_RETRY_BACKOFF(self) -> int:
        return 30

    @property
    def active_task_types(self) -> list[_TaskType]:
        """All TaskType values — one Celery task registered per type."""
        return list(_TaskType)


# Re-create settings as the enhanced wrapper.
# This reads from the same env vars as AppConfig (env_prefix="APP_", .env file).
settings = _SettingsWithCeleryShortcuts()

__all__ = ["TaskType", "settings"]