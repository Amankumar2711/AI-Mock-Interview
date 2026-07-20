import hashlib
import json
import redis
from config_llm.settings import REDIS_CONFIG


class EvalCache:
    """
    Caches LLM evaluation results keyed by (task_type, system_prompt_hash, user_message_hash).
    Avoids re-evaluating identical (transcript, question) pairs — common with retries
    or duplicate submissions.
    """

    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_CONFIG.host,
            port=REDIS_CONFIG.port,
            db=REDIS_CONFIG.db_cache,
            decode_responses=True,
        )
        self.ttl = REDIS_CONFIG.cache_ttl_sec

    @staticmethod
    def _make_key(task_type: str, system_prompt: str, user_message: str) -> str:
        h = hashlib.sha256()
        h.update(task_type.encode())
        h.update(system_prompt.encode())
        h.update(user_message.encode())
        return f"eval_cache:{h.hexdigest()}"

    def get(self, task_type: str, system_prompt: str, user_message: str):
        key = self._make_key(task_type, system_prompt, user_message)
        val = self.client.get(key)
        return json.loads(val) if val else None

    def set(self, task_type: str, system_prompt: str, user_message: str, result: dict):
        key = self._make_key(task_type, system_prompt, user_message)
        self.client.set(key, json.dumps(result), ex=self.ttl)


eval_cache = EvalCache()