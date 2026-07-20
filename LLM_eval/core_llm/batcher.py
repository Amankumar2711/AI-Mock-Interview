"""
Multi-queue batching engine.

- One queue per task_type (== one system prompt group).
- Each queue flushes when:
    a) it reaches max_batch_size, OR
    b) max_wait_time_sec has elapsed since the oldest pending item.
- A background thread per queue (or a single dispatcher thread polling all
  queues) handles flushing -> calls VLLMEngine.generate_batch.

Designed to run inside a Celery worker process (one engine + batchers per worker).
"""
import threading
import time
import logging
from queue import Queue, Empty
from typing import Dict, List

from config_llm.config import BATCH_CONFIG, SYSTEM_PROMPTS
from llm_engine import VLLMEngine
from core_llm.cache import eval_cache

logger = logging.getLogger(__name__)


class TaskQueue:
    """Holds pending requests for one task_type and tracks oldest-enqueue time."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.system_prompt = SYSTEM_PROMPTS.get(task_type)
        self.queue: Queue = Queue()
        self.lock = threading.Lock()
        self._oldest_ts: float = None

    def put(self, item: dict):
        with self.lock:
            if self._oldest_ts is None:
                self._oldest_ts = time.time()
            self.queue.put(item)

    def size(self) -> int:
        return self.queue.qsize()

    def should_flush(self) -> bool:
        with self.lock:
            if self.queue.qsize() == 0:
                return False
            if self.queue.qsize() >= BATCH_CONFIG.max_batch_size:
                return True
            if self._oldest_ts and (time.time() - self._oldest_ts) >= BATCH_CONFIG.max_wait_time_sec:
                return True
            return False

    def drain(self, max_items: int) -> List[dict]:
        items = []
        with self.lock:
            for _ in range(min(max_items, self.queue.qsize())):
                try:
                    items.append(self.queue.get_nowait())
                except Empty:
                    break
            self._oldest_ts = time.time() if not self.queue.empty() else None
        return items


class BatchDispatcher:
    """
    Singleton dispatcher: owns one TaskQueue per configured system prompt,
    runs a polling loop that flushes ready queues into the vLLM engine.

    result_callback(request_id, result_dict) is invoked per item after inference.
    """
    _instance = None

    def __init__(self):
        self.queues: Dict[str, TaskQueue] = {
            t: TaskQueue(t) for t in SYSTEM_PROMPTS.task_types()
        }
        self.engine = VLLMEngine.get_instance()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"BatchDispatcher started for task types: {list(self.queues.keys())}")

    @classmethod
    def get_instance(cls) -> "BatchDispatcher":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def submit(self, task_type: str, request_id: str, user_message: str, result_event_store: dict):
        if task_type not in self.queues:
            raise ValueError(f"Unsupported task_type: {task_type}")
        result_event_store[request_id] = {"event": threading.Event(), "result": None}
        self.queues[task_type].put({
            "request_id": request_id,
            "user_message": user_message,
            "_store": result_event_store,
        })

    def _run(self):
        while not self._stop_event.is_set():
            for task_type, tq in self.queues.items():
                if tq.should_flush():
                    self._flush_queue(tq)
            time.sleep(BATCH_CONFIG.poll_interval_sec)

    def _flush_queue(self, tq: TaskQueue):
        items = tq.drain(BATCH_CONFIG.max_batch_size)
        if not items:
            return

        # Check cache first; only send cache-misses to LLM
        to_infer = []
        cached_results = []
        for it in items:
            cached = eval_cache.get(tq.task_type, tq.system_prompt, it["user_message"])
            if cached:
                cached_results.append((it, {**cached, "request_id": it["request_id"], "cached": True}))
            else:
                to_infer.append(it)

        inferred_results = []
        if to_infer:
            try:
                inferred_results = self.engine.generate_batch(tq.system_prompt, to_infer)
            except Exception as e:
                logger.exception(f"Inference failure for task_type={tq.task_type}")
                inferred_results = [
                    {"request_id": it["request_id"], "raw_output": "", "parsed_output": None,
                     "error": str(e)} for it in to_infer
                ]

        # Store inferred results in cache
        for it, res in zip(to_infer, inferred_results):
            if not res.get("error"):
                eval_cache.set(tq.task_type, tq.system_prompt, it["user_message"], {
                    "raw_output": res["raw_output"],
                    "parsed_output": res["parsed_output"],
                })

        # Resolve futures for all items
        for it, res in zip(to_infer, inferred_results):
            self._resolve(it, res)
        for it, res in cached_results:
            self._resolve(it, res)

    @staticmethod
    def _resolve(item: dict, result: dict):
        store = item["_store"][item["request_id"]]
        store["result"] = result
        store["event"].set()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)