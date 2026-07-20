# LLM Evaluation Service

Production-ready AI interview evaluation system with:
- **Per-task-type batch queues** (time + size based flushing)
- **vLLM** for high-throughput inference (prefix-caching for shared system prompts)
- **Celery + Redis** for async task queuing, retries, and result storage
- **Dev / Prod modes** switchable via a single env var

---

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │              FastAPI Service                 │
                     │  POST /eval/submit  →  EvalOrchestrator     │
                     └──────────────┬──────────────────────────────┘
                                    │ routes by task_type
           ┌────────────────────────▼──────────────────────────┐
           │             EvalOrchestrator                       │
           │  BatchManager(grammar) │ BatchManager(fluency) … │
           │   flush on size/time  │ flush on size/time   … │
           └────────────────────────┬──────────────────────────┘
                                    │ Celery .apply_async()
           ┌────────────────────────▼──────────────────────────┐
           │  Redis (broker + result backend)                   │
           │  Queue per task_type: eval_grammar_evaluation …   │
           └────────────────────────┬──────────────────────────┘
                                    │ consumed by
           ┌────────────────────────▼──────────────────────────┐
           │  Celery Worker  →  process_eval_batch()           │
           │  → LLMClient.infer_batch()                        │
           │  → POST /v1/chat/completions (vLLM)               │
           │  → cache_result() → Redis                         │
           └───────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Copy and edit environment

```bash
cp .env.example .env
# Set APP_RUN_MODE=dev for local testing
```

### 2. Start Redis

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### 3. Start vLLM

```bash
# DEV (quantised – ~6GB VRAM, works on a gaming laptop)
bash scripts/start_vllm_dev.sh

# PROD (full-precision)
bash scripts/start_vllm_prod.sh
```

Or use the cloud OpenAI endpoint for dev:
```bash
APP_BACKEND=openai OPENAI_API_KEY=sk-... uvicorn api.app:app
```

### 4. Start Celery worker

```bash
bash scripts/start_worker.sh
```

### 5. Start the API

```bash
uvicorn api.app:app --reload --port 9000
```

### 6. Full stack via Docker Compose

```bash
docker compose up --build
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `APP_RUN_MODE` | `dev` | `dev` or `production` |
| `APP_BACKEND` | `vllm` | `vllm` or `openai` |
| `BATCH_MAX_BATCH_SIZE` | `32` | Max requests per batch |
| `BATCH_MAX_WAIT_SECONDS` | `5.0` | Max wait before auto-flush |
| `BATCH_MAX_CONCURRENT_BATCHES` | `4` | Back-pressure per queue |
| `VLLM_DEV_MODEL` | GPTQ Mistral 7B | Quantised model for dev |
| `VLLM_DEV_QUANTIZATION` | `gptq` | Quantisation backend |
| `VLLM_PROD_MODEL` | Mistral 7B Instruct | Full-precision prod model |
| `REDIS_RESULT_TTL` | `3600` | Result cache TTL (seconds) |
| `CELERY_WORKER_CONCURRENCY` | `4` | Celery worker threads |

---

## Adding a New Evaluation Task

1. Add your system prompt to `config/config.py → SYSTEM_PROMPTS`
2. Add the corresponding value to `core/models.py → TaskType`
3. That's it – a new BatchManager and Celery queue are created automatically.

---

## API Usage

```bash
# Submit a single evaluation
curl -X POST http://localhost:9000/eval/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "grammar_evaluation",
    "user_message": "Question: Describe your day.\nTranscript: I goes to school today and learn many thing.",
    "metadata": {"student_id": "stu_001", "session_id": "ses_abc"}
  }'
# → {"request_id": "...", "status": "queued", ...}

# Poll for result
curl http://localhost:9000/eval/result/<request_id>

# Submit a batch of 3 evaluations in one call
curl -X POST http://localhost:9000/eval/submit/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"task_type": "grammar_evaluation",  "user_message": "..."},
      {"task_type": "fluency_evaluation",  "user_message": "..."},
      {"task_type": "content_evaluation",  "user_message": "..."}
    ]
  }'
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```