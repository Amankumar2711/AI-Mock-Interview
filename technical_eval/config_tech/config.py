SEMANTIC_SUB_WEIGHTS = {
    "similarity":    0.45,
    "keyword":       0.35,
    "vocab":         0.0,
    "length":        0.20
}

# ---------------------------------------------------------------------------
# Embedding model — env-aware routing
# ---------------------------------------------------------------------------
# Dev  (APP_ENV=dev):  local all-MiniLM-L6-v2 SentenceTransformer.
#                      Fast, no GPU, no network calls.
# Prod (APP_ENV=prod): Qwen2.5-7B-Instruct embedding endpoint served by
#                      vLLM at VLLM_BASE_URL/embeddings.
#                      Set EMBEDDING_API_URL=http://<vllm-host>:8000/v1/embeddings
#                      in your .env for production.
# ---------------------------------------------------------------------------
import os as _os

_APP_ENV = _os.getenv("APP_ENV", "dev").lower()

if _APP_ENV == "prod":
    # Production: use Qwen model served by vLLM for embeddings.
    # EMBEDDING_MODEL_NAME must match the model loaded in vLLM.
    EMBEDDING_MODEL_NAME: str = _os.getenv(
        "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"
    )
    # Build the embedding URL from VLLM_HOST + VLLM_PORT (set to 'vllm:8001' by
    # docker-compose), matching the same routing fix in final_integration/config.py.
    EMBEDDING_API_URL: str = _os.getenv(
        "EMBEDDING_API_URL",
        f"http://{_os.getenv('VLLM_HOST', 'localhost')}:{_os.getenv('VLLM_PORT', '8001')}/v1/embeddings",
    )
    EMBEDDING_API_KEY: str    = _os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_SERVER_URL: str = _os.getenv("EMBEDDING_SERVER_URL", "")
    # Hard-fail if vLLM embedding endpoint is unreachable — do NOT silently
    # fall back to a local model in prod (would load an extra large model into
    # a worker that already holds Whisper + SpeechBrain + Wav2Vec2).
    ALLOW_LOCAL_FALLBACK: bool = (
        _os.getenv("ALLOW_LOCAL_FALLBACK", "false").lower() != "false"
    )
else:
    # Development: local SentenceTransformer — no server required.
    EMBEDDING_MODEL_NAME: str = _os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_API_URL: str    = _os.getenv("EMBEDDING_API_URL", "")
    EMBEDDING_API_KEY: str    = _os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_SERVER_URL: str = _os.getenv("EMBEDDING_SERVER_URL", "")
    # Local fallback always enabled in dev so offline dev works without vLLM.
    ALLOW_LOCAL_FALLBACK: bool = (
        _os.getenv("ALLOW_LOCAL_FALLBACK", "true").lower() != "false"
    )

DOMAIN_VOCAB = {
    "python": [
        "iterator", "generator", "decorator", "comprehension",
        "lambda", "closure", "mutable", "immutable", "GIL",
        "threading", "asyncio", "context manager"
    ],
    "ml": [
        "gradient", "backpropagation", "overfitting", "regularization",
        "embedding", "inference", "epoch", "batch", "loss function",
        "activation", "neural network", "feature engineering"
    ],
    "database": [
        "normalization", "indexing", "transaction", "ACID",
        "sharding", "replication", "foreign key", "join",
        "query optimization", "schema", "NoSQL", "ORM"
    ],
    "dsa": [
        "complexity", "Big O", "recursion", "dynamic programming",
        "graph", "tree", "heap", "stack", "queue",
        "binary search", "sorting", "hashing"
    ],
    "system_design": [
        "load balancer", "cache", "microservices", "API gateway",
        "message queue", "CDN", "horizontal scaling", "vertical scaling",
        "database sharding", "consistency", "availability", "partition"
    ],
    "general": [
        "problem solving", "communication", "teamwork",
        "deadline", "leadership", "agile", "debugging",
        "documentation", "version control", "code review"
    ]
}
