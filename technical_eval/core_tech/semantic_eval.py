"""
semantic_eval.py — semantic relevance scoring for technical evaluation.


"""

import os
import sys
import asyncio
import threading

import aiohttp
import numpy as np
import requests
from huggingface_hub import snapshot_download

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import logger
import technical_eval.config_tech.config as config

# ---------------------------------------------------------------------------
# Local embedding model — pinned in ModelRegistry (loaded once per process)
# ---------------------------------------------------------------------------
_local_embedding_model = None   # kept for backward-compat reference; real singleton in registry


def load_local_model():
    """
    Return the local SentenceTransformer embedding model.

    Delegates to ModelRegistry.get_semantic_model() so the model is
    loaded exactly once per process and pinned in RAM, regardless of
    how many tasks call this function concurrently.
    """
    global _local_embedding_model
    if _local_embedding_model is not None:
        return _local_embedding_model  # fast-path (process-local cache)

    try:
        # Resolve PROJECT_ROOT so we can locate core_main.registry even
        # when this module is imported in a subprocess with a minimal path.
        import os as _os, sys as _sys
        _tech_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        _project_root = _os.path.dirname(_tech_root)
        for _p in [_project_root, _tech_root]:
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from core_main.registry import ModelRegistry  # noqa: PLC0415
    except ImportError:
        ModelRegistry = None

    def _do_load():
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            import torch as _torch  # noqa: PLC0415

            model_name = config.EMBEDDING_MODEL_NAME
            # SentenceTransformer targets GPU — VRAM is shared with WhisperX,
            # but the embedding model is small (< 100 MB) so this is safe.
            _st_device = "cuda" if _torch.cuda.is_available() else "cpu"

            model_source = model_name
            cache_dir = os.environ.get("HF_MODELS_CACHE", "/app/models/hf_cache")

            if os.environ.get("APP_ENV", "dev").lower() == "prod":
                # Attempt 1: load from pre-baked cache (offline-first).
                try:
                    model_source = snapshot_download(
                        repo_id=model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                    )
                    logger.info(f"SentenceTransformer: loaded from pre-baked cache: {model_source}")
                except Exception as _snap_err:
                    # Attempt 2: download from Hub (network fallback).
                    logger.warning(
                        f"SentenceTransformer: offline load failed ({_snap_err}). "
                        "Falling back to network download."
                    )
                    model_source = snapshot_download(
                        repo_id=model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                    )

            logger.info(f"Loading SentenceTransformer from: {model_source} on {_st_device}")
            model = SentenceTransformer(model_source, device=_st_device)
            logger.info("SentenceTransformer initialised successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise

    if ModelRegistry is not None:
        _local_embedding_model = ModelRegistry.get_semantic_model(_do_load)
    else:
        # Fallback: no registry available — load directly (tests / standalone)
        _local_embedding_model = _do_load()

    return _local_embedding_model


# ---------------------------------------------------------------------------
# Shared async aiohttp session — one per process, reused across all calls
# ---------------------------------------------------------------------------
_async_session: aiohttp.ClientSession | None = None
_session_lock: asyncio.Lock | None = None
_session_init_lock = threading.Lock()   # guards _session_lock creation itself


def _get_session_lock() -> asyncio.Lock:
    """Return (and lazily create) the asyncio.Lock for session init."""
    global _session_lock
    if _session_lock is None:
        with _session_init_lock:
            if _session_lock is None:
                _session_lock = asyncio.Lock()
    return _session_lock


async def _get_async_session() -> aiohttp.ClientSession:
    """
    Return the shared aiohttp.ClientSession, creating it on first call.
    Uses asyncio.Lock so concurrent coroutines don't race to create multiple
    sessions. The session uses a connector with keep-alive enabled so TCP
    connections to the embedding API are reused across requests.
    """
    global _async_session
    if _async_session is not None and not _async_session.closed:
        return _async_session

    lock = _get_session_lock()
    async with lock:
        # Double-check after acquiring lock
        if _async_session is None or _async_session.closed:
            connector = aiohttp.TCPConnector(
                limit=20,               # max concurrent connections
                keepalive_timeout=30,   # keep TCP alive between calls
            )
            timeout = aiohttp.ClientTimeout(
                connect=5,
                total=15,
            )
            _async_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
            logger.info("Shared aiohttp.ClientSession created for embedding calls.")
    return _async_session


# ---------------------------------------------------------------------------
# Sync embedding (used by get_semantic_score)
# ---------------------------------------------------------------------------
def get_embedding(text: str):
    """
    Sync embedding retrieval.
    1. Remote API (OpenAI schema, e.g. Qwen3).
    2. TEI microservice.
    3. Local SentenceTransformer fallback.
    """
    if config.EMBEDDING_API_URL:
        try:
            headers = {"Content-Type": "application/json"}
            if config.EMBEDDING_API_KEY:
                headers["Authorization"] = f"Bearer {config.EMBEDDING_API_KEY}"
            response = requests.post(
                config.EMBEDDING_API_URL,
                headers=headers,
                json={"input": text, "model": config.EMBEDDING_MODEL_NAME},
                timeout=10,
            )
            if response.status_code == 200:
                res_data = response.json()
                embedding = res_data.get("data", [{}])[0].get("embedding")
                if embedding:
                    return np.array(embedding)
        except Exception as e:
            logger.warning(f"Embedding API failure: {e}. Falling back to TEI/Local.")

    if config.EMBEDDING_SERVER_URL:
        try:
            response = requests.post(
                config.EMBEDDING_SERVER_URL,
                json={"inputs": text},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if response.status_code == 200:
                res_data = response.json()
                if isinstance(res_data, list):
                    if isinstance(res_data[0], list):
                        return np.array(res_data[0])
                    return np.array(res_data)
        except Exception as e:
            logger.warning(f"Embedding microservice failure: {e}. Falling back to local.")

    if not config.ALLOW_LOCAL_FALLBACK:
        raise Exception("Local fallback disabled and remote embedding API failed/unavailable.")

    model = load_local_model()
    return model.encode(text)


# ---------------------------------------------------------------------------
# Async embedding (used by get_semantic_score_async)
# ---------------------------------------------------------------------------
async def get_embedding_async(text: str):
    """
    Async embedding retrieval using the shared aiohttp.ClientSession.
    TCP connections to the embedding API are reused across calls — no
    per-call session creation.
    """
    session = await _get_async_session()

    if config.EMBEDDING_API_URL:
        try:
            headers = {"Content-Type": "application/json"}
            if config.EMBEDDING_API_KEY:
                headers["Authorization"] = f"Bearer {config.EMBEDDING_API_KEY}"
            async with session.post(
                config.EMBEDDING_API_URL,
                headers=headers,
                json={"input": text, "model": config.EMBEDDING_MODEL_NAME},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    embedding = data.get("data", [{}])[0].get("embedding")
                    if embedding:
                        return np.array(embedding)
        except Exception as e:
            logger.warning(f"Async Embedding API failure: {e}")

    if config.EMBEDDING_SERVER_URL:
        try:
            async with session.post(
                config.EMBEDDING_SERVER_URL,
                json={"inputs": text},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    res_data = await response.json()
                    if isinstance(res_data, list):
                        if isinstance(res_data[0], list):
                            return np.array(res_data[0])
                        return np.array(res_data)
        except Exception as e:
            logger.warning(f"Async Embedding microservice failure: {e}")

    if not config.ALLOW_LOCAL_FALLBACK:
        raise Exception("Local fallback disabled")

    model = load_local_model()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, model.encode, text)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity_vectors(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Async semantic score — called by pipeline._run_semantic_eval()
# ---------------------------------------------------------------------------
async def get_semantic_score_async(question, answer, expected_keywords=None, domain="general", model_answer=None):
    """
    Async semantic relevance grading.
    Fetches both embeddings concurrently via asyncio.gather using the
    shared aiohttp session — no new connections per call.
    """
    if not answer or len(answer.strip()) < 5:
        logger.warning("Answer text too short for semantic grading.")
        return _empty_result("Answer too short or empty.")

    try:
        if model_answer and len(model_answer.strip()) > 0:
            target_comparison_text = model_answer
            logger.info(f"Async comparing answer with model answer: {target_comparison_text[:100]}...")
        elif expected_keywords and len(expected_keywords) > 0:
            target_comparison_text = ", ".join(expected_keywords)
            logger.info(f"Async comparing answer with expected keywords: {target_comparison_text}")
        else:
            target_comparison_text = question
            logger.info("Async comparing answer with question text.")

        # Both embedding calls share the same session — no new connections.
        target_emb, a_emb = await asyncio.gather(
            get_embedding_async(target_comparison_text),
            get_embedding_async(answer),
        )

        if target_emb is None or a_emb is None:
            logger.error("Async embedding retrieval failed.")
            similarity_score = 50.0
        else:
            similarity = cosine_similarity_vectors(target_emb, a_emb)
            similarity = max(0.0, similarity)
            similarity_score = round(similarity * 100, 2)

    except Exception as e:
        logger.error(f"Async Embedding scoring computation error: {e}")
        similarity_score = 50.0

    keyword_result = check_keywords(answer, expected_keywords)
    length_result = check_answer_length(answer)
    # domain_vocab_score = check_domain_vocab(answer, domain)

    sub_w = config.SEMANTIC_SUB_WEIGHTS
    combined = round(
        (similarity_score * sub_w["similarity"])
        + (keyword_result["score"] * sub_w["keyword"])
        # + (domain_vocab_score * sub_w["vocab"])
        + (length_result["score"] * sub_w["length"])
    )

    feedback = generate_semantic_feedback(similarity_score, keyword_result, length_result)

    return {
        "similarity_score": similarity_score,
        "keyword_score": keyword_result["score"],
        "keyword_coverage": keyword_result["coverage"],
        "keywords_found": keyword_result["found"],
        "keywords_missing": keyword_result["missing"],
        "length_score": length_result["score"],
        "word_count": length_result["word_count"],
        # "domain_vocab_score": domain_vocab_score,
        "combined_score": combined,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Sync semantic score — kept for backwards compat / fallback
# ---------------------------------------------------------------------------
def get_semantic_score(question, answer, expected_keywords=None, domain="general", model_answer=None):
    """
    Sync semantic relevance grading.
    Prefer get_semantic_score_async() for production use.
    """
    if not answer or len(answer.strip()) < 5:
        logger.warning("Answer text too short for semantic grading.")
        return _empty_result("Answer too short or empty.")

    try:
        if model_answer and len(model_answer.strip()) > 0:
            target_comparison_text = model_answer
        elif expected_keywords and len(expected_keywords) > 0:
            target_comparison_text = ", ".join(expected_keywords)
        else:
            target_comparison_text = question

        target_emb = get_embedding(target_comparison_text)
        a_emb = get_embedding(answer)
        similarity = cosine_similarity_vectors(target_emb, a_emb)
        similarity = max(0.0, similarity)
        similarity_score = round(similarity * 100, 2)

    except Exception as e:
        logger.error(f"Embedding scoring computation error: {e}")
        similarity_score = 50.0

    keyword_result = check_keywords(answer, expected_keywords)
    length_result = check_answer_length(answer)
    # domain_vocab_score = check_domain_vocab(answer, domain)

    sub_w = config.SEMANTIC_SUB_WEIGHTS
    combined = round(
        (similarity_score * sub_w["similarity"])
        + (keyword_result["score"] * sub_w["keyword"])
        # + (domain_vocab_score * sub_w["vocab"])
        + (length_result["score"] * sub_w["length"])
    )

    feedback = generate_semantic_feedback(similarity_score, keyword_result, length_result)
    logger.info(f"Semantic grading completed | combined: {combined} | keyword count: {len(keyword_result['found'])}")

    return {
        "similarity_score": similarity_score,
        "keyword_score": keyword_result["score"],
        "keyword_coverage": keyword_result["coverage"],
        "keywords_found": keyword_result["found"],
        "keywords_missing": keyword_result["missing"],
        "length_score": length_result["score"],
        "word_count": length_result["word_count"],
        # "domain_vocab_score": domain_vocab_score,
        "combined_score": combined,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", 
    "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
    "don", "should", "now", "vs", "versus"
}

def clean_words(phrase):
    cleaned = (
        phrase.lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace(";", " ")
        .replace(":", " ")
        .replace(".", " ")
    )
    words = []
    for w in cleaned.split():
        w = w.strip(".;:!?\"'")
        if w and w not in STOPWORDS:
            words.append(w)
    return words

def check_keywords(answer, expected_keywords):
    if not expected_keywords:
        return {"score": 70, "coverage": 1.0, "found": [], "missing": []}

    answer_lower = answer.lower()
    found = []
    missing = []
    for kw in expected_keywords:
        kw_clean = kw.strip()
        if not kw_clean:
            continue
        words_count = len(kw_clean.split())
        # If it's a short keyword/phrase (like old format), do exact substring match
        if words_count <= 2:
            if kw_clean.lower() in answer_lower:
                found.append(kw_clean)
            else:
                missing.append(kw_clean)
        else:
            # For longer rubric items, check if at least one key word is present in candidate answer
            kw_words = clean_words(kw_clean)
            if not kw_words:
                if kw_clean.lower() in answer_lower:
                    found.append(kw_clean)
                else:
                    missing.append(kw_clean)
                continue
            match_found = False
            for w in kw_words:
                if w in answer_lower:
                    match_found = True
                    break
            if match_found:
                found.append(kw_clean)
            else:
                missing.append(kw_clean)

    total = len(found) + len(missing)
    coverage = round(len(found) / total, 2) if total > 0 else 1.0
    return {"score": round(coverage * 100), "coverage": coverage, "found": found, "missing": missing}


# def check_domain_vocab(answer, domain=None):
#     if not domain:
#         return 0
#     vocab_list = config.DOMAIN_VOCAB.get(domain, [])
#     if not vocab_list:
#         return 0
#     answer_lower = answer.lower()
#     found = [w for w in vocab_list if w.lower() in answer_lower]
#     return round(len(found) / len(vocab_list) * 100)


def check_answer_length(answer):
    words = len(answer.split())
    if words < 10:
        return {"score": 20, "word_count": words, "feedback": "Answer is too brief. Please explain with structural detail."}
    elif words < 30:
        return {"score": 55, "word_count": words, "feedback": "Response is somewhat short. Expand on terms and mechanics."}
    elif words <= 150:
        return {"score": 95, "word_count": words, "feedback": "Appropriate response length."}
    elif words <= 250:
        return {"score": 80, "word_count": words, "feedback": "Detailed answer, but try to be slightly more concise."}
    else:
        return {"score": 50, "word_count": words, "feedback": "Response is excessively wordy. Focus on core requirements."}


def generate_semantic_feedback(similarity, keyword_result, length_result):
    feedback = []
    if similarity >= 70:
        feedback.append("Highly relevant explanation that aligns with the topic.")
    elif similarity >= 45:
        feedback.append("Moderately relevant response. Try aligning vocabulary directly to the question.")
    else:
        feedback.append("Response appears off-topic. Maintain focus on the technical details asked.")

    if keyword_result["missing"]:
        feedback.append(f"Missing key terms: {', '.join(keyword_result['missing'][:3])}.")
    else:
        feedback.append("Demonstrated solid usage of key domain keywords.")

    feedback.append(length_result["feedback"])
    return " ".join(feedback)


def _empty_result(reason=""):
    return {
        "similarity_score": 0,
        "keyword_score": 0,
        "keyword_coverage": 0,
        "keywords_found": [],
        "keywords_missing": [],
        "length_score": 0,
        "word_count": 0,
        # "domain_vocab_score": 0,
        "combined_score": 0,
        "feedback": reason,
    }