FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    curl \
    build-essential \
    cargo \
    libgomp1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt ./

# Install PyTorch with CUDA wheels before requirements.txt so pip finds
# the CUDA build already satisfied and skips the CPU-only PyPI wheel.
# Remove build tools in the same layer to keep the image smaller.
RUN pip install --no-cache-dir \
        torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
    && SO_FILES=$(find /usr/local/lib /usr/lib -name 'libctranslate2*.so*' 2>/dev/null) \
    && [ -n "$SO_FILES" ] || { echo "ERROR: libctranslate2 not found!"; exit 1; } \
    && echo "$SO_FILES" | xargs patchelf --clear-execstack \
    && apt-get purge -y build-essential cargo \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ── Project source ────────────────────────────────────────────────────────────
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────────────
# Created before model download so weights are owned by appuser from the start.
# No post-download chown needed.
RUN useradd -m -u 1001 appuser \
    && mkdir -p /app/models/hf_cache /app/uploads /app/logs/api \
    && chown -R appuser:appuser /app

USER appuser

# ── Pre-bake model weights into the image layer ───────────────────────────────
# Models baked in:
#   - facebook/wav2vec2-lv-60-espeak-cv-ft  (~1.2 GB) — phoneme CTC
#   - Whisper base                           (~145 MB)  — ASR
ENV HF_MODELS_CACHE=/app/models/hf_cache

RUN python scripts/docker/download_wav2vec2.py
RUN python scripts/docker/download_whisper.py
# Download Spacy model
RUN python -m spacy download en_core_web_sm

# Download extra models
COPY scripts/docker/download_extra_models.py scripts/docker/
RUN python scripts/docker/download_extra_models.py


EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
