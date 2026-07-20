from huggingface_hub import snapshot_download


# 1. Download Tone model
snapshot_download("speechbrain/emotion-recognition-wav2vec2-IEMOCAP")

# 2. Download Technical Eval model
snapshot_download("sentence-transformers/all-MiniLM-L6-v2")

# 3. Download Kokoro weights (if relying on HF Hub)
snapshot_download("hexgrad/Kokoro-82M")