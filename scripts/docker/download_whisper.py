import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

cache = os.environ.get("HF_MODELS_CACHE", "/app/models/hf_cache")
print("Downloading faster-whisper-base model ...")

for attempt in range(5):
    try:
        snapshot_download(repo_id="Systran/faster-whisper-base", cache_dir=cache)
        break
    except HfHubHTTPError as e:
        if attempt < 4:
            wait = 30 * (attempt + 1)
            print(f"Rate limited (attempt {attempt + 1}/5), retrying in {wait}s ...")
            time.sleep(wait)
        else:
            raise

print("Whisper base download complete.")
