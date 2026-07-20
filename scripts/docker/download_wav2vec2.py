import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from transformers import Wav2Vec2ForCTC

cache = os.environ["HF_MODELS_CACHE"]
model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
print(f"Downloading {model_name} to {cache} ...")

for attempt in range(5):
    try:
        snapshot_download(repo_id=model_name, cache_dir=cache)
        break
    except HfHubHTTPError as e:
        if attempt < 4:
            wait = 30 * (attempt + 1)
            print(f"Rate limited (attempt {attempt + 1}/5), retrying in {wait}s ...")
            time.sleep(wait)
        else:
            raise

Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache)
print("Wav2Vec2 download complete.")
