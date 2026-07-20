from huggingface_hub import snapshot_download

model_id = "bobboyms/wav2vec2-base-en-phoneme-ctc-41h" 
local_dir = "./wave2vec2_base"

snapshot_download(repo_id=model_id, local_dir=local_dir)
