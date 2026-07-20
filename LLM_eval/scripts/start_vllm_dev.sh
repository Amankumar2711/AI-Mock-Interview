#!/usr/bin/env bash
# Start vLLM in DEV mode (quantised, low memory footprint – runs on a laptop GPU)
set -euo pipefail

MODEL="${VLLM_DEV_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
QUANT="${VLLM_DEV_QUANTIZATION:-gptq}"
GPU_MEM="${VLLM_DEV_GPU_MEMORY_UTILIZATION:-0.60}"
MAX_LEN="${VLLM_DEV_MAX_MODEL_LEN:-2048}"
PORT="${VLLM_PORT:-8000}"

echo "▶ Starting vLLM (dev/quantised) …"
echo "  Model : $MODEL"
echo "  Quant : $QUANT"
echo "  GPU%  : $GPU_MEM"
echo "  Port  : $PORT"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization "$QUANT" \
  --dtype float16 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --host 0.0.0.0 \
  --port "$PORT"
