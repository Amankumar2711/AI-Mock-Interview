#!/usr/bin/env bash
# Start vLLM in PROD mode (full-precision, high-throughput)
set -euo pipefail

MODEL="${VLLM_PROD_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
DTYPE="${VLLM_PROD_DTYPE:-float16}"
GPU_MEM="${VLLM_PROD_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LEN="${VLLM_PROD_MAX_MODEL_LEN:-4096}"
TP="${VLLM_PROD_TENSOR_PARALLEL_SIZE:-1}"
PORT="${VLLM_PORT:-8000}"

echo "▶ Starting vLLM (production) …"
echo "  Model   : $MODEL"
echo "  dtype   : $DTYPE"
echo "  GPU%    : $GPU_MEM"
echo "  TP size : $TP"
echo "  Port    : $PORT"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --tensor-parallel-size "$TP" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-prefix-caching   # critical for shared system prompts
EOF

chmod +x /home/claude/llm_eval/scripts/*.sh
echo "done"
