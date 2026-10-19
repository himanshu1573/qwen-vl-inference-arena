#!/usr/bin/env bash
# Launch vLLM on a rented bare GPU box (RunPod, Lambda, Vast, etc.) with the
# SAME flags the Modal app uses, so results are comparable across platforms.
#
# Run this ON the GPU machine. Then, from your laptop (or the same box):
#   cd starter_code
#   python run_infertutor_experiment.py --url http://<box-ip>:8000 --label m-base-100u \
#       --replicas 1 --max-seqs 32 --max-batch-tokens 4096 --mode mixed --users 100 \
#       --duration 90 --ramp-up 20 --max-tokens 96
# `--url` skips the Modal deploy but still stamps the flags you pass into the
# result JSON, so pass the same knobs here (env vars) and there (CLI flags).
#
# Override any knob with an env var, e.g.:
#   MAX_NUM_SEQS=64 MAX_NUM_BATCHED_TOKENS=8192 tools/runpod_serve.sh
#   FAST_BOOT=false tools/runpod_serve.sh          # compiled mode
#   ENABLE_PREFIX_CACHING=false tools/runpod_serve.sh
#   API_SERVER_COUNT=2 tools/runpod_serve.sh       # spread HTTP/tokenize/image preprocessing over processes
#
# Install once on the box (matches the Modal image pins):
#   pip install "vllm==0.21.0" "qwen-vl-utils==0.0.14"
#   export HF_TOKEN=<token>

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
DTYPE=${DTYPE:-bfloat16}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-true}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-true}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
FAST_BOOT=${FAST_BOOT:-true}
MM_MAX_PIXELS=${MM_MAX_PIXELS:-401408}   # 512*28*28
PORT=${PORT:-8000}
API_SERVER_COUNT=${API_SERVER_COUNT:-1}

cmd=(vllm serve "$MODEL_NAME"
  --served-model-name "$MODEL_NAME"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization 0.90
  --uvicorn-log-level=warning
  --limit-mm-per-prompt '{"image": 1, "video": 0}'
  --mm-processor-kwargs "{\"min_pixels\": 784, \"max_pixels\": $MM_MAX_PIXELS, \"fps\": 1}"
)

if [ "$FAST_BOOT" = "true" ]; then cmd+=(--enforce-eager); else cmd+=(--no-enforce-eager); fi
[ "$ENABLE_PREFIX_CACHING" = "true" ] && cmd+=(--enable-prefix-caching)
[ "$ENABLE_CHUNKED_PREFILL" = "true" ] && cmd+=(--enable-chunked-prefill)
[ "$API_SERVER_COUNT" != "1" ] && cmd+=(--api-server-count "$API_SERVER_COUNT")

echo "Starting vLLM: ${cmd[*]}"
exec "${cmd[@]}"
