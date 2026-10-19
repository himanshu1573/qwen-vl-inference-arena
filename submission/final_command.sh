#!/usr/bin/env bash
# Exact commands for the submitted final run (m-final, 2026-10-19, 1x H100 SXM on RunPod).
#
# This run did not go through Modal: vLLM was started on the GPU box, and the runner load
# tested it over localhost with --url. --api-server-count is not a runner flag, so it is not
# stamped in final_result.json; the server line below is copied from logs/vllm_m-final.log.
#
# Usage, on the GPU box from the repo root (vllm==0.21.0, qwen-vl-utils==0.0.14 installed):
#   submission/final_command.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Server. Same as: MAX_NUM_SEQS=256 FAST_BOOT=false API_SERVER_COUNT=2 tools/runpod_serve.sh
mkdir -p logs
vllm serve Qwen/Qwen3-VL-4B-Instruct --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 8192 --max-num-batched-tokens 4096 --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 --uvicorn-log-level=warning \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"min_pixels": 784, "max_pixels": 401408, "fps": 1}' \
  --no-enforce-eager --enable-prefix-caching --enable-chunked-prefill \
  --api-server-count 2 > logs/vllm_m-final.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done

# 2. Load test (the recorded run used tools/runpod_record.sh with these same runner flags).
cd starter_code
python run_infertutor_experiment.py --url http://127.0.0.1:8000 --label m-final \
  --gpu-type H100 --replicas 1 --duration 90 --ramp-up 20 --max-tokens 96 \
  --mode mixed --users 300 --max-seqs 256 --no-fast-boot
