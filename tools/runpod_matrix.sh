#!/usr/bin/env bash
# Experiment matrix on a single rented GPU box (RunPod, Lambda, etc.). Run it ON the box.
#
# For each experiment, start vLLM through tools/runpod_serve.sh with that run's server
# knobs, then load test http://127.0.0.1:$PORT with the matching runner flags, so the
# result JSON is stamped with what the server actually ran. vLLM restarts only when the
# server knobs change, so the user sweep (runs 1-3) shares one server.
# The load tester runs on the same box, so laptop-to-cloud RTT is not part of TTFT.
#
# Usage (inside the repo, venv active, nothing else on $PORT):
#   tools/runpod_matrix.sh                       # run everything
#   tools/runpod_matrix.sh m-base-50u m-nopc     # run only the named experiments
#   tools/runpod_matrix.sh m-nopc-r2             # repeat a run under its own label (-r<N> suffix)
#   tools/runpod_matrix.sh m-compiled-api2       # same run with N API server processes (-api<N>)
#   KNEE_USERS=150 tools/runpod_matrix.sh m-nopc m-nocp
#
# vLLM logs: logs/vllm_<label>.log. Results: starter_code/results_infertutor/
# The 2-replica run (m-r2) is not here: it needs a second GPU and a load balancer.

set -euo pipefail
cd "$(dirname "$0")/.."

DURATION=${DURATION:-90}
RAMP=${RAMP:-20}
MAX_TOKENS=${MAX_TOKENS:-96}
KNEE_USERS=${KNEE_USERS:-100}   # update after the user sweep (runs 1-3)
BASE_SEQS=${BASE_SEQS:-32}      # max_num_seqs for the toggle runs; update after the seqs sweep
GPU_TYPE=${GPU_TYPE:-H100}      # stamped into the result JSON only
PORT=${PORT:-8000}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
URL="http://127.0.0.1:$PORT"

COMMON=(--url "$URL" --gpu-type "$GPU_TYPE" --replicas 1
  --duration "$DURATION" --ramp-up "$RAMP" --max-tokens "$MAX_TOKENS")

# Server knobs (env vars for runpod_serve.sh) and the matching runner flags.
# Keep the two in sync: the runner only records flags, it cannot see the server.
experiment() {
  case "$1" in
    # 1-3: user sweep on the default config to find the TTFT knee
    m-base-50u)   SERVER_ENV="";  ARGS="--mode mixed --users 50" ;;
    m-base-100u)  SERVER_ENV="";  ARGS="--mode mixed --users 100" ;;
    m-base-150u)  SERVER_ENV="";  ARGS="--mode mixed --users 150" ;;
    # max_num_seqs sweep: at 100 users with seq32, TTFT rose 5x while ITL held, so requests queue
    m-s64-100u)   SERVER_ENV="MAX_NUM_SEQS=64";   ARGS="--mode mixed --users 100 --max-seqs 64" ;;
    m-s128-100u)  SERVER_ENV="MAX_NUM_SEQS=128";  ARGS="--mode mixed --users 100 --max-seqs 128" ;;
    m-s128-150u)  SERVER_ENV="MAX_NUM_SEQS=128";  ARGS="--mode mixed --users 150 --max-seqs 128" ;;
    # 4-5: feature toggles at the knee
    m-nopc)       SERVER_ENV="MAX_NUM_SEQS=$BASE_SEQS ENABLE_PREFIX_CACHING=false"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs $BASE_SEQS --no-prefix-cache" ;;
    m-nocp)       SERVER_ENV="MAX_NUM_SEQS=$BASE_SEQS ENABLE_CHUNKED_PREFILL=false"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs $BASE_SEQS --no-chunked-prefill" ;;
    # 6-7: batch knobs
    m-s64-b8192)  SERVER_ENV="MAX_NUM_SEQS=64 MAX_NUM_BATCHED_TOKENS=8192"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs 64 --max-batch-tokens 8192" ;;
    m-s32-b2048)  SERVER_ENV="MAX_NUM_SEQS=$BASE_SEQS MAX_NUM_BATCHED_TOKENS=2048"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs $BASE_SEQS --max-batch-tokens 2048" ;;
    m-b8192)      SERVER_ENV="MAX_NUM_SEQS=$BASE_SEQS MAX_NUM_BATCHED_TOKENS=8192"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs $BASE_SEQS --max-batch-tokens 8192" ;;
    # compiled mode (CUDA graphs) at the knee: ITL held at ~25 ms from 100 to 300 users, so step
    # time looks like fixed per-step overhead rather than batch size
    m-compiled)   SERVER_ENV="MAX_NUM_SEQS=$BASE_SEQS FAST_BOOT=false"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs $BASE_SEQS --no-fast-boot" ;;
    # compiled cut ITL 3x but raised req/s 68%, so prefills queued (TTFT p95 832 ms); widen prefill budget
    m-compiled-b8192)  SERVER_ENV="MAX_NUM_SEQS=$BASE_SEQS FAST_BOOT=false MAX_NUM_BATCHED_TOKENS=8192"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs $BASE_SEQS --no-fast-boot --max-batch-tokens 8192" ;;
    # 8: image pixel budget
    m-px200k)     SERVER_ENV="MAX_NUM_SEQS=$BASE_SEQS MM_MAX_PIXELS=200704"
                  ARGS="--mode mixed --users $KNEE_USERS --max-seqs $BASE_SEQS --mm-max-pixels 200704" ;;
    # text track (optional)
    t-eager-100u)     SERVER_ENV="";  ARGS="--mode text --users 100" ;;
    t-compiled-100u)  SERVER_ENV="FAST_BOOT=false";  ARGS="--mode text --users 100 --no-fast-boot" ;;
    t-compiled-200u)  SERVER_ENV="FAST_BOOT=false";  ARGS="--mode text --users 200 --no-fast-boot" ;;
    # any other seqs/users point, e.g. m-s256-300u
    m-s*-*u)
      local seqs=${1#m-s}; seqs=${seqs%%-*}
      local users=${1##*-}; users=${users%u}
      case "$seqs$users" in ''|*[!0-9]*) return 1 ;; esac
      SERVER_ENV="MAX_NUM_SEQS=$seqs";  ARGS="--mode mixed --users $users --max-seqs $seqs" ;;
    *) return 1 ;;
  esac
}
ORDER=(m-base-50u m-base-100u m-base-150u m-s64-100u m-s128-100u m-s128-150u m-nopc m-nocp m-s64-b8192 m-s32-b2048 m-px200k
  t-eager-100u t-compiled-100u t-compiled-200u)

# Label suffixes: -api<N> adds N API server processes, -r<N> marks a repeat.
resolve() {
  local name=${1%-r[0-9]} api=""
  case "$name" in *-api[0-9]) api=${name##*-api}; name=${name%-api[0-9]} ;; esac
  experiment "$name" || return 1
  if [ -n "$api" ]; then SERVER_ENV="$SERVER_ENV API_SERVER_COUNT=$api"; fi
}

SERVER_PID=""
CURRENT_ENV=""

stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

start_server() {
  local log=$1
  stop_server
  echo "Starting vLLM (${SERVER_ENV:-defaults}), log: $log"
  # shellcheck disable=SC2086  # SERVER_ENV is a list of VAR=value words
  env $SERVER_ENV PORT="$PORT" tools/runpod_serve.sh > "$log" 2>&1 &
  SERVER_PID=$!
  CURRENT_ENV=$SERVER_ENV
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -sf "$URL/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!! vLLM exited during startup. Last lines of $log:"
      tail -n 30 "$log"
      SERVER_PID=""
      return 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "!! vLLM not healthy after ${STARTUP_TIMEOUT}s, see $log"
      stop_server
      return 1
    fi
    sleep 5
  done
  echo "vLLM healthy after ${SECONDS}s since script start"
}

if curl -sf "$URL/health" > /dev/null; then
  echo "Something is already serving on $URL. Stop it first (e.g. tmux kill-session -t vllm)."
  exit 1
fi

SELECTED=("$@")
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=("${ORDER[@]}")
for label in "${SELECTED[@]}"; do
  resolve "$label" || { echo "Unknown experiment: $label"; exit 1; }
done

mkdir -p logs
for label in "${SELECTED[@]}"; do
  resolve "$label"
  echo
  echo "=============================== $label ==============================="
  if [ -z "$SERVER_PID" ] || ! kill -0 "$SERVER_PID" 2>/dev/null || [ "$SERVER_ENV" != "$CURRENT_ENV" ]; then
    start_server "logs/vllm_$label.log" || { echo "!! $label skipped"; continue; }
  fi
  echo "python run_infertutor_experiment.py --label $label ${COMMON[*]} $ARGS"
  # shellcheck disable=SC2086
  (cd starter_code && python run_infertutor_experiment.py --label "$label" "${COMMON[@]}" $ARGS) \
    || echo "!! $label failed"
done

echo
echo "Done. Results:"
if [ -d starter_code/results_infertutor ]; then
  python tools/results_table.py starter_code/results_infertutor --sort score
else
  echo "No result JSONs yet."
fi
