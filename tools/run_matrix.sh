#!/usr/bin/env bash
# Minimum experiment matrix for the Multimodal Product track.
#
# Each run deploys its own Modal app (infertutor-<label>), load tests it, then
# stops the app so no GPU keeps billing. Results land in
# starter_code/results_infertutor/<label>_<mode>_<users>u_<ts>.json
#
# Usage:
#   tools/run_matrix.sh            # run everything
#   tools/run_matrix.sh m-base-50u m-nopc   # run only the named experiments
#
# Prerequisites: modal token + `huggingface` secret (see PLAN.md Phase 1).

set -euo pipefail
cd "$(dirname "$0")/../starter_code"

DURATION=${DURATION:-90}
RAMP=${RAMP:-20}
MAX_TOKENS=${MAX_TOKENS:-96}
KNEE_USERS=${KNEE_USERS:-100}   # update after the user sweep (runs 1-3)

COMMON=(--gpu-type H100 --duration "$DURATION" --ramp-up "$RAMP" --max-tokens "$MAX_TOKENS")

run() {
  local label=$1; shift
  echo
  echo "=============================== $label ==============================="
  echo "python run_infertutor_experiment.py --label $label ${COMMON[*]} $*"
  python run_infertutor_experiment.py --label "$label" "${COMMON[@]}" "$@" || echo "!! $label failed"
  modal app stop "infertutor-${label//_/-}" || true
}

# Experiment definitions (a case statement keeps this compatible with macOS bash 3.2).
experiment_args() {
  case "$1" in
    # 1-3: user sweep on the default 1-replica config to find the TTFT knee
    m-base-50u)   echo "--mode mixed --replicas 1 --users 50" ;;
    m-base-100u)  echo "--mode mixed --replicas 1 --users 100" ;;
    m-base-150u)  echo "--mode mixed --replicas 1 --users 150 --concurrent-inputs 160" ;;
    # 4-5: feature toggles at the knee
    m-nopc)       echo "--mode mixed --replicas 1 --users $KNEE_USERS --no-prefix-cache" ;;
    m-nocp)       echo "--mode mixed --replicas 1 --users $KNEE_USERS --no-chunked-prefill" ;;
    # 6-7: batch knobs
    m-s64-b8192)  echo "--mode mixed --replicas 1 --users $KNEE_USERS --max-seqs 64 --max-batch-tokens 8192" ;;
    m-s32-b2048)  echo "--mode mixed --replicas 1 --users $KNEE_USERS --max-batch-tokens 2048" ;;
    # 8: image pixel budget (benchmark image is 256x192, so 200704 px is still lossless)
    m-px200k)     echo "--mode mixed --replicas 1 --users $KNEE_USERS --mm-max-pixels 200704" ;;
    # 9: scale out
    m-r2)         echo "--mode mixed --replicas 2 --users $((KNEE_USERS * 2))" ;;
    # text track (optional)
    t-eager-100u)     echo "--mode text --replicas 1 --users 100" ;;
    t-compiled-100u)  echo "--mode text --replicas 1 --users 100 --no-fast-boot" ;;
    t-compiled-200u)  echo "--mode text --replicas 1 --users 200 --no-fast-boot --concurrent-inputs 256" ;;
    *) return 1 ;;
  esac
}
ORDER=(m-base-50u m-base-100u m-base-150u m-nopc m-nocp m-s64-b8192 m-s32-b2048 m-px200k m-r2 t-eager-100u t-compiled-100u t-compiled-200u)

SELECTED=("$@")
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=("${ORDER[@]}")

for label in "${SELECTED[@]}"; do
  args=$(experiment_args "$label") || { echo "Unknown experiment: $label"; exit 1; }
  # shellcheck disable=SC2086
  run "$label" $args
done

echo
echo "Done. Remaining Modal apps (should be none from this matrix):"
modal app list || true
