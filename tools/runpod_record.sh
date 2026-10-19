#!/usr/bin/env bash
# Record one benchmark run as a terminal video: live GPU stats on top, the load test below.
# Run ON the box, after the vLLM server for that run is already healthy on $PORT, so the
# recording shows the benchmark and not the model loading. Writes demo.cast, demo.gif, demo.mp4.
#
# Usage (runner flags, same as run_infertutor_experiment.py minus --url):
#   tools/runpod_record.sh --label m-final --mode mixed --users 300 --max-seqs 256 --no-fast-boot

set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8000}

curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "No healthy vLLM on port $PORT. Start the server first."; exit 1; }
command -v asciinema > /dev/null || apt-get install -y -qq asciinema > /dev/null
command -v ffmpeg > /dev/null || apt-get install -y -qq ffmpeg > /dev/null
if ! command -v agg > /dev/null; then
  curl -sfL -o /usr/local/bin/agg https://github.com/asciinema/agg/releases/latest/download/agg-x86_64-unknown-linux-gnu
  chmod +x /usr/local/bin/agg
fi

gpu="watch -n1 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv"
run="cd starter_code && python run_infertutor_experiment.py --url http://127.0.0.1:$PORT --gpu-type H100 --replicas 1 --duration 90 --ramp-up 20 --max-tokens 96 $*; sleep 10; tmux -L rec kill-server"
# A separate tmux server (-L rec) so this works from inside the user's own tmux session.
env -u TMUX asciinema rec --overwrite -i 2 demo.cast \
  -c "tmux -L rec new-session -s demo '$gpu' \; split-window -v -l 80% \"bash -c '$run'\""

agg --font-size 18 demo.cast demo.gif
ffmpeg -y -loglevel error -i demo.gif -movflags faststart -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" demo.mp4
ls -lh demo.cast demo.gif demo.mp4
