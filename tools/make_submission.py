"""
Package the final run into submission/.

Picks the highest-scoring result JSON for the chosen mode (or a specific file),
copies it to submission/final_result.json, and reconstructs the exact
run_infertutor_experiment.py command from the stamped serving config into
submission/final_command.sh.

Usage:
    python tools/make_submission.py                 # best mixed run
    python tools/make_submission.py --mode text
    python tools/make_submission.py --file starter_code/results_infertutor/m-final_mixed_100u_123.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_code"))
from score_infertutor import score  # noqa: E402

RESULTS = ROOT / "starter_code" / "results_infertutor"
SUBMISSION = ROOT / "submission"


def build_command(data: dict) -> str:
    cfg = data["config"]
    serving = cfg.get("serving")
    if serving is None:
        raise SystemExit(
            "Result has no config.serving block. It was produced by load_test_infertutor.py directly, "
            "not by run_infertutor_experiment.py, so the serving command cannot be reconstructed."
        )
    flags = [
        f"--label {cfg['label']}",
        f"--gpu-type {serving['gpu_type']}",
        f"--tp {serving['tp']}",
        f"--replicas {serving['replicas']}",
        f"--dtype {serving['dtype']}",
        "--prefix-cache" if serving["prefix_cache"] else "--no-prefix-cache",
        "--chunked-prefill" if serving["chunked_prefill"] else "--no-chunked-prefill",
        "--fast-boot" if serving["fast_boot"] else "--no-fast-boot",
        f"--max-model-len {serving['max_model_len']}",
        f"--max-batch-tokens {serving['max_batch_tokens']}",
        f"--max-seqs {serving['max_seqs']}",
        f"--concurrent-inputs {serving['concurrent_inputs']}",
        f"--mm-max-pixels {serving['mm_max_pixels']}",
        f"--mode {cfg['mode']}",
        f"--users {cfg['users']}",
        f"--duration {cfg['duration']}",
        f"--ramp-up {cfg['ramp_up']}",
        f"--max-tokens {cfg['max_tokens']}",
    ]
    body = " \\\n  ".join(["python run_infertutor_experiment.py"] + flags)
    return f"#!/usr/bin/env bash\n# Exact command for the submitted final run (run from starter_code/).\nset -euo pipefail\ncd \"$(dirname \"$0\")/../starter_code\"\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mixed", choices=["text", "long", "image", "mixed"])
    parser.add_argument("--file", default=None, help="Use this result file instead of the best one")
    args = parser.parse_args()

    if args.file:
        chosen = Path(args.file)
        data = json.loads(chosen.read_text())
    else:
        candidates = []
        for file in sorted(RESULTS.glob("*.json")):
            data = json.loads(file.read_text())
            if data["config"].get("mode") == args.mode and "serving" in data["config"]:
                candidates.append((score(data), file, data))
        if not candidates:
            raise SystemExit(f"No {args.mode} results with a serving block under {RESULTS}")
        _, chosen, data = max(candidates, key=lambda c: c[0])

    SUBMISSION.mkdir(exist_ok=True)
    shutil.copy(chosen, SUBMISSION / "final_result.json")
    (SUBMISSION / "final_command.sh").write_text(build_command(data))
    (SUBMISSION / "final_command.sh").chmod(0o755)

    r = data["results"]
    print(f"Selected: {chosen.name}")
    print(f"Score {score(data):,.0f} | TTFT p95 {r['ttft_p95_ms']:.0f} ms | ITL p95 {r['itl_p95_ms']:.1f} ms | "
          f"{r['aggregate_stream_chunks_per_s']:.0f} chunks/s | err {100 * r['error_rate']:.1f}% | "
          f"{data['config'].get('total_gpus', 1)} GPU(s)")
    print(f"Wrote {SUBMISSION / 'final_result.json'} and {SUBMISSION / 'final_command.sh'}")


if __name__ == "__main__":
    main()
