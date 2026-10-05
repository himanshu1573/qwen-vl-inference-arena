"""
Turn results_infertutor/*.json into a Markdown experiment table for REPORT.md.

Usage:
    python tools/results_table.py [results_dir] [--sort score|time]

The score formula is imported from the official scorer so the two never drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_code"))
from score_infertutor import score  # noqa: E402


def config_summary(cfg: dict) -> str:
    """Compact description of the serving knobs recorded in the result file."""
    cfg = cfg.get("serving", cfg)  # run_infertutor_experiment.py stamps knobs under config.serving
    parts = []
    for key, short in [
        ("replicas", "r"),
        ("max_seqs", "seq"),
        ("max_batch_tokens", "b"),
        ("concurrent_inputs", "ci"),
        ("mm_max_pixels", "px"),
    ]:
        if key in cfg:
            parts.append(f"{short}{cfg[key]}")
    if "fast_boot" in cfg:
        parts.append("eager" if cfg["fast_boot"] else "compiled")
    if cfg.get("prefix_cache") is False:
        parts.append("no-pc")
    if cfg.get("chunked_prefill") is False:
        parts.append("no-cp")
    return " ".join(parts) or "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default=str(ROOT / "starter_code" / "results_infertutor"))
    parser.add_argument("--sort", choices=["score", "time"], default="time")
    args = parser.parse_args()

    path = Path(args.results_dir)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No result JSON found under {path}")

    rows = []
    for file in files:
        data = json.loads(file.read_text())
        cfg, res = data["config"], data["results"]
        rows.append(
            {
                "label": cfg.get("label", file.stem),
                "mode": cfg.get("mode", ""),
                "config": config_summary(cfg),
                "users": cfg.get("users", 0),
                "gpus": cfg.get("total_gpus", 1),
                "err": 100 * res.get("error_rate", 0),
                "ttft": res.get("ttft_p95_ms", 0),
                "itl": res.get("itl_p95_ms", 0),
                "tput": res.get("aggregate_stream_chunks_per_s", 0),
                "score": score(data),
                "file": file.name,
            }
        )
    if args.sort == "score":
        rows.sort(key=lambda r: r["score"], reverse=True)

    print("| # | Label | Mode | Config | Users | GPUs | Err % | TTFT p95 | ITL p95 | Throughput | Score |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, 1):
        print(
            f"| {i} | `{r['label']}` | {r['mode']} | {r['config']} | {r['users']} | {r['gpus']} | "
            f"{r['err']:.1f} | {r['ttft']:.0f} ms | {r['itl']:.1f} ms | {r['tput']:.0f} chunks/s | {r['score']:,.0f} |"
        )
    best = max(rows, key=lambda r: r["score"])
    print(f"\nBest score: {best['score']:,.0f} ({best['label']}, {best['file']})")


if __name__ == "__main__":
    main()
