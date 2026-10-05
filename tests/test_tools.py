"""
Tests for the GPU-free tooling: mock server, serving-config stamping,
results table, and submission packaging. Run with:  pytest -q
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(ROOT / "starter_code"))
sys.path.insert(0, str(ROOT / "tools"))

import run_infertutor_experiment as runner  # noqa: E402
import results_table  # noqa: E402
import make_submission  # noqa: E402
from score_infertutor import score  # noqa: E402


def sample_result(label="exp", mode="mixed", users=100, serving=True, ttft=800.0, itl=30.0, tput=2000.0):
    cfg = {
        "url": "http://x", "model": "Qwen/Qwen3-VL-4B-Instruct", "mode": mode, "users": users,
        "duration": 90, "ramp_up": 20, "max_tokens": 96, "request_timeout": 180,
        "min_pause": 0.2, "max_pause": 1.2, "label": label, "total_gpus": 1,
    }
    if serving:
        cfg["serving"] = {
            "gpu_type": "H100", "tp": 1, "replicas": 1, "dtype": "bfloat16", "prefix_cache": True,
            "chunked_prefill": False, "fast_boot": True, "max_model_len": 8192, "max_batch_tokens": 4096,
            "max_seqs": 32, "concurrent_inputs": 64, "mm_max_pixels": 401408,
        }
    return {"config": cfg, "results": {
        "total_requests": 100, "total_errors": 0, "error_rate": 0.0, "ttft_p95_ms": ttft,
        "itl_p95_ms": itl, "aggregate_stream_chunks_per_s": tput,
    }}


# --- score sanity: reference numbers from the assignment PDF ---------------

def test_reference_scores_match_plan():
    r2 = sample_result(users=100, ttft=1168.9, itl=28.7, tput=2243)
    r2["config"]["total_gpus"] = 2
    r4 = sample_result(users=120, ttft=897.6, itl=38.1, tput=2756)
    r4["config"]["total_gpus"] = 4
    assert score(r2) > score(r4), "2-replica reference should outscore 4-replica reference"
    assert 3.2e6 < score(r2) < 3.4e6
    assert 2.3e6 < score(r4) < 2.5e6


# --- record_serving_config --------------------------------------------------

def test_record_serving_config_stamps_latest_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    results = tmp_path / "results_infertutor"
    results.mkdir()
    old = results / "exp_mixed_100u_1.json"
    new = results / "exp_mixed_100u_2.json"
    old.write_text(json.dumps(sample_result(serving=False)))
    time.sleep(0.01)
    new.write_text(json.dumps(sample_result(serving=False)))
    args = argparse.Namespace(label="exp", mode="mixed", users=100, gpu_type="H100", tp=1, replicas=2,
                              dtype="bfloat16", prefix_cache=False, chunked_prefill=True, fast_boot=True,
                              max_model_len=8192, max_batch_tokens=2048, max_seqs=32, concurrent_inputs=64,
                              mm_max_pixels=200704)
    runner.record_serving_config(args)
    stamped = json.loads(new.read_text())["config"]["serving"]
    assert stamped["replicas"] == 2 and stamped["prefix_cache"] is False and stamped["max_batch_tokens"] == 2048
    assert "serving" not in json.loads(old.read_text())["config"]


def test_record_serving_config_no_files_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "results_infertutor").mkdir()
    runner.record_serving_config(argparse.Namespace(label="x", mode="text", users=1))  # must not raise


# --- results_table ----------------------------------------------------------

def test_config_summary_reads_serving_block():
    cfg = sample_result()["config"]
    assert results_table.config_summary(cfg) == "r1 seq32 b4096 ci64 px401408 eager no-cp"


def test_config_summary_without_serving_block():
    assert results_table.config_summary(sample_result(serving=False)["config"]) == "-"


def test_results_table_cli_sorts_by_score(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(sample_result(label="slow", ttft=2000)))
    (tmp_path / "b.json").write_text(json.dumps(sample_result(label="fast", ttft=500)))
    out = subprocess.run([PY, str(ROOT / "tools/results_table.py"), str(tmp_path), "--sort", "score"],
                         capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if l.startswith("| ")]
    assert "`fast`" in lines[1] and "`slow`" in lines[2]
    assert "Best score" in out and "fast" in out.splitlines()[-1]


# --- make_submission --------------------------------------------------------

def test_build_command_roundtrips_flags():
    cmd = make_submission.build_command(sample_result())
    for flag in ["--label exp", "--replicas 1", "--prefix-cache", "--no-chunked-prefill", "--fast-boot",
                 "--max-batch-tokens 4096", "--max-seqs 32", "--mm-max-pixels 401408", "--mode mixed",
                 "--users 100", "--duration 90", "--ramp-up 20", "--max-tokens 96"]:
        assert flag in cmd, flag
    assert "--no-prefix-cache" not in cmd


def test_build_command_requires_serving_block():
    with pytest.raises(SystemExit):
        make_submission.build_command(sample_result(serving=False))


def test_make_submission_picks_best_for_mode(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text(json.dumps(sample_result(label="worse", ttft=2000)))
    (results / "b.json").write_text(json.dumps(sample_result(label="best", ttft=400)))
    (results / "c.json").write_text(json.dumps(sample_result(label="text-run", mode="text", ttft=100)))
    monkeypatch.setattr(make_submission, "RESULTS", results)
    monkeypatch.setattr(make_submission, "SUBMISSION", tmp_path / "submission")
    monkeypatch.setattr(sys, "argv", ["make_submission.py", "--mode", "mixed"])
    make_submission.main()
    final = json.loads((tmp_path / "submission" / "final_result.json").read_text())
    assert final["config"]["label"] == "best"
    assert "--label best" in (tmp_path / "submission" / "final_command.sh").read_text()


# --- mock server + real load tester -----------------------------------------

@pytest.fixture
def mock_server():
    port = 8917
    proc = subprocess.Popen([PY, str(ROOT / "tools/mock_vllm_server.py"), "--port", str(port),
                             "--ttft-ms", "20", "--itl-ms", "2"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("mock server did not start")
    yield url
    proc.kill()


def test_mock_server_streams_openai_chunks(mock_server):
    with httpx.stream("POST", f"{mock_server}/v1/chat/completions",
                      json={"model": "m", "max_tokens": 5, "stream": True,
                            "messages": [{"role": "user", "content": "hi"}]}, timeout=10) as resp:
        assert resp.status_code == 200
        lines = [l for l in resp.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert len(chunks) == 5
    assert all(c["choices"][0]["delta"]["content"] for c in chunks)


def test_load_tester_end_to_end_against_mock(mock_server, tmp_path):
    # Run the official load tester from a temp copy so results land in tmp, not the repo.
    work = tmp_path / "starter_code"
    work.mkdir()
    for name in ["load_test_infertutor.py", "prompts.json"]:
        (work / name).write_bytes((ROOT / "starter_code" / name).read_bytes())
    subprocess.run([PY, "load_test_infertutor.py", "--url", mock_server, "--mode", "mixed", "--users", "5",
                    "--duration", "4", "--ramp-up", "1", "--max-tokens", "8", "--label", "t"],
                   cwd=work, check=True, capture_output=True, timeout=60)
    files = list((work / "results_infertutor").glob("t_mixed_5u_*.json"))
    assert len(files) == 1
    res = json.loads(files[0].read_text())["results"]
    assert res["total_errors"] == 0 and res["total_requests"] > 0
    assert res["ttft_p95_ms"] > 0 and res["aggregate_stream_chunks_per_s"] > 0
