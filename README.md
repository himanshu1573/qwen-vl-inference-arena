# qwen-vl-inference-arena

Capstone submission for the Vizuara Inference Engineering workshop (InferTutor Arena).

My implementation of the Vizuara Inference Engineering capstone:
serve `Qwen/Qwen3-VL-4B-Instruct` with vLLM on Modal H100s, load test it, tune it, report it.

- `PLAN.md` — the engineering plan, hypotheses, experiment matrix, budget.
- `REPORT.md` — the one-page engineering report (method written, numbers filled after the runs).
- `EXPERIMENTS.md` — per-run log with hypotheses written before each run.
- `docs/knob_notes.md` — why each serving knob moves which metric.
- `tools/runpod_serve.sh` — launch vLLM with identical flags on a rented non-Modal GPU.
- `tools/make_submission.py` — copies the best run and reconstructs the exact command into `submission/`.
- `tools/stop_all_apps.sh` — stops every Modal app from this project.
- `starter_code/` — official starter kit. One addition: the runner stamps the serving knobs into each result JSON (`config.serving`) so experiments are self-describing. Scorer and prompts are untouched.
- `tools/run_matrix.sh` — runs the required experiment matrix and stops each Modal app afterwards.
- `tools/results_table.py` — renders all result JSONs as a Markdown table with scores.
- `tools/mock_vllm_server.py` — local OpenAI-compatible mock so the harness can be tested without a GPU.
- `docs/` — the assignment and Modal runbook Markdown sources.
- `submission/` — final JSON, exact final command, report.

## Setup

```bash
uv venv --python 3.12
uv pip install -r starter_code/requirements.txt
source .venv/bin/activate
modal token new
modal secret create huggingface HF_TOKEN=<your_hf_token>
```

## Local dry run (no GPU)

```bash
python tools/mock_vllm_server.py --port 8000 &
python starter_code/load_test_infertutor.py --url http://127.0.0.1:8000 --mode mixed --users 20 --duration 20 --ramp-up 5 --label mock
python tools/results_table.py
```

## Real runs

```bash
cd starter_code
python run_infertutor_experiment.py --label smoke --gpu-type H100 --replicas 1 --mode text --users 5 --duration 30 --ramp-up 5 --max-tokens 64
modal app stop infertutor-smoke
cd ..
tools/run_matrix.sh                 # full matrix
tools/run_matrix.sh m-base-100u     # one experiment
python tools/results_table.py --sort score
```

## Renting a GPU elsewhere (RunPod, Lambda, etc.)

The starter runner can benchmark any OpenAI-compatible endpoint via `--url`.
On the GPU box, install `vllm==0.21.0 qwen-vl-utils==0.0.14`, export `HF_TOKEN`,
then start the server with the same flags the Modal app uses:

```bash
MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=4096 tools/runpod_serve.sh
```

From your laptop, point the runner at it and pass the same knobs so they are stamped into the result:

```bash
cd starter_code
python run_infertutor_experiment.py --url http://<box-ip>:8000 --label m-base-100u \
  --replicas 1 --max-seqs 32 --max-batch-tokens 4096 --mode mixed --users 100 --duration 90 --ramp-up 20 --max-tokens 96
```

## Wrapping up

```bash
python tools/make_submission.py            # copies best mixed run + exact command into submission/
tools/stop_all_apps.sh                     # stop every infertutor-* Modal app
pytest -q                                  # tests for the local tooling
```
