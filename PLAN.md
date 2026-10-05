# InferTutor Arena — Project Plan

Capstone for the Vizuara Inference Engineering workshop. Upstream spec:
https://github.com/VizuaraAI/infertutor-arena-capstone (copied to `docs/`).

## 1. What we are building

Deploy `Qwen/Qwen3-VL-4B-Instruct` behind vLLM's OpenAI-compatible server on
Modal H100s, load test it with the fixed prompt set, and tune serving
parameters to maximise:

```
Score = goodput_chunks_per_s * users * (1 - error_rate)
        / (p95_TTFT_s * p95_ITL_s * total_GPUs)
```

Primary track: **Multimodal Product (`--mode mixed`)**, budget ≤ 4 H100.
Secondary track: **Text Speed (`--mode text`)** if budget allows.

## 2. Reading the score before touching knobs

What the scorer (`starter_code/score_infertutor.py`) actually does:

- `users` is the *configured* `--users`, not a measured value. So users is a
  free multiplier as long as latency and errors hold.
- Throughput is aggregate streamed chunks/s over the whole run, including
  ramp-up. Long ramp-up on a short run drags throughput down.
- ITL is the *mean* gap per request, then p95 across requests. Long prompts and
  image requests share the same p95, so one slow class of traffic sets ITL.
- GPU count divides linearly. Replicas only pay off if throughput and latency
  improve more than linearly per GPU. Reference numbers confirm this:

| Reference run (mixed) | Score |
|---|---:|
| eager, 2 replicas, 100 users | ≈ 3.3M |
| eager, 4 replicas, 120 users | ≈ 2.4M |

So on the mixed track **fewer GPUs pushed harder** beats more GPUs, until the
error rate or p95 bends. Until saturation, score grows roughly as
`users² / (TTFT × ITL)`, so finding the concurrency knee is the highest-leverage
experiment.

Working hypotheses to test, in order:

1. **Concurrency knee**: sweep users on 1 replica; find where TTFT p95 bends.
2. **Prefix caching**: every request shares the same system prompt, and the
   `long` prompts repeat their own context. Expect lower TTFT with it on.
3. **Chunked prefill**: protects decode (ITL) from `long` and image prefills.
   Expect ITL p95 to be worse with it off on mixed traffic.
4. **Batch knobs**: `max_num_seqs` 32 vs 64 and `max_num_batched_tokens` 4096
   vs 2048/8192. Lower batched tokens should protect ITL; higher seqs should
   raise throughput but hurt ITL. The score multiplies both, so we need the
   product, not either alone.
5. **Image pixel budget**: `--mm-max-pixels` lower than the default
   401,408 (512×28×28). The benchmark image is 256×192, so a smaller budget
   should be free of quality loss and cuts vision-encoder prefill.
6. **Scale out**: 1 vs 2 replicas at the per-replica sweet spot. Only keep 2
   if score per GPU holds.
7. **Compiled mode** (`--no-fast-boot`): text track only. Spec warns it was
   poor on mixed traffic; verify once, then keep eager for mixed.

## 3. Phases

### Phase 0 — Local setup (done, no GPU needed)
- [x] Copy starter kit, `uv venv --python 3.12`, install requirements.
- [x] `tools/mock_vllm_server.py`: local OpenAI-compatible streaming mock so
      the load tester and scorer can be validated end to end without Modal.
- [x] `tools/results_table.py`: turns `results_infertutor/*.json` into the
      Markdown experiment table the report requires.
- [x] `tools/run_matrix.sh`: the minimum experiment matrix as one script with
      `modal app stop` between runs so we never pay for idle GPUs.
- [x] `REPORT.md` template with every required question.

### Phase 1 — Credentials and smoke test (needs your tokens)
```bash
source .venv/bin/activate
modal token new                          # or: modal token set --token-id ... --token-secret ...
modal secret create huggingface HF_TOKEN=<hf_token>
cd starter_code
python run_infertutor_experiment.py --label smoke --gpu-type H100 --replicas 1 \
  --mode text --users 5 --duration 30 --ramp-up 5 --max-tokens 64
modal app stop infertutor-smoke
```
Pass condition: 0 errors, a JSON under `starter_code/results_infertutor/`.

### Phase 2 — Minimum experiment matrix (1 GPU unless stated)
`tools/run_matrix.sh` runs these in order. Duration 90 s, ramp-up 20 s,
max-tokens 96, mode mixed, unless noted.

| # | Label | Change vs baseline | Question answered |
|---|---|---|---|
| 1 | `m-base-50u` | none, 50 users | baseline |
| 2 | `m-base-100u` | 100 users | where does TTFT bend? |
| 3 | `m-base-150u` | 150 users | past the knee? errors? |
| 4 | `m-nopc` | `--no-prefix-cache` at knee users | prefix caching on vs off |
| 5 | `m-nocp` | `--no-chunked-prefill` | chunked prefill on vs off |
| 6 | `m-s64-b8192` | `--max-seqs 64 --max-batch-tokens 8192` | wider batch |
| 7 | `m-s32-b2048` | `--max-batch-tokens 2048` | narrower batch |
| 8 | `m-px200k` | `--mm-max-pixels 200704` | smaller image budget |
| 9 | `m-r2` | `--replicas 2`, 2× knee users | scale out |
| 10 | `m-final` | best combination, clean run | submission |

Text track (optional, only if budget remains):

| # | Label | Config |
|---|---|---|
| T1 | `t-eager-100u` | 1 replica eager, 100 users |
| T2 | `t-compiled-100u` | `--no-fast-boot`, 100 users |
| T3 | `t-compiled-200u` | `--no-fast-boot`, 200 users |

### Phase 3 — Analysis and report
- `python tools/results_table.py starter_code/results_infertutor` → table.
- Fill `REPORT.md`: score, best TTFT/ITL/throughput, GPU count, what helped,
  what failed, next steps.
- `submission/` holds the final JSON, exact command, and the report.

## 4. Budget estimate

Modal H100 is roughly 4 USD per GPU-hour. Each 1-GPU run is about 90 s of
load plus 2–4 min of container start (the first run also downloads ~8 GB of
weights into the cache volume). Estimate:

| Item | GPU-minutes | USD |
|---|---:|---:|
| Smoke + 8 single-GPU mixed runs | ≈ 60 | ≈ 4 |
| 2-replica run + final | ≈ 20 | ≈ 1.5 |
| Text track (3 runs, compile warmup) | ≈ 30 | ≈ 2 |
| Slack for retries | ≈ 30 | ≈ 2 |
| **Total** | | **≈ 10 USD** |

Always run `modal app stop <name>` after each experiment. The app has a
10-minute scale-down window and `min_containers=replicas`, so a forgotten app
keeps billing until stopped.

## 5. Things to watch (known sharp edges)

- **Client location matters.** The load tester runs on this laptop, so
  network RTT is inside TTFT. Run all experiments from the same network so
  comparisons stay fair. Do not compare a coffee-shop run against a home run.
- **Cold start inside the run.** `min_containers=replicas` should pre-warm,
  but a 2-replica run whose second container is still loading during ramp-up
  will show a TTFT p95 spike. Check `modal app logs` if run 9 looks odd.
- **`--concurrent-inputs 64` per container** caps in-flight requests at the
  Modal layer. At 150 users on 1 replica, requests above 64 queue in Modal
  before vLLM sees them. That shows up as TTFT, not errors. Raising it is a
  legitimate knob for the knee experiments.
- **vLLM 0.21.0 is pinned** in the image. Do not bump it mid-project; the
  Qwen3-VL processor kwargs are version-sensitive.
- **Compiled mode** needs a longer `startup_timeout`; the app allows 15 min.

## 6. Deliverables checklist

- [ ] Final benchmark JSON in `submission/`
- [ ] Exact final command in `submission/final_command.sh`
- [ ] One-page engineering report `REPORT.md`
- [ ] Experiment table with ≥ 5 experiments (generated by `tools/results_table.py`)
- [ ] Explanation of best configuration
- [ ] One surprising failure or tradeoff

## 7. Status (2026-09-21)

Done without GPU: setup, plan, mock server, experiment matrix script, results
table, submission packager, RunPod launcher, stop-all script, experiment log
with hypotheses, knob notes, report skeleton, 11 passing tests.

Next session (needs a rented GPU): Phase 1 smoke test → Phase 2 matrix →
fill `EXPERIMENTS.md` and `REPORT.md` → `tools/make_submission.py`.
