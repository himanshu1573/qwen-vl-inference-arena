# Experiment Log

One row per run. Hypothesis and "what to look at" are written **before** the
run; observation and decision are filled **after**. Numbers come from
`python tools/results_table.py`. Keep the row even when a run fails; failures
are part of the report.

Conventions: mixed mode, 1× H100, eager, seq32/b4096, prefix cache on,
chunked prefill on, ci64, px401408, 90 s, ramp 20 s, 96 max tokens, unless the
"Change" column says otherwise. "Knee" = the user count where TTFT p95 bends.

| # | Label | Change | Hypothesis (before) | What to look at | Observation (after) | Keep? |
|---|---|---|---|---|---|---|
| 0 | `smoke` | text, 5 users, 30 s | Infra works, 0 errors | error count, that a JSON is written | | n/a |
| 1 | `m-base-50u` | none | Under-saturated. TTFT p95 < 1 s, ITL p95 ≈ 20–30 ms, GPU has headroom | throughput per user, TTFT vs ref 1,169 ms @ 2 GPU | | baseline |
| 2 | `m-base-100u` | 100 users | Throughput ~2× run 1, TTFT rises but stays < 1.5 s. Score ≈ 4× run 1 if latency holds (users² effect) | did TTFT bend? errors? | | |
| 3 | `m-base-150u` | 150 users, ci160 | Past the knee: TTFT p95 > 2 s from queueing in vLLM's waiting queue (max_num_seqs=32 caps running set). Score may still rise if throughput keeps scaling | TTFT p95 vs run 2, error rate, req/s plateau | | |
| 4 | `m-nopc` | `--no-prefix-cache` @ knee | TTFT p95 worse: every request re-prefills the shared system prompt (~60 tokens) and `long` prompts repeat their own context. Small but measurable | TTFT p95 delta, throughput delta | | expect: keep ON |
| 5 | `m-nocp` | `--no-chunked-prefill` @ knee | ITL p95 worse: a `long` or image prefill runs as one step and stalls all decoders in the batch. TTFT may slightly improve for the long request itself | ITL p95 delta (dominant), TTFT p95 | | expect: keep ON |
| 6 | `m-s64-b8192` | seq64 / b8192 | Throughput up (bigger decode batch), ITL p95 up (more work per step, bigger prefill chunks). Score = tput/ITL so likely a wash or loss, matching the dry run note | ITL p95 vs run 2; tput/ITL ratio | | |
| 7 | `m-s32-b2048` | b2048 | ITL p95 down (smaller prefill chunks interrupt decode less), TTFT p95 up for long/image prompts (more steps to finish prefill). Score depends on which p95 moves more | both p95s; the product TTFT×ITL | | |
| 8 | `m-px200k` | mm-max-pixels 200704 | Benchmark image is 256×192 = 49,152 px, below both budgets, so the vision encoder does the same work. Expect no change. Included to prove the knob is not a free win on this prompt set | TTFT p95 unchanged within noise | | expect: no effect |
| 9 | `m-r2` | 2 replicas, 2× knee users | Per-GPU score roughly preserved only if Modal spreads load evenly. Reference data says 2 replicas beat 4. Watch for a cold-start spike on replica 2 | TTFT p95 vs run 2, `modal app logs` for second container start time | | |
| 10 | `m-final` | best combo | Clean re-run of the winning config for submission | reproducibility vs the run it copies | | submission |
| T1 | `t-eager-100u` | text mode | Text is decode-heavy and short-prompt; ITL p95 ≈ 20 ms eager | ITL p95 | | |
| T2 | `t-compiled-100u` | `--no-fast-boot` | CUDA graphs cut per-step launch overhead: ITL p95 ≈ halves (ref: 10.8 ms). Startup 3–5 min longer | ITL p95, startup time | | |
| T3 | `t-compiled-200u` | 200 users, ci256 | Throughput up, TTFT up. Find whether score still grows at 200 | score vs T2 | | |

## Noise control

- Run every experiment from the same network. TTFT includes laptop→Modal RTT.
- Before trusting a surprising result, re-run it once. Two runs that disagree
  by more than ~15% on a p95 mean the run was too short or a container was
  cold. Prefer a longer `DURATION` over more users when in doubt.
- Note the wall-clock time of each run here; Modal region load varies by hour.

## Surprises / failures (feed into the report)

_Fill during the runs. Candidates: an errors-at-scale run, compiled mode on mixed
traffic, a knob that moved the "wrong" p95._
