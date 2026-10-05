# InferTutor Arena — Engineering Report

Track: Multimodal Product (`--mode mixed`) · Model: `Qwen/Qwen3-VL-4B-Instruct` · vLLM 0.21.0 · H100

> Numbers marked _TBD_ are filled after the GPU runs. Method, hypotheses and
> mechanisms are written up front; see `EXPERIMENTS.md` and `docs/knob_notes.md`.

## Final result

| Question | Answer |
|---|---|
| Final score | _TBD_ |
| Best TTFT p95 | _TBD_ |
| Best ITL p95 | _TBD_ |
| Best throughput | _TBD_ chunks/s |
| Total GPU count | _TBD_ |
| Error rate | _TBD_ |

Reference to beat (internal dry run): eager, 2 replicas, 100 users, TTFT p95 1,169 ms, ITL p95 28.7 ms, 2,243 chunks/s → score ≈ 3.3M.

Final command: `submission/final_command.sh`. Final JSON: `submission/final_result.json`.

## Method

1. Read the scorer before tuning. Score = throughput × users × (1−err) / (TTFT_p95 × ITL_p95 × GPUs). Users is a configured multiplier, so the first job is to find the highest user count one GPU sustains before p95 latency bends or errors appear.
2. Establish a 1-GPU baseline, then sweep users (50 / 100 / 150).
3. Toggle one knob at a time at the knee: prefix caching, chunked prefill, batch size, image pixel budget.
4. Scale out to 2 replicas only if the per-GPU score holds.
5. Re-run the best configuration cleanly for submission and stop all apps.

All runs: 90 s, 20 s ramp, 96 max tokens, same network, same time of day where possible.

## Experiment table

<!-- python tools/results_table.py starter_code/results_infertutor -->

_TBD_

## Best configuration and why it won

_TBD — structure: bottleneck suspected → metric that showed it → knob → result → why kept._

## Which optimization helped the most

_TBD_

## Which optimization failed or surprised us

_TBD_ (candidates tracked in `EXPERIMENTS.md` → Surprises)

## What we would try next with more time

- Split routing: send `long` and `image` requests to a replica tuned for prefill (larger batched tokens) and `text` to one tuned for decode (compiled mode, small batch), so one class's p95 stops setting the other's.
- Run the load generator from a Modal CPU container in the same region to remove laptop→cloud RTT from TTFT and measure the server alone.
- Speculative decoding or a small draft model for the fixed 96-token decode phase.
- FP8 KV cache or weight quantization, validated against the quality pass rate before use.
