# Experiment Log

One row per run, in the order they ran. Hypothesis and "what to look at" were written
**before** the run; observation and decision were filled **after**. Numbers come from
`python tools/results_table.py`. Failed hypotheses stay in the log; they are part of the report.

All runs: 2026-10-19, 1× H100 SXM 80GB on RunPod (On-Demand), vLLM 0.21.0, mixed mode,
90 s, ramp 20 s, 96 max tokens, bf16, prefix cache on, chunked prefill on, b4096, px401408,
unless the "Change" column says otherwise. vLLM and the load tester ran **on the same box**
against `127.0.0.1` (`tools/runpod_matrix.sh`), so TTFT has no network RTT in it.
`ci64` in the generated table is a Modal-only knob; it had no effect on RunPod.

| # | Label | Change | Hypothesis (before) | Observation (after) | Keep? |
|---|---|---|---|---|---|
| 1 | `m-base-50u` | seq32, 50 users (also the smoke test) | Under-saturated. TTFT p95 < 1 s, ITL p95 ≈ 20–30 ms | TTFT 565 ms, ITL 20.8 ms, 0 errors. Score 6.6M | baseline |
| 2 | `m-base-100u` | 100 users | Throughput ≈ 2× run 1, TTFT < 1.5 s | **Wrong.** Throughput +14% only (1,773 chunks/s), TTFT 2.9 s, ITL flat at 19.5 ms. Requests were queueing for admission, not computing | knee ≤ 50u |
| 3 | `m-base-150u` | 150 users | TTFT > 2 s from queueing behind max_num_seqs=32 | TTFT 5.7 s, throughput flat (1,757). Confirmed: TTFT grows with queue length at a fixed service rate | — |
| 4 | `m-s64-100u` | seq64 | Seats are the cap: TTFT drops, ITL rises a little (decode is memory-bound) | TTFT 507 ms, ITL 21.1 ms, throughput 3,139. Score 29M (9.5× run 2) | — |
| 5 | `m-s128-100u` | seq128 | same | TTFT **106 ms**, ITL 25.4 ms. Score 121M (39× run 2 at the same users) | yes |
| 6 | `m-s128-150u` | seq128, 150u | Still below the knee; score ≈ users² | TTFT 123 ms, ITL 24.5 ms. Score 232M | — |
| 7 | `m-s256-200u` | seq256, 200u | Keep growing until ITL or TTFT bends | TTFT 125 ms, ITL 24.7 ms. Score 400M | — |
| 8 | `m-s256-300u` | seq256, 300u | same | TTFT 175 ms, ITL 24.9 ms, 8,727 chunks/s. Score 600M | **eager best** |
| 9 | `m-s512-450u` | seq512, 450u | Between 450 and 600 either ITL climbs (GPU busy), TTFT jumps, or the client saturates | Throughput plateaus at ~10.2k chunks/s; TTFT 918 ms, ITL 32 ms. Score 155M | knee 300–450u |
| 10 | `m-s512-600u` | seq512, 600u | same | Throughput +1% over run 9, TTFT 1.9 s. Score 92M | — |
| 11 | `m-s512-375u` | seq512, 375u | A better point between 300 and 450 | TTFT 330 ms, ITL 28.3 ms. Score 404M, below run 8 | no |
| 12 | `m-compiled` | CUDA graphs, seq256/300u | ITL stuck at ~25 ms regardless of batch looks like fixed per-step overhead; graphs should cut it | ITL **8.5 ms** (3× better), throughput 12,478 (+43%), but TTFT p95 **832 ms**. Score 533M, *lower* than eager | see run 19 |
| 13 | `m-nopc` | prefix cache off | TTFT worse: shared system prompt re-prefilled | 665M vs 600M baseline. Within noise (see runs 17–18) | no effect |
| 14 | `m-nocp` | chunked prefill off | ITL worse: long/image prefills stall decode | ITL 25.7 ms vs 20.8–24.9 ms baseline. Within noise | no effect |
| 15 | `m-s32-b2048` | b2048 (seq256 despite the label) | ITL down, TTFT up | 620M. Within noise | no effect |
| 16 | `m-b8192` | b8192 | Throughput up, ITL up | 644M. Within noise | no effect |
| 17 | `m-compiled-b8192` | compiled + b8192 | Compiled raised req/s 68%, so prefills queue; a bigger prefill budget brings TTFT down | **Wrong.** TTFT 962 ms, worse. Prefill budget was not the cause | no |
| 18 | `m-nopc-r2` | repeat of run 13 | Noise check | 767M (run 13: 665M) | — |
| 19 | `m-s256-300u-r2` | repeat of run 8 | Noise check | **802M** (run 8: 600M, +34%). Single runs are not comparable below ~30% | — |
| 20 | `m-compiled-api2` | compiled + `--api-server-count 2` | `top` showed the single API server pinned at 100% CPU under compiled mode; two processes should drop TTFT toward ~200 ms with ITL ≈ 8 ms | TTFT **127 ms**, ITL 8.7 ms, 16,182 chunks/s. Score **4.41B** | **yes** |
| 21 | `m-final` | same as run 20, recorded | Reproduces run 20 | TTFT 123 ms, ITL 8.5 ms, 16,375 chunks/s, 0 errors / 16,211 requests. Score **4.68B** (+6% vs run 20) | **submission** |

Not run: `m-px200k` (the benchmark image is 256×192 = 49,152 px, below both pixel budgets, so
no change is expected), `m-r2` (needs a second GPU and a load balancer), and the text track (budget).

## Where the bottleneck was, run by run

`top` was captured during the load phase of runs 13, 17 and 20 (one Python process can use at
most one core, so 100% means pinned; the box had 8 vCPUs and ~25% total CPU use).

| Stage | Wall | Evidence | Fix |
|---|---|---|---|
| seq32 | vLLM admission queue | Throughput flat 1.77k from 100→150u while TTFT doubled and ITL stayed flat | `max_num_seqs` 256 |
| eager, seq256 | `VLLM::EngineCore` at 100% CPU | GPU utilization ~40% on the RunPod dashboard; ITL ~25 ms at any batch size | CUDA graphs (`--no-enforce-eager`) |
| compiled | API server at 100% CPU (EngineCore 108%, client 87%) | TTFT 832 ms while ITL dropped to 8.5 ms | `--api-server-count 2` |
| compiled + api2 | **load tester** at 99% CPU (API servers 82% / 72%) | — | Measurement limit; see Surprises |

## Noise control

- Runs 8 and 19 are the same config 30 minutes apart and differ by 34% in score (ITL p95 24.9
  vs 20.8 ms). The host's load average was 50–70 from other tenants, and the serving path was
  CPU-bound, so neighbour load moves ITL. The score multiplies two p95s, which amplifies this.
- So runs 13–16 cannot rank prefix caching, chunked prefill or batch tokens; the honest result
  is "no measurable effect at this noise level". The changes that matter are 5–40× and survive it.
- Runs 20 and 21 (same config) agree within 6%.

## Surprises / failures (feed into the report)

1. **The default config was capped by `max_num_seqs=32`, not by the GPU.** 100 users gave +14%
   throughput and 5× TTFT. Raising it to 128 gave 39× the score at the same user count.
2. **Compiled mode alone made the score worse** (600M → 533M) even though ITL improved 3×. The
   faster engine moved the wall to the single API server process, and TTFT went up.
3. **Hypothesis 17 was wrong.** A bigger prefill budget made compiled TTFT worse, not better.
   `top` pointed at the API server instead, and run 20 confirmed it.
4. **The final run is limited by the load generator.** At 16k chunks/s the single-process Python
   client sits at ~99% CPU, so measured TTFT includes client-side delay and the server's real
   capacity is higher than reported.
5. **RunPod template gotcha.** The PyTorch template sets `HF_HUB_ENABLE_HF_TRANSFER=1` for its
   system Python. A fresh venv lacks `hf_transfer`, and vLLM's model download fails with a vague
   "Can't load the configuration of 'Qwen/Qwen3-VL-4B-Instruct'". Fix:
   `export HF_HUB_ENABLE_HF_TRANSFER=0`.
