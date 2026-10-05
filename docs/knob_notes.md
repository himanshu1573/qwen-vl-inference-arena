# Serving Knob Notes

Why each knob moves which metric, in the language of this capstone. Written
before the runs so the report can cite mechanisms, not just numbers.

## The workload, in scheduler terms

| Class | Share of `mixed` | Prompt tokens (approx) | What it stresses |
|---|---:|---:|---|
| text | 55% | ~60 system + ~40 user ≈ 100 | decode; short prefill |
| long | 20% | ~60 + ~150–200 ≈ 250 | prefill; KV pressure is mild at 8k ctx |
| image | 25% | ~60 + ~25 text + image tokens | vision encoder + prefill |

Image tokens: the benchmark PNG is 256×192. Qwen3-VL uses 28×28 patches with 2×2
merge, so ≈ (256/28)·(192/28) ≈ 9·7 = 63 vision tokens before merge, ~16 after.
This is tiny. The cost of an image request is mostly the vision encoder
forward pass and preprocessing on the CPU side of the container, not KV.

Every request ends after 96 output tokens (`--max-tokens 96`), so decode
length is fixed. The load tester holds one in-flight request per user with a
0.2–1.2 s think time, so `users` ≈ concurrent requests × (1 + think/latency).

## Metrics and what dominates them

- **TTFT** = network RTT + Modal ingress queue + vLLM waiting-queue time +
  prefill time (+ vision encoder for images). At high users the waiting queue
  dominates because `max_num_seqs` caps the running set.
- **ITL** (harness definition: mean gap per request, then p95 across requests)
  = decode step time. Step time grows with batch size (`max_num_seqs`) and
  jumps when a prefill chunk is co-scheduled in the step
  (`max_num_batched_tokens` sets the size of that chunk).
- **Throughput** = Σ output chunks / wall time. Grows with concurrency until
  the GPU is saturated, then flat; errors reduce it.

## Knobs

### `max_num_seqs` (running batch cap)
Higher → more sequences decode per step → higher throughput, higher step time
→ higher ITL. Lower → protects ITL, but pushes extra users into the waiting
queue → higher TTFT. Dry run: 32 beat 64 for this streamed tutor workload,
because the score divides by ITL.

### `max_num_batched_tokens` (tokens per scheduler step)
With chunked prefill on, this is the prefill chunk size that can ride along in
a decode step. Big → prefills finish in fewer steps (lower TTFT for long and
image prompts) but each such step is slow (ITL spike for everyone). Small →
smoother ITL, longer prefill completion. 2048 vs 4096 vs 8192 is the
TTFT-vs-ITL dial.

### Prefix caching
All requests share the system prompt (~60 tokens). Cached prefix blocks skip
prefill compute and KV writes for those tokens. Gain is proportional to
shared-prefix length ÷ total prompt length, so biggest for `text` (~60%),
smaller for `long`. `long` prompt 1 also repeats its own "course context"
sentence, which prefix caching does not help (it is not a prefix of another
request) but which the KV cache still has to hold.

### Chunked prefill
Off: a prefill runs as one scheduler step; a 250-token `long` prompt or an
image stalls every active decode for that step. On: the prefill is split into
`max_num_batched_tokens`-sized pieces and mixed with decode tokens each step.
Directly protects ITL p95 in mixed traffic.

### Eager vs compiled (`--fast-boot` / `--no-fast-boot`)
Compiled = torch.compile + CUDA graph capture. Removes per-kernel launch
overhead, which is a large fraction of a decode step for a 4B model at small
batch: ITL roughly halves on text (ref 10.8 ms vs ~20 ms). Costs minutes of
startup and, per the dry run, behaved badly on multimodal traffic in vLLM
0.21 (variable-shape vision inputs defeat graph capture). Use on text only.

### `--concurrent-inputs` (Modal `@modal.concurrent(max_inputs)`)
Requests beyond this per container wait in Modal's queue before vLLM ever
sees them. Default 64. At 100+ users on one replica this queue is part of
TTFT and is invisible to vLLM metrics. Raise it for user sweeps so the
bottleneck you measure is vLLM's, not Modal's.

### Replicas vs tensor parallelism
A 4B bf16 model is ~8 GB; one H100 holds it with ~60 GB left for KV. TP
across GPUs adds all-reduce latency per layer for no memory need, so replicas
are the right scale-out. The score divides by GPU count, so a second replica
must at least double `users × throughput / (TTFT × ITL)` to break even, which
only happens if one replica was saturated.

### `--mm-max-pixels`
Caps image resolution before the vision encoder. Only matters when input
images exceed the cap. The benchmark image (49k px) is under the default
401k cap, so this knob should be inert here. Document that, do not claim a win.

## What "goodput" means for the final evaluator
The hidden evaluator may count real tokens and a quality pass rate. Nothing
in this plan trades quality for speed (no quantization, no shorter outputs,
no lower image resolution below the image's native size), so quality pass rate
should stay at the model's natural level.
