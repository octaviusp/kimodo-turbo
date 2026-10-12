# PERF — kimodo-turbo

Hardware: Apple M5 Pro, 48 GB, macOS 26. Software: MLX 0.32.3, torch 2.14.1, transformers 5.1.0,
kimodo @ 58e7818, model Kimodo-SOMA-RP-v1, 100 DDIM steps, separated CFG [2, 2], seed 0.

## Budgets (set by the user) and results

| Budget | Target | Measured (`bench/latency.py`, fresh processes, 5 reps) | Status |
|---|---|---|---|
| Memory | ≤ 5 GB | peak physical footprint **3.74 GB** (max RSS 3.04 GB); 2.40 GB when the prompt's embedding is cached | ✅ |
| Latency, 5 s animation, end to end | < 30 s | **8.76 s** median, 8.95 s max; 8.2 s cached | ✅ |
| Precision | same as original | see below: equal or closer to an fp32 oracle | ✅ |

The original `kimodo_gen` on the same Mac (CPU, 2 reps) took 71.8 s median (60.0–83.6 s) with a
17.0 GB peak footprint (15.5 GB RSS). That makes kimodo-turbo **8.2× faster with 4.5× less memory**.

These are warm-disk numbers: the 14 GB of encoder shards were in the macOS page cache, which is
reclaimable and not counted in the process footprint. Uncached SSD reads measured 9.6–13 GB/s
with `F_NOCACHE`, so a cold run adds an estimated ~1.5–2 s. This is not measured directly,
because that would need `sudo purge`.

### Where the time goes (5 s motion)

- Denoising: 5.3 s compiled, 100 steps × 2 CFG rows × 202 tokens. That is ~4 TFLOP/s of fp32 GEMM.
- Encoder stream: ~0.35 s warm. Cached prompts skip it.
- Imports and model load: ~1.5 s.
- Post-processing and export: the remainder.

## Precision evidence

| Check | Original | kimodo-turbo |
|---|---|---|
| Denoiser forward vs FP64 oracle (`tests/precision_forward.py`), max rel | torch fp32: 2.1e-4 | MLX fp32: **1.3e-4** (TF32 on: 7.5e-2) |
| Text embedding vs fp32-compute oracle, 8 prompts, mean 1 − cos | torch bf16: 2.85e-5 | MLX bf16: **2.83e-5** |
| Final motion vs oracle (`bench/quality.py`, 8 prompts, post-processing on): MPJPE | 0.52 cm | **0.43 cm** |
| ... worst joint over all frames and prompts | 15.7 cm | **5.6 cm** |
| ... foot-contact agreement | 99.88 % | 99.92 % |
| Turbo vs original: MPJPE / foot contacts | — | 0.48 cm / 99.96 % |

Here the oracle is the same model with the text encoder computed in fp32 and kimodo's torch fp32
denoiser. Over 100 steps the diffusion amplifies bf16 rounding, so two equally valid bf16
encoders give motions a few millimetres apart. kimodo-turbo stays inside that noise band.

`tests/parity_denoiser.py` uses identical text features, so it isolates the MLX denoiser:

| Case | Max joint diff vs torch |
|---|---|
| CLI path | 0.0002 cm |
| Post-processed | 0.12 cm |
| 2D waypoint constraint (3 CFG rows) | 0.11 cm |
| Padded batch | 0.0007 cm |

`tests/parity_encoder.py` confirms identical token IDs, bidirectional attention (causal gives 100 %
error) and cosine ≥ 0.99996 for all 8 prompts.

## Log (hypothesis → change → measured delta → decision)

1. **16 GB encoder resident.** Changed to stream one decoder layer (441 MB) at a time with
   background prefetch, plus a memory-mapped embedding table. Encoder peak went from 16 GB to
   1.24 GB, and it ran 0.35 s warm vs 1.6–2.3 s on torch CPU. **Kept.**
2. **fp32 denoiser on CPU.** Changed to MLX GPU fp32: 100 steps went from 29–37 s to 4.6–6.9 s.
   **Kept.**
3. **MLX default TF32.** Accuracy check: 7.5e-2 relative error vs 1.3e-4 with full FP32.
   **TF32 disabled** (precision contract).
4. **Separated CFG with no active constraint.** The constraint and unconditional rows have
   identical inputs, so 2 rows are evaluated instead of 3. This is exact: the guidance term is
   0 bit for bit, and it saves 1/3 of denoiser FLOPs. **Kept.**
5. **Constant prefix tokens** (text embedding Linear and heading): computed once per generation
   instead of per step. Exact. **Kept.**
6. **`mx.compile` of the full step.** 5.79 → 5.29 s (5 s motion) and 8.61 → 8.35 s (10 s motion),
   bit-identical. **Kept.**
7. **Eager import of transformers/peft by `kimodo.model`.** Made lazy via PEP 562
   (`patches/kimodo-lazy-llm2vec.patch`): import went from 3.2 s to 0.92 s. **Kept.**
8. **Repeated prompts.** Added an exact on-disk embedding cache keyed by shard identity, dtype and
   prompt: −0.6 s and −1.3 GB peak. **Kept.**

## Dead ends / not done (on purpose)

- **Lower-precision denoiser** (bf16/fp16/TF32, quantization): violates the FP32 contract. The
  model is ill-conditioned, as the TF32 result shows.
- **Fewer diffusion steps or smaller encoder:** changes the model's output distribution, so it is
  outside the quality contract.
- **Merging the supervised LoRA into the bf16 weights:** the original runs it unmerged in fp32, and
  merging would add an extra bf16 rounding.

## Next bottleneck

fp32 GEMM in the denoiser accounts for ~60 % of wall time. Further exact gains would have to come
from a custom fused fp32 attention + MLP kernel. The expected gain is small, because the GEMMs
are already near the measured fp32 rate.
