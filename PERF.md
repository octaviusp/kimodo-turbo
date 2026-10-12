# Measured performance

Apple M5 Pro, 48 GB RAM, macOS 26. MLX 0.32.3, PyTorch 2.14.1, Transformers 5.1.0.
Original Kimodo: `58e7818`; model: `Kimodo-SOMA-RP-v1`.

## Original vs Turbo

Both generate a 5-second motion with 100 DDIM steps, separated CFG `[2, 2]`, seed 0,
and post-processing enabled. The original uses PyTorch CPU; Turbo uses MLX GPU.

| Fresh process | Median wall time | Maximum process footprint | Runs |
|---|---:|---:|---:|
| Original | 71.78 s | 17.01 GB | 2 |
| Turbo, embedding cache disabled | 8.76 s | 3.74 GB | 5 |
| Turbo, embedding cached | 8.20 s | 2.40 GB | 5 |

Time includes imports, loading, generation, post-processing and export. Memory is macOS
`/usr/bin/time -l` **peak memory footprint**, including GPU allocations; it is not total system
memory or just RSS. GB means 10⁹ bytes. Runs used a warm OS disk cache; cold SSD performance
was not measured. The original's two runs took 59.98 and 83.58 seconds.

[Raw timing and memory data](docs/latency.json) preserve the recorded summaries and runs.
There is one later extra run for each Turbo variant in the raw list; the published summary
uses the original five runs. The charts use that summary, not the extra runs.

The historical comparison was interleaved in fresh processes using:

```sh
/usr/bin/time -l .venv/bin/kimodo_gen \
  "A person jumps over an obstacle and lands softly." \
  --model Kimodo-SOMA-RP-v1 --duration 5 --seed 0 --output out/motions/original
KIMODO_TURBO_NO_CACHE=1 /usr/bin/time -l .venv/bin/kimodo-turbo gen \
  "A person jumps over an obstacle and lands softly." \
  --duration 5 --seed 0 --output out/motions/turbo
```

Reproducing the original requires its full encoder weights, removed from this inference-only
checkout after conversion. The old research harness was removed. These measurements describe
the recorded pipeline; they are not a new benchmark of the cleanup or a CUDA GPU comparison.

## Precision

The encoder retains BF16 projections and unmerged FP32 LoRA. The denoiser retains FP32;
TF32 is disabled. Model weights and diffusion-step count are unchanged.

[Eight held-out prompts](docs/quality.json), compared with the same model using an FP32-compute
text encoder as an oracle:

| Metric | Original | Turbo |
|---|---:|---:|
| Mean joint-position error vs oracle | 0.52 cm | 0.43 cm |
| Worst joint-position error vs oracle | 15.73 cm | 5.65 cm |
| Foot-contact agreement vs oracle | 99.88% | 99.92% |

Turbo vs original: 0.48 cm mean joint-position difference and 99.96% foot-contact agreement.
These are numerical agreement checks, not a claim of better motion quality. Small changes in
BF16 rounding are amplified by diffusion.

## Why it improves

- Stream one encoder layer at a time instead of keeping the full 8B encoder resident.
- Run the two-stage FP32 denoiser on MLX GPU with a compiled step.
- Compute constant text/heading tokens once; omit the duplicate CFG row without constraints.
- Cache exact pooled text embeddings; lazily import the original encoder dependencies.

Constraint handling, multi-prompt stitching, forward kinematics, C++ post-processing and
motion export still use the pinned upstream implementation.
