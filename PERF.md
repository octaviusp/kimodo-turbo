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

## Character autorig

Native executable `9e83e79b0ec6…`, 1.51 MB, measured on the same Mac on 2026-10-12.
Ten original GLBs, 6,260–23,822 source vertices, from the local character collection.
Source hashes and [recorded results](docs/autorig-validation.json) identify every case;
the source files stay local and unmodified.

| Preparation | Measured result |
|---|---:|
| Warm fresh-process median, range across ten characters | **54.56–94.48 ms** |
| Internal native median, same range | 48.63–88.81 ms |
| Maximum process RSS | **50.28 MB** |
| Maximum process footprint | 48.81 MB |
| Repeated deterministic outputs | 10/10, five runs each |

Automatic up/facing detection, fresh rigging and GLB export are included. Each case had one
discarded warmup followed by five round-robin fresh-process runs with owned inference,
compilation and browser QA idle. Process timings include `/usr/bin/time` spawning; medians
have 0.34–2.29 ms MAD. MB means 10⁶ bytes. These standalone autorig figures exclude motion
inference and the persistent UI/model process. All cases meet the <250 ms / <128 MB budgets.

### Accuracy checks

All ten pass valid GLB/bind recovery, finite normalized weights, 27 body joints, unchanged
world-space rest triangle corners, normals, UVs, materials and embedded image bytes.
Clothing seams duplicate existing vertices; they do not change rest surface geometry.
Independent NumPy skinning and actual Three.js skinning agree over 1,068 samples to
1.16e-7 character heights. Transformation checks cover translation, 0.001×/100× scale and Z-up.

Six poses cover rest, raised arms, wide stance, deep crouch, torso twist and asymmetric motion.
Fixed screening flags more than 1.5% area with edges stretched >2×, more than 1% triangle
collapse below 10% rest area, or stationary limbs/hands. **10/10 clear this screen**.
The old demo rig's raised-arm p99 edge stretch falls from **6.03× to 1.70×**; surface with
edges >2× falls from **2.49% to 0%** in that pose. Corrected sandal/ankle surface remains rigid
in stance, crouch and asymmetric tests. These thresholds screen severe deformation, not
perfect volume preservation.

The real UI generated a 150-frame, 5-second Kimodo motion in 6.008 s, retargeted it to three
characters and downloaded its NPZ without browser errors. Captures and full private receipts
are in `out/autorig-validation/final-ui-proof/`; full ten-case QA is in `final-approved/`.

### Method and limits

Exact triangle cross-sections fit anatomy. Multi-source geodesics, measured body/arm radii
and a screened surface Laplacian prevent limb-to-torso leakage. A deterministic preconditioned
CG solver replaces slow diffusion iterations: independent full-field agreement with Jacobi
512 is <1.5e-5 on two characters, with posed mesh differences <1.3e-6 character heights.
This solver comparison precedes the final body prior and validates the linear solve.

Continuous lower garments use geometric contour ownership; learned base-color palettes
refine ambiguous footwear contacts. PNG/JPEG decoding is bounded. Missing texture support
or indistinguishable palettes retain geometry and record an explicit fallback. Garment/foot
seams separate their weights, and bounded footwear stays attached to its anatomical foot.

Linear skinning can flatten shoulders and bend armor. The robe follows the pelvis without
cloth or collision simulation; extreme stance/crouch exposes ankle ends because the source
has no hidden leg mesh. One demo triangle mixes painted skin and cloth (0.0173% total surface)
and follows the robe. The six-pose corpus and real clip do not guarantee every future character
or animation. Existing authored skins are preserved unless forced and are not re-certified.

Dependencies: glam 0.34.1, gltf 1.4.1, serde_json 1.0.151, and
[image 0.25.10](https://docs.rs/image/0.25.10/image/) with only PNG/JPEG formats enabled.
Cargo source cache, build artifacts and the executable stay under `out/`.
