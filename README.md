# Kimodo Turbo

Generate human motion on Apple Silicon with **8.2× faster generation** and **78% less peak
process memory** than the original Kimodo running on CPU on the same Mac.

Same model. Same 100 diffusion steps. BF16 text encoder and FP32 denoiser.

## Original vs Turbo

A **5-second animation**, end to end, on an **Apple M5 Pro with 48 GB RAM**.
Lower is better in both charts. Turbo's embedding cache is disabled for this comparison.

![Generation time: Original 71.78 seconds, Turbo 8.76 seconds](docs/assets/generation-time.png)

![Peak process memory: Original 17.01 GB, Turbo 3.74 GB](docs/assets/peak-memory.png)

| | Original Kimodo | Kimodo Turbo |
|---|---:|---:|
| Median generation time | 71.78 s | **8.76 s** |
| Maximum process memory | 17.01 GB | **3.74 GB** |
| With a cached prompt | — | **8.20 s / 2.40 GB** |

Measured in fresh processes with a warm OS disk cache: 2 Original runs and 5 Turbo runs.
Memory is macOS peak process footprint, including GPU allocations, not total system RAM.
The original runs on PyTorch CPU; Turbo runs on MLX GPU. This is not a CUDA GPU comparison.
See [measurement details and precision checks](PERF.md) and [raw results](docs/latency.json).

## Run

Requirements: Apple Silicon, macOS, Python 3.12 and `uv`, plus CMake and a C++ compiler for
Kimodo's post-processing extension. Keep the checkout in its installed location.

```sh
./scripts/setup.zsh
.venv/bin/kimodo-turbo ui
```

Open **http://127.0.0.1:8765**. Describe the motion, set its duration, and click **Generate motion**.
The included character plays the result. Scrub, pause, change playback speed, or download the NPZ.
The UI uses 100 diffusion steps and foot-skate post-processing by default.

Use the CLI for constraints, multiple prompts and export options:

```sh
.venv/bin/kimodo-turbo gen "A person jumps forward and lands softly." \
  --duration 5 --seed 0 --bvh --output out/motions/jump
.venv/bin/kimodo-turbo gen --help
```

`gen` accepts the original `kimodo_gen` options and writes its NPZ/BVH/CSV formats.
Outputs default to `out/motions/`. Default model: `Kimodo-SOMA-RP-v1`.
`KIMODO_TURBO_NO_CACHE=1` disables the exact text-embedding cache.
Use `ui --port 9000` to select another local port.

## Weights and setup

Setup pins and patches upstream Kimodo, installs dependencies in `.venv/`, downloads the
checkpoint and converts the text encoder once. Conversion needs about **17 GB RAM** temporarily;
regular inference uses the footprint above. Existing converted weights are reused.

Only inference weights remain in `out/weights/`: the motion checkpoint and streamed encoder
shards, about **16.3 GB on disk**. Conversion source weights are deleted after setup. Weights,
caches and generated motions stay local and are excluded from Git.

The Llama-3 mirror is verified against Meta's published SHA-256 values before conversion.
Upstream Kimodo code uses Apache-2.0; Kimodo weights use the NVIDIA Open Model License and
Llama weights use the Meta Llama 3 Community License. Setup retains the checkpoint's license.

## Code

```text
kimodo_turbo/   CLI, streamed encoder, MLX denoiser, upstream adapter, local UI server
ui/            Simple motion preview, bundled character and local Three.js modules
scripts/       Reproducible setup and source-weight verification
patches/       Apple Silicon build support and lazy encoder import
docs/         Recorded benchmark evidence and comparison charts
out/           Local inference weights, caches and generated motions (ignored)
```

Turbo streams encoder layers, compiles the FP32 denoising step, reuses constant tokens and skips
identical CFG rows. Upstream Kimodo still handles constraints, stitching, kinematics,
post-processing and export. Numerical agreement was checked on eight held-out prompts;
see [PERF.md](PERF.md).
