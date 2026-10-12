# kimodo-turbo

[NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) text-to-motion generation on Apple Silicon,
with the same model precision as the original pipeline: a 5 s animation in ~9 s end to end with a
3.7 GB peak memory footprint, versus ~72 s and 17 GB for the original `kimodo_gen` on this Mac
(M5 Pro). Evidence and method are in `PERF.md`.

```zsh
kimodo-turbo gen "A person jumps over an obstacle and lands softly." --duration 5 --seed 0 --output out/motions/jump
```

`gen` accepts every `kimodo_gen` option (`--model`, `--duration`, `--num_samples`,
`--diffusion_steps`, `--constraints`, `--bvh`, `--cfg_type/--cfg_weight`, `--no-postprocess`,
`--seed`, ...). It writes the same NPZ/BVH/CSV outputs, because it runs kimodo's own generate
script with only the model loader replaced. The default model is `Kimodo-SOMA-RP-v1`.

## Gym (web UI)

```zsh
kimodo-turbo gym            # http://127.0.0.1:8765
```

Pick a character (the bundled *Personaje realista*, or any `.glb` via the file chooser /
drag and drop), describe the motion (one row per segment; prompts work best in English), and
the animation plays on the character in real time. Controls: scrub, speed, loop, in-place,
camera follow, and a Kimodo skeleton overlay. Every motion is also saved to `out/gym/motions/`.

- **Characters without a skeleton** are auto-rigged on the server (`kimodo_turbo/autorig.py`). It
  detects facing from the nose, finds crotch, armpits, arms, neck and feet in horizontal mesh
  cross-sections, and builds Kimodo-named joints. Skin weights come from body-part labels plus
  bone distance, smoothed over the mesh. It handles separated legs, long clothing, hanging
  arms and A/T poses. Results are cached in `out/gym/rigged/`. `tests/autorig_debug.py model.glb`
  renders the joints and weights.
- **Rigged characters** (Mixamo, Unreal, Character Creator, Rigify, generic names) are mapped
  automatically, and the mapping can be edited in the UI. Retargeting (`gym/retarget.js`)
  calibrates against each rig's own rest pose, so T-pose, A-pose and arms-down rigs all work
  with twist preserved. The UI shows the mean bone-direction error, which was 0.1° on the
  tested characters.
- **QA:** `tests/gym_qa.py` drives the UI in headless Chromium and saves screenshots to
  `out/gym/qa/`.

## How it works

| Stage | Original (`kimodo_gen`, Mac CPU) | kimodo-turbo |
|---|---|---|
| Text encoder: LLM2Vec Llama-3-8B, bf16 | 16 GB resident, torch CPU | MLX GPU, **streamed one decoder layer at a time** (441 MB) from `out/weights`, embedding rows gathered from a memory map; exact on-disk cache of pooled embeddings |
| Denoiser: two-stage transformer, 282M params, fp32 | torch CPU | MLX GPU fp32 (TF32 disabled), compiled step, constant prefix tokens computed once |
| CFG `separated` | 3 rows per step | 2 rows when no constraint is active: the constraint row equals the unconditional row, so its guidance term is exactly zero |
| Constraints, multi-prompt stitching, FK, foot-skate post-processing, export | kimodo torch + C++ | unchanged kimodo code (`TurboKimodo` overrides only `_generate`) |

Initial noise comes from the same `torch.randn` call, so a given `--seed` yields the same noise as
`kimodo_gen`.

## Precision evidence

- **Denoiser forward vs FP64 oracle:** MLX fp32 max relative error 1.3e-4, torch fp32 CPU 2.1e-4.
  With MLX's default TF32 the error is 7.5e-2, so TF32 is disabled in `kimodo_turbo/__init__.py`.
- **Text embeddings vs an fp32-compute oracle (same weights), 8 prompts:** torch bf16 has
  1 − cos = 2.85e-5, MLX bf16 has 2.83e-5. Both implementations sit at the same bf16 rounding
  distance from exact. Token IDs are identical, and bidirectional attention was confirmed
  layer by layer.
- **Final motion** (`bench/quality.py`, 8 held-out prompts, 5 s, post-processing on): MPJPE vs
  oracle 0.43 cm for turbo vs 0.52 cm for the original; foot contacts agree 99.96 %.

## Setup

```zsh
scripts/setup.zsh     # pinned+patched kimodo, .venv, weights (~17 GB) in out/hf, encoder shards
```

The one-time `kimodo-turbo convert`, which `setup.zsh` runs, loads kimodo's own bf16 encoder
(~17 GB RAM for about a minute). It exports the exact weights the original computes with:
MNTP-merged base projections plus the unmerged supervised LoRA in fp32.

- **Llama-3 weights:** Meta's repo is gated. Setup downloads `NousResearch/Meta-Llama-3-8B-Instruct`
  and `scripts/verify_llama.py` checks that its four safetensors match Meta's published SHA-256
  values before aliasing it. The Meta Llama 3 Community License applies.
- **arm64 patches** (`patches/`): kimodo's MotionCorrection C++ extension only builds on x86. The
  patches add `sse2neon` on arm64, the missing Clang `FORCE_INLINE` and `<stdlib.h>`, and an
  `_mm_permutevar_ps` shim. A separate patch makes `kimodo.model.LLM2VecEncoder` import lazily,
  which saves about 2 s of transformers/peft import time.

## Layout

```
kimodo_turbo/   encoder.py (streamed LLM2Vec) · denoiser.py (MLX two-stage) · pipeline.py (TurboKimodo)
                loading.py · convert.py · embedding_cache.py · cli.py
                gym_server.py · autorig.py · glb.py
gym/            index.html · app.js · retarget.js · mapping.js · style.css · vendor/ (three.js r170)
                assets/personaje-realista.glb
scripts/        setup.zsh · verify_llama.py · reference_encoder.py (oracle capture)
tests/          parity_denoiser.py · parity_encoder.py · precision_forward.py
bench/          quality.py · latency.py · denoiser_speed.py · prompts.json
out/            hf/ weights/ cache/ reference/ bench/ motions/ logs/   (generated, gitignored)
```

Environment variables:

- `KIMODO_TURBO_NO_CACHE=1` disables the embedding cache.
- `HF_HOME` defaults to `out/hf`.

Uninstall by removing `~/.local/bin/kimodo-turbo` and this folder.
