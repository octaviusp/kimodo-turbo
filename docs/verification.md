# Cleanup verification

Verified on Apple M5 Pro, macOS 26, 2026-10-12.

- Offline CLI: 5-second motion, 100 steps, NPZ and BVH export. Output: 150 × 77 joints; finite values.
- Denoiser parity with fixed text features and identical noise: four cases passed. Maximum joint
  difference: CLI 0.0002 cm; post-processed 0.1208 cm; waypoint constraints 0.1091 cm; padded batch 0.0007 cm.
- Chromium: character loading, generation, playback/pause, scrubbing, skeleton overlay, NPZ download
  and a 390-pixel mobile viewport passed. No browser errors. Downloaded motion: 150 × 77 joints.
- API: empty prompt, out-of-range duration, invalid JSON shape and negative seed return clear errors.
- Encoder prefetch propagates missing-shard errors; valid and empty streams passed.
  Embeddings remain bit-identical to the committed encoder on two prompts.
- Ruff, formatting and shell syntax checks passed.

Validation used temporary scripts outside the repository. Historical performance evidence is
separate: see [PERF.md](../PERF.md). The README charts were not replaced with single-run timings.

## Hugging Face publication

The [complete inference bundle](https://huggingface.co/Javascript/Llama-3-Kimodo-Turbo/tree/bcdf39bf070269e6dbab045334c93cb47a73f033)
was verified without authentication: all 56 published bundle files match local SHA-256 hashes
and sizes, including 44 runtime files. The Hub adds its own root `.gitattributes`.
Tokenizer, configuration, statistics, normalization weights and both comparison images passed.
The pinned setup download completed anonymously using existing large weights; shell syntax
and whitespace checks passed.

## Fresh installation and UI

The Hugging Face tutorial was tested from GitHub commit `5900f85` in a new checkout with
a clean `.venv`, UV cache, Cargo dependencies and native build. The unchanged setup script
completed in **46.54 s**. Only the 34 large weight transfers were replaced with verified local
hardlinks; the real anonymous HF client checked metadata and downloaded the other 19 files.
All runtime modules and the compiled motion-correction extension loaded from the new checkout.

Chromium opened the UI on port 8876, keeping the existing server running. With an empty prompt
cache, the Generate motion button produced a genuine 5-second animation in **9.77 s**:
100 diffusion steps, 150 frames, 77 joints, 30 FPS. Playback, pause, scrubbing, 2× speed,
loop end, skeleton visibility and the browser's NPZ download passed. All arrays were finite,
rotation errors stayed below `1e-3`, and API positions matched the downloaded NPZ exactly.
A second real generation with the cached prompt produced identical arrays. No browser or
network errors occurred. Local receipts, NPZ files and screenshots are in `out/fresh-install-qa/`.
These individual runs are functional checks; the benchmark charts retain their original data.
