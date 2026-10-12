# Cleanup verification

Verified on Apple M5 Pro, macOS 26, 2026-10-12.

- Offline CLI: 5-second motion, 100 steps, NPZ and BVH export. Output: 150 × 77 joints; finite values.
- Denoiser parity with fixed text features and identical noise: four cases passed. Maximum joint
  difference: CLI 0.0002 cm; post-processed 0.1208 cm; waypoint constraints 0.1091 cm; padded batch 0.0007 cm.
- Chromium: character loading, generation, playback/pause, scrubbing, skeleton overlay, NPZ download
  and a 390-pixel mobile viewport passed. No browser errors. Downloaded motion: 150 × 77 joints.
- API: empty prompt, out-of-range duration, invalid JSON shape and negative seed return clear errors.
- The executable setup script passed with existing converted weights; full first-time conversion was not rerun.
- Encoder prefetch propagates missing-shard errors; valid and empty streams passed.
  Embeddings remain bit-identical to the committed encoder on two prompts.
- Ruff, formatting and shell syntax checks passed.

Validation used temporary scripts outside the repository. Historical performance evidence is
separate: see [PERF.md](../PERF.md). The README charts were not replaced with single-run timings.
