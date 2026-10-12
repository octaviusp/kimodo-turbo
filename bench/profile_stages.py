"""Stage timings of one resident TurboKimodo generation (5 s motion), warm, median of N.

Stages: text encoder (no cache), denoising loop (_generate with given text features), and the rest
of Kimodo.__call__ (conditions, inverse/FK, post-processing, SOMA77 conversion).
"""

import statistics
import sys
import time

import mlx.core as mx
import torch

from kimodo.tools import seed_everything
from kimodo_turbo.cli import SHARDS
from kimodo_turbo.encoder import StreamingLlm2Vec
from kimodo_turbo.loading import load_turbo_model

PROMPT = "A person runs forward at a steady pace."
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


class Fixed:
    def __init__(self, feat):
        self.feat = feat

    def __call__(self, texts):
        return self.feat.repeat(len(texts), 1, 1), [1] * len(texts)


def timed(fn, reps=REPS):
    fn()  # warm-up (compile, page cache)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        mx.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    encoder = StreamingLlm2Vec(SHARDS)
    model, _ = load_turbo_model(None, encoder)
    frames = 150

    t_encode = timed(lambda: encoder([PROMPT]))
    feat = encoder([PROMPT])[0]
    model.text_encoder = Fixed(feat)

    def call(post: bool):
        seed_everything(0)
        model([PROMPT], [frames], num_denoising_steps=100, num_samples=1, multi_prompt=True,
              post_processing=post, return_numpy=True, progress_bar=lambda x: x)

    t_full_post = timed(lambda: call(True))
    t_full_nopost = timed(lambda: call(False))

    pad = torch.ones(1, frames, dtype=torch.bool)
    zeros = torch.zeros(1, frames, model.motion_rep.motion_rep_dim)

    def denoise():
        model._generate([PROMPT], frames, 100, pad, torch.zeros(1), zeros, zeros, cfg_weight=[2.0, 2.0],
                        text_feat=feat, progress_bar=lambda x: x)

    t_denoise = timed(denoise)
    print(f"text encoder (stream, warm)  {t_encode * 1000:8.1f} ms")
    print(f"denoise 100 steps            {t_denoise * 1000:8.1f} ms")
    print(f"call w/o postprocess         {t_full_nopost * 1000:8.1f} ms  (other = {(t_full_nopost - t_denoise) * 1000:.1f} ms)")
    print(f"call with postprocess        {t_full_post * 1000:8.1f} ms  (postprocess = {(t_full_post - t_full_nopost) * 1000:.1f} ms)")
    print(f"end-to-end estimate          {(t_encode + t_full_post) * 1000:8.1f} ms")


if __name__ == "__main__":
    main()
