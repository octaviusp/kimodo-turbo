"""Denoising-loop speed: eager vs mx.compile'd step, interleaved, same seed; checks they agree.

Uses a fixed text embedding so only the 100-step loop is measured (post-processing off).
"""

import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from kimodo.tools import seed_everything
from kimodo_turbo.loading import load_turbo_model


class FixedEncoder:
    def __call__(self, texts):
        gen = torch.Generator().manual_seed(0)
        return (torch.randn(len(texts), 1, 4096, generator=gen) * 0.05), [1] * len(texts)


def run(model, seconds: float) -> tuple[float, np.ndarray]:
    seed_everything(0)
    start = time.perf_counter()
    out = model(["A person walks forward."], [int(seconds * model.fps)], num_denoising_steps=100, num_samples=1,
                multi_prompt=True, post_processing=False, return_numpy=True)
    return time.perf_counter() - start, out["posed_joints"]


def main() -> None:
    model, _ = load_turbo_model(None, FixedEncoder())
    for seconds in (5.0, 10.0):
        times = {False: [], True: []}
        joints = {}
        for compiled in (False, True):  # warm-up (includes compile trace)
            model.compile_steps = compiled
            run(model, seconds)
        for _ in range(3):
            for compiled in (False, True):
                model.compile_steps = compiled
                elapsed, joints[compiled] = run(model, seconds)
                times[compiled].append(elapsed)
        diff = np.abs(joints[True] - joints[False]).max() * 100
        print(f"{seconds:.0f}s motion: eager median={statistics.median(times[False]):.2f}s "
              f"compiled median={statistics.median(times[True]):.2f}s  max joint diff={diff:.5f} cm")
    print(f"mlx peak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
