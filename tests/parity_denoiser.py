"""Parity of the MLX denoiser against kimodo's torch fp32 denoiser, with a fixed text embedding.

Checks one forward pass on random inputs, then full generations through Kimodo.__call__ for
both models with the same seed: kimodo_gen's multi-prompt path, post-processing, a 2D waypoint
constraint (three distinct CFG rows) and a batch with different lengths (key padding mask). Run with
MLX_ENABLE_TF32=0 to compare full-fp32 GEMMs.
"""

import os
import sys
import time
import zlib

import mlx.core as mx
import numpy as np
import torch

from kimodo.constraints import Root2DConstraintSet
from kimodo.model.common import instantiate_from_dict
from kimodo.model.kimodo_model import Kimodo
from kimodo.tools import seed_everything
from kimodo_turbo.loading import load_config, load_turbo_model, resolve_checkpoint


class FixedTextEncoder:
    """Deterministic stand-in for LLM2Vec: one 4096-d vector per prompt."""

    def __call__(self, texts):
        rows = []
        for text in texts:
            gen = torch.Generator().manual_seed(zlib.crc32(text.encode()))
            rows.append(torch.randn(4096, generator=gen) * 0.05)
        return torch.stack(rows)[:, None], [1] * len(texts)


def max_rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a - b)
    return float(diff.max()), float(diff.max() / (np.abs(b).max() + 1e-12))


def main() -> int:
    encoder = FixedTextEncoder()
    turbo, _ = load_turbo_model(None, encoder)
    _, checkpoint_dir = resolve_checkpoint(None)
    cfg = load_config(checkpoint_dir)
    reference = Kimodo(
        denoiser=instantiate_from_dict(cfg["denoiser"]),
        text_encoder=encoder,
        num_base_steps=cfg["num_base_steps"],
        device="cpu",
        cfg_type=cfg["cfg_type"],
    ).eval()

    # 1) Single forward on random inputs (rows of a separated-CFG batch).
    torch.manual_seed(0)
    batch, frames, dim = 3, 150, turbo.motion_rep.motion_rep_dim
    x = torch.randn(batch, frames, dim)
    text = encoder(["A person walks forward."] * batch)[0]
    text[1:] = 0
    mask = torch.zeros(batch, frames, dim)
    observed = torch.zeros(batch, frames, dim)
    heading = torch.zeros(batch)
    timesteps = torch.tensor([999, 500, 3])
    pad = torch.ones(batch, frames, dtype=torch.bool)
    with torch.inference_mode():
        ref = reference.denoiser.model(x, pad, text, torch.ones(batch, 1, dtype=torch.bool), timesteps, heading, mask, observed)
    from kimodo_turbo.pipeline import _pad_text

    prefix = turbo.mlx_denoiser.prepare(_pad_text(text, turbo.num_text_tokens), mx.array(heading.numpy()))
    out = turbo.mlx_denoiser(
        mx.array(x.numpy()), prefix, mx.array(timesteps.numpy().astype(np.int32)), mx.array(mask.numpy()),
        mx.array(observed.numpy()), None, mx.full((batch,), frames - 1, dtype=mx.int32),
    )
    abs_err, rel_err = max_rel(np.array(out), ref.numpy())
    print(f"forward: max_abs={abs_err:.3e} max_rel={rel_err:.3e}")

    # 2) Full generations: CLI path, waypoint constraints (3 distinct CFG rows), padded batch.
    waypoints = Root2DConstraintSet(
        turbo.skeleton, torch.tensor([0, 60, 119]), torch.tensor([[0.0, 0.0], [1.0, 0.6], [2.0, 0.0]])
    )
    cases = [
        ("walk, CLI path", dict(prompts=["A person walks forward."], num_frames=[150], multi_prompt=True, num_samples=1)),
        ("dance, post-processed", dict(prompts=["Someone dances and spins."], num_frames=[120], multi_prompt=True, num_samples=1,
                                       post_processing=True)),
        ("waypoints constraint", dict(prompts=["A person walks along a path."], num_frames=[120], multi_prompt=True, num_samples=1,
                                      constraint_lst=[waypoints], post_processing=True)),
        ("padded batch", dict(prompts=["A person walks forward.", "A person jumps."], num_frames=[150, 105])),
    ]
    failures = 0
    for label, kwargs in cases:
        outputs = {}
        for name, model in [("reference", reference), ("turbo", turbo)]:
            seed_everything(0)
            start = time.perf_counter()
            outputs[name] = model(**kwargs, num_denoising_steps=100, return_numpy=True, progress_bar=lambda x: x)
            print(f"  {name}: {time.perf_counter() - start:.2f}s")
        joints = np.linalg.norm(outputs["turbo"]["posed_joints"] - outputs["reference"]["posed_joints"], axis=-1)
        print(f"{label}: max joint diff={joints.max() * 100:.4f} cm mean={joints.mean() * 100:.4f} cm")
        failures += joints.max() > 5e-3  # 5 mm with identical text features
    return 1 if failures or rel_err > 1e-3 else 0

if __name__ == "__main__":
    print("MLX_ENABLE_TF32 =", os.environ.get("MLX_ENABLE_TF32", "<default>"))
    sys.exit(main())
