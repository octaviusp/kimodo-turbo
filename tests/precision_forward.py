"""FP64 oracle for one denoiser forward: how far are torch-fp32 (reference) and MLX from exact?"""

import mlx.core as mx
import numpy as np
import torch

from kimodo.model.common import instantiate_from_dict
from kimodo_turbo.loading import load_config, load_turbo_model, resolve_checkpoint
from kimodo_turbo.pipeline import _pad_text


def main() -> None:
    turbo, _ = load_turbo_model(None, text_encoder=None)
    cfg = load_config(resolve_checkpoint(None)[1])
    ref = instantiate_from_dict(cfg["denoiser"]).eval()

    torch.manual_seed(0)
    batch, frames, dim = 2, 150, turbo.motion_rep.motion_rep_dim
    x = torch.randn(batch, frames, dim)
    text = torch.randn(batch, 1, 4096) * 0.05
    text[1:] = 0
    zeros = torch.zeros(batch, frames, dim)
    heading = torch.zeros(batch)
    timesteps = torch.tensor([999, 500])
    pad = torch.ones(batch, frames, dtype=torch.bool)
    tmask = torch.ones(batch, 1, dtype=torch.bool)

    def run_torch(dtype):
        model = ref.to(dtype)
        model.motion_rep.global_root_stats.mean = model.motion_rep.global_root_stats.mean.to(dtype)
        with torch.inference_mode():
            return model(x.to(dtype), pad, text.to(dtype), tmask, timesteps, heading.to(dtype), zeros.to(dtype), zeros.to(dtype)).double().numpy()

    exact = run_torch(torch.float64)
    torch_fp32 = run_torch(torch.float32)
    prefix = turbo.mlx_denoiser.prepare(_pad_text(text, turbo.num_text_tokens), mx.array(heading.numpy()))
    mlx_out = np.array(
        turbo.mlx_denoiser(
            mx.array(x.numpy()), prefix, mx.array(timesteps.numpy().astype(np.int32)), mx.array(zeros.numpy()),
            mx.array(zeros.numpy()), None, mx.full((batch,), frames - 1, dtype=mx.int32),
        )
    ).astype(np.float64)
    scale = np.abs(exact).max()
    for name, value in [("torch fp32", torch_fp32), ("mlx fp32", mlx_out)]:
        err = np.abs(value - exact)
        print(f"{name:10s} vs fp64: max_abs={err.max():.3e} max_rel={err.max() / scale:.3e} mean_abs={err.mean():.3e}")


if __name__ == "__main__":
    main()
