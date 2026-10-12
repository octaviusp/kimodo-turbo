"""Kimodo with the denoising loop on MLX; every other stage is kimodo's own torch code.

TurboKimodo subclasses kimodo.model.Kimodo and overrides only `_generate` (text encoding +
diffusion loop). Constraint handling, multi-prompt stitching, inverse kinematics,
post-processing and export run unchanged, so their semantics cannot drift.
"""

from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.model.kimodo_model import Kimodo

from .denoiser import TwostageDenoiser


class DdimSchedule:
    """Per-step DDIM constants computed by kimodo's own torch Diffusion (bit-identical fp32)."""

    def __init__(self, diffusion: Diffusion, num_steps: int):
        use_timesteps, map_tensor = diffusion.space_timesteps(num_steps)
        diffusion.calc_diffusion_vars(use_timesteps)
        steps = torch.arange(num_steps)
        self.model_t = map_tensor[steps].numpy()
        alpha_bar_prev = diffusion.alphas_cumprod_prev[steps]
        self.recip = diffusion.sqrt_recip_alphas_cumprod[steps].numpy()
        self.recipm1 = diffusion.sqrt_recipm1_alphas_cumprod[steps].numpy()
        self.sqrt_prev = torch.sqrt(alpha_bar_prev).numpy()
        self.sqrt_one_minus_prev = torch.sqrt(1 - alpha_bar_prev).numpy()

        self.coefficients = mx.array(np.stack([self.recip, self.recipm1, self.sqrt_prev, self.sqrt_one_minus_prev], 1))

    @staticmethod
    def step(coef: mx.array, x_t: mx.array, pred_xstart: mx.array) -> mx.array:
        """DDIMSampler.__call__ (eta = 0) with the same operation order; coef = one row of coefficients."""
        eps = (coef[0] * x_t - pred_xstart) / coef[1]
        return pred_xstart * coef[2] + coef[3] * eps


class CfgPlan:
    """Rows to evaluate per step and how to combine them (ClassifierFreeGuidedModel semantics).

    Rows whose inputs are provably identical are evaluated once: with separated CFG and no
    active constraint, the constraint row equals the unconditional row, so its guidance term
    is exactly zero.
    """

    def __init__(self, cfg_type: str, cfg_weight, constrained: bool):
        self.cfg_type = cfg_type
        self.weight = cfg_weight
        if cfg_type == "nocfg":
            self.rows = [("text", "mask")]
        elif cfg_type == "regular":
            self.rows = [("text", "mask"), ("zero", "zero")]
        elif cfg_type == "separated":
            self.rows = [("text", "zero"), ("zero", "mask"), ("zero", "zero")] if constrained else [
                ("text", "zero"),
                ("zero", "zero"),
            ]
        else:
            raise ValueError(f"Invalid cfg_type: {cfg_type}")

    def combine(self, outs: List[mx.array]) -> mx.array:
        if self.cfg_type == "nocfg":
            return outs[0]
        if self.cfg_type == "regular":
            out, uncond = outs
            return uncond + (self.weight * (out - uncond))
        if len(outs) == 3:
            text, constraint, uncond = outs
            return uncond + (self.weight[0] * (text - uncond)) + (self.weight[1] * (constraint - uncond))
        text, uncond = outs
        return uncond + (self.weight[0] * (text - uncond))


class TurboKimodo(Kimodo):
    """Drop-in Kimodo whose text encoder and denoiser run on MLX."""

    def __init__(
        self,
        denoiser: TwostageDenoiser,
        motion_rep,
        text_encoder,
        num_base_steps: int,
        cfg_type: str,
        num_text_tokens: int,
        compile_steps: bool = True,
    ):
        nn.Module.__init__(self)
        self.compile_steps = compile_steps
        self.mlx_denoiser = denoiser
        self.motion_rep = motion_rep
        self.skeleton = motion_rep.skeleton
        self.fps = motion_rep.fps
        self.diffusion = Diffusion(num_base_steps=num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        self.text_encoder = text_encoder
        self.cfg_type_default = cfg_type
        self.num_text_tokens = num_text_tokens
        self.device = "cpu"

    def train(self, mode: bool):
        if mode:
            raise RuntimeError("TurboKimodo is inference-only")
        return self

    def eval(self):
        return self

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Same contract as Kimodo._generate's text branch: [B, 1, llm_dim] fp32, empty texts zeroed."""
        text_feat, _ = self.text_encoder(texts)
        text_feat = text_feat.to("cpu")
        empty_text_mask = [len(text.strip()) == 0 for text in texts]
        text_feat[empty_text_mask] = 0
        return text_feat

    def _generate(
        self,
        texts: List[str],
        max_frames: int,
        num_denoising_steps: int,
        pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        cfg_weight: Optional[float] = 2.0,
        text_feat: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
        progress_bar=tqdm,
    ) -> torch.Tensor:
        if guide_masks is not None:
            raise NotImplementedError("guide_masks is not used by kimodo's sampler and is not supported")
        if text_feat is None:
            text_feat = self.encode_texts(texts)
        batch_size = text_feat.shape[0]
        shape = (batch_size, int(max_frames), self.motion_rep.motion_rep_dim)
        # Same torch RNG call as the reference, so a given seed yields the same initial noise.
        noise = torch.randn(shape, device="cpu")

        if motion_mask is None or observed_motion is None:
            motion_mask = torch.zeros(shape)
            observed_motion = torch.zeros(shape)
        motion_mask = motion_mask.to(torch.float32)
        plan = CfgPlan(cfg_type or self.cfg_type_default, cfg_weight, constrained=bool(motion_mask.any()))

        text_rows = {"text": _pad_text(text_feat, self.num_text_tokens), "zero": None}
        text_rows["zero"] = mx.zeros_like(text_rows["text"])
        mask = mx.array(motion_mask.numpy())
        mask_rows = {"mask": mask, "zero": mx.zeros_like(mask)}
        rows = len(plan.rows)
        row_text = mx.concatenate([text_rows[t] for t, _ in plan.rows])
        row_mask = mx.concatenate([mask_rows[m] for _, m in plan.rows])
        row_observed = mx.concatenate([mx.array(observed_motion.numpy())] * rows)
        heading = mx.array(first_heading_angle.to(torch.float32).cpu().numpy())
        prefix = self.mlx_denoiser.prepare(row_text, mx.concatenate([heading] * rows))

        valid = pad_mask.cpu().numpy().astype(bool)
        last_index = mx.array(np.tile(valid.sum(-1) - 1, rows).astype(np.int32))
        key_mask = None
        if not valid.all():
            keys = np.concatenate([np.ones((batch_size, prefix["root"][0].shape[1] + 2), bool), valid], axis=1)
            additive = np.where(np.tile(keys, (rows, 1)), 0.0, -np.inf).astype(np.float32)
            key_mask = mx.array(additive[:, None, None, :])

        def denoise_step(x: mx.array, model_t: mx.array, coef: mx.array) -> mx.array:
            timesteps = mx.broadcast_to(model_t, (rows * batch_size,))
            out = self.mlx_denoiser(
                mx.concatenate([x] * rows), prefix, timesteps, row_mask, row_observed, key_mask, last_index
            )
            return DdimSchedule.step(coef, x, plan.combine(mx.split(out, rows)))

        step_fn = mx.compile(denoise_step) if self.compile_steps else denoise_step
        schedule = DdimSchedule(self.diffusion, num_denoising_steps)
        model_t = mx.array(schedule.model_t.astype(np.int32))
        x = mx.array(noise.numpy())
        for i in progress_bar(list(range(num_denoising_steps))[::-1]):
            x = step_fn(x, model_t[i], schedule.coefficients[i])
            mx.async_eval(x)
        return torch.from_numpy(np.array(x, copy=True))


def _pad_text(text_feat: torch.Tensor, num_tokens: int) -> mx.array:
    """pad_x_and_mask_to_fixed_size for the [B, L, llm_dim] text features."""
    batch, length, dim = text_feat.shape
    padded = np.zeros((batch, num_tokens, dim), np.float32)
    padded[:, :length] = text_feat.numpy()
    return mx.array(padded)
