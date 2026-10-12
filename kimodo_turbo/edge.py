"""Edge engine: the file the autoresearch loop edits (bench/edge_eval.py is the fixed harness).

Contract: build_engine() returns an object whose generate(prompt, seconds, seed) produces one
motion as a dict of numpy arrays without a sample axis (posed_joints [T,77,3], foot_contacts,
root_positions, ...), exactly like kimodo_gen's single-sample output.

Sampling is the probability-flow ODE of Kimodo's x0-predicting denoiser, solved with DDIM
(first order, Kimodo's default) or DPM-Solver++ multistep in data-prediction form, on a
configurable integer timestep schedule. Precision policy: the transformer stacks may run in
bf16/fp16 while the root velocity conversion and solver arithmetic stay fp32.
"""

import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import torch

from kimodo.tools import seed_everything

from .cli import SHARDS
from .denoiser import global_root_to_local_root
from .encoder import StreamingLlm2Vec
from .loading import load_turbo_model
from .pipeline import CfgPlan, TurboKimodo, _pad_text

os.environ.setdefault("TQDM_DISABLE", "1")
DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


@dataclass(frozen=True)
class EdgeConfig:
    steps: int = 100
    solver: str = "ddim"  # ddim | dpmpp2m
    spacing: str = "uniform"  # uniform | logsnr
    dtype: str = "float32"  # transformer compute dtype
    root_fp32: bool = False  # keep the root stage in fp32 when dtype is lower


CONFIG = EdgeConfig()


def schedule(alphas_cumprod: np.ndarray, steps: int, spacing: str) -> np.ndarray:
    """Descending integer timesteps ending at 0 (Kimodo's DDIM spacing for 'uniform')."""
    top = len(alphas_cumprod) - 1
    if spacing == "uniform":
        ts = np.round(np.arange(steps) * (top / max(1, steps - 1)))
    elif spacing == "logsnr":
        lam = 0.5 * np.log(alphas_cumprod / (1 - alphas_cumprod))
        targets = np.linspace(lam[top], lam[0], steps)
        ts = np.array([np.argmin(np.abs(lam - x)) for x in targets])[::-1]
    else:
        raise ValueError(spacing)
    ts = np.unique(np.clip(ts, 0, top).astype(np.int64))[::-1]
    return ts


class LowPrecisionStacks:
    """Casts the denoiser's transformer parameters once; activations follow the parameter dtype."""

    def __init__(self, denoiser, dtype, root_fp32: bool):
        self.denoiser = denoiser
        self.dtype = dtype
        for stack, keep in ((denoiser.root, root_fp32), (denoiser.body, False)):
            target = mx.float32 if keep else dtype
            stack.p = {k: v.astype(target) for k, v in stack.p.items()}
            stack.dtype = target
        denoiser.pe_by_dtype = {d: denoiser.pe.astype(d) for d in {mx.float32, dtype}}


class EdgeKimodo(TurboKimodo):
    config = CONFIG

    def denoise(self, x, prefix, timesteps, motion_mask, observed, key_mask, last_index):
        """TwostageDenoiser.__call__ with per-stage compute dtypes (fp32 root velocities)."""
        d = self.mlx_denoiser
        x = x * (1 - motion_mask) + observed * motion_mask
        root_dtype, body_dtype = getattr(d.root, "dtype", mx.float32), getattr(d.body, "dtype", mx.float32)
        pe = getattr(d, "pe_by_dtype", {mx.float32: d.pe})
        pe_rows = d.pe[0][timesteps][:, None]

        text, heading = prefix["root"]
        root_in = mx.concatenate([x, motion_mask], axis=-1).astype(root_dtype)
        root_mask = None if key_mask is None else key_mask.astype(root_dtype)
        root_pred = d.root(root_in, text, d.root.embed_time(pe_rows.astype(root_dtype)), heading, pe[root_dtype], root_mask)
        root_pred = root_pred.astype(mx.float32)
        local_root = global_root_to_local_root(root_pred, d.stats, d.fps, last_index)

        text, heading = prefix["body"]
        body_in = mx.concatenate([local_root, x[..., d.body_start:], motion_mask], axis=-1).astype(body_dtype)
        body_mask = None if key_mask is None else key_mask.astype(body_dtype)
        body_pred = d.body(body_in, text, d.body.embed_time(pe_rows.astype(body_dtype)), heading, pe[body_dtype], body_mask)
        return mx.concatenate([root_pred, body_pred.astype(mx.float32)], axis=-1)

    def conditioning(self, text_feat, pad_mask, first_heading_angle, motion_mask=None, observed_motion=None,
                     cfg_weight=(2.0, 2.0), cfg_type=None) -> dict:
        """Per-generation constants: CFG rows, prefix tokens, masks (shared by every solver step)."""
        batch, frames = text_feat.shape[0], pad_mask.shape[1]
        shape = (batch, frames, self.motion_rep.motion_rep_dim)
        if motion_mask is None or observed_motion is None:
            motion_mask = torch.zeros(shape)
            observed_motion = torch.zeros(shape)
        motion_mask = motion_mask.to(torch.float32)
        plan = CfgPlan(cfg_type or self.cfg_type_default, cfg_weight, constrained=bool(motion_mask.any()))
        rows = len(plan.rows)
        d = self.mlx_denoiser
        text_rows = {"text": _pad_text(text_feat, self.num_text_tokens)}
        text_rows["zero"] = mx.zeros_like(text_rows["text"])
        mask = mx.array(motion_mask.numpy())
        mask_rows = {"mask": mask, "zero": mx.zeros_like(mask)}
        row_text = mx.concatenate([text_rows[t] for t, _ in plan.rows])
        heading = mx.concatenate([mx.array(first_heading_angle.to(torch.float32).cpu().numpy())] * rows)
        prefix = {}
        for name in ("root", "body"):
            stack = getattr(d, name)
            dtype = getattr(stack, "dtype", mx.float32)
            prefix[name] = stack.embed_prefix(row_text.astype(dtype), heading.astype(dtype))
        valid = pad_mask.cpu().numpy().astype(bool)
        key_mask = None
        if not valid.all():
            keys = np.concatenate([np.ones((batch, prefix["root"][0].shape[1] + 2), bool), valid], axis=1)
            key_mask = mx.array(np.where(np.tile(keys, (rows, 1)), 0.0, -np.inf).astype(np.float32)[:, None, None, :])
        return {
            "plan": plan, "rows": rows, "batch": batch, "prefix": prefix, "key_mask": key_mask,
            "row_mask": mx.concatenate([mask_rows[m] for _, m in plan.rows]),
            "row_observed": mx.concatenate([mx.array(observed_motion.numpy())] * rows),
            "last_index": mx.array(np.tile(valid.sum(-1) - 1, rows).astype(np.int32)),
        }

    def solve(self, x: mx.array, cond: dict, steps: int, config: EdgeConfig) -> mx.array:
        """Integrate the guided probability-flow ODE from noise x (fp32) to a clean motion."""
        rows, batch, plan = cond["rows"], cond["batch"], cond["plan"]

        def model_x0(x, t):
            timesteps = mx.broadcast_to(t, (rows * batch,))
            out = self.denoise(mx.concatenate([x] * rows), cond["prefix"], timesteps, cond["row_mask"],
                               cond["row_observed"], cond["key_mask"], cond["last_index"])
            return plan.combine(mx.split(out, rows))

        model_x0 = mx.compile(model_x0)
        ab = self.diffusion.alphas_cumprod_base.double().numpy()
        ts = schedule(ab, steps, config.spacing)
        alpha, sigma = np.sqrt(ab), np.sqrt(1 - ab)
        lam = np.log(alpha / sigma)
        prev_x0, prev_h = None, None
        for i, t in enumerate(ts):
            x0 = model_x0(x, mx.array(int(t), dtype=mx.int32))
            if i == len(ts) - 1:
                return x0
            s = ts[i + 1]
            h = lam[s] - lam[t]
            if config.solver == "ddim" or prev_x0 is None:
                D = x0
            elif config.solver == "dpmpp2m":
                r = prev_h / h
                D = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * prev_x0
            else:
                raise ValueError(config.solver)
            x = float(sigma[s] / sigma[t]) * x - float(alpha[s] * np.expm1(-h)) * D
            mx.async_eval(x)
            prev_x0, prev_h = x0, h
        return x

    def _generate(self, texts, max_frames, num_denoising_steps, pad_mask, first_heading_angle, motion_mask,
                  observed_motion, cfg_weight=2.0, text_feat=None, text_pad_mask=None, guide_masks=None,
                  cfg_type=None, progress_bar=None):
        if text_feat is None:
            text_feat = self.encode_texts(texts)
        shape = (text_feat.shape[0], int(max_frames), self.motion_rep.motion_rep_dim)
        noise = torch.randn(shape, device="cpu")  # same RNG call as Kimodo: a seed gives the same noise
        cond = self.conditioning(text_feat, pad_mask, first_heading_angle, motion_mask, observed_motion, cfg_weight, cfg_type)
        x = self.solve(mx.array(noise.numpy()), cond, num_denoising_steps, self.config)
        return torch.from_numpy(np.array(x.astype(mx.float32), copy=True))


class EdgeEngine:
    def __init__(self, config: EdgeConfig = CONFIG):
        self.name = f"{config.solver}-{config.steps}-{config.spacing}-{config.dtype}{'-rootfp32' if config.root_fp32 else ''}"
        self.encoder = StreamingLlm2Vec(SHARDS)
        model, _ = load_turbo_model(None, self.encoder)
        model.__class__ = EdgeKimodo
        model.config = config
        if config.dtype != "float32":
            LowPrecisionStacks(model.mlx_denoiser, DTYPES[config.dtype], config.root_fp32)
        self.model = model
        self.config = config

    def generate(self, prompt: str, seconds: float, seed: int) -> dict:
        seed_everything(seed)
        out = self.model(
            [prompt], [int(round(seconds * self.model.fps))], num_denoising_steps=self.config.steps, num_samples=1,
            multi_prompt=True, post_processing=True, return_numpy=True,
        )
        return {key: value[0] for key, value in out.items()}


def build_engine() -> EdgeEngine:
    return EdgeEngine(CONFIG)
