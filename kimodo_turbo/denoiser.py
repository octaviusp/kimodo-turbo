"""MLX port of kimodo's TwostageDenoiser (fp32, post-norm transformer encoders).

Mirrors kimodo.model.twostage_denoiser / backbone exactly: the root stage predicts global root
features, which are converted to local root features before the body stage. Constant prefix
tokens (text, heading) are built once per generation by `prepare`, so each denoising step only
embeds the noisy motion and the timestep.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import torch

PREFIX_KEY = "denoiser.backbone."
LAYER_NORM_EPS = 1e-5


def _linear(x: mx.array, params: dict, name: str) -> mx.array:
    return mx.addmm(params[f"{name}.bias"], x, params[f"{name}.weight"].T)


class EncoderStack:
    """One TransformerEncoderBlock: torch.nn.TransformerEncoderLayer stack with norm_first=False."""

    def __init__(self, params: dict, num_layers: int, num_heads: int):
        self.p = params
        self.num_layers = num_layers
        self.num_heads = num_heads

    def embed_prefix(self, text_feat: mx.array, heading_feats: mx.array) -> tuple[mx.array, mx.array]:
        """Text tokens and the first-heading token: constant across denoising steps."""
        text = _linear(text_feat, self.p, "embed_text")  # [B, 50, D]
        heading = _linear(heading_feats, self.p, "linear_first_heading_angle")[:, None]  # [B, 1, D]
        return text, heading

    def embed_time(self, pe_rows: mx.array) -> mx.array:
        hidden = _linear(pe_rows, self.p, "embed_timestep.time_embed.0")
        hidden = hidden * mx.sigmoid(hidden)
        return _linear(hidden, self.p, "embed_timestep.time_embed.2")  # [B, 1, D]

    def __call__(
        self,
        x: mx.array,
        text: mx.array,
        time: mx.array,
        heading: mx.array,
        pe: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        x = _linear(x, self.p, "input_linear")
        seq = mx.concatenate([text, time, heading, x], axis=1)
        prefix_len = seq.shape[1] - x.shape[1]
        seq = seq + pe[:, : seq.shape[1]]
        for i in range(self.num_layers):
            seq = self._layer(seq, f"seqTransEncoder.layers.{i}", mask)
        return _linear(seq[:, prefix_len:], self.p, "output_linear")

    def _layer(self, x: mx.array, name: str, mask: mx.array | None) -> mx.array:
        p = self.p
        batch, length, dim = x.shape
        heads = self.num_heads
        qkv = mx.addmm(p[f"{name}.self_attn.in_proj_bias"], x, p[f"{name}.self_attn.in_proj_weight"].T)
        qkv = qkv.reshape(batch, length, 3, heads, dim // heads).transpose(2, 0, 3, 1, 4)
        attn = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=(dim // heads) ** -0.5, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        attn = _linear(attn, p, f"{name}.self_attn.out_proj")
        x = mx.fast.layer_norm(x + attn, p[f"{name}.norm1.weight"], p[f"{name}.norm1.bias"], LAYER_NORM_EPS)
        hidden = _linear(x, p, f"{name}.linear1")
        hidden = 0.5 * hidden * (1.0 + mx.erf(hidden * (2.0**-0.5)))  # exact GELU, as torch "gelu"
        hidden = _linear(hidden, p, f"{name}.linear2")
        return mx.fast.layer_norm(x + hidden, p[f"{name}.norm2.weight"], p[f"{name}.norm2.bias"], LAYER_NORM_EPS)


@dataclass(frozen=True)
class RootStats:
    """Normalization constants for the global/local root features (kimodo.motion_rep.stats.Stats)."""

    global_mean: mx.array
    global_scale: mx.array
    local_mean: mx.array
    local_scale: mx.array

    @classmethod
    def from_motion_rep(cls, motion_rep) -> "RootStats":
        def scale(stats):
            # Same fp32 expression as Stats.normalize, so the constants are bit-identical.
            return mx.array(torch.sqrt(stats.std**2 + stats.eps).numpy())

        return cls(
            global_mean=mx.array(motion_rep.global_root_stats.mean.numpy()),
            global_scale=scale(motion_rep.global_root_stats),
            local_mean=mx.array(motion_rep.local_root_stats.mean.numpy()),
            local_scale=scale(motion_rep.local_root_stats),
        )


def _repeat_last_valid(values: mx.array, last_index: mx.array) -> mx.array:
    """values[b, len_b - 1] = values[b, len_b - 2], as kimodo's compute_vel_* do. values: [B, T, ...]."""
    frames = mx.arange(values.shape[1])
    is_last = (frames[None, :] == last_index[:, None]).reshape(*values.shape[:2], *([1] * (values.ndim - 2)))
    previous = mx.take_along_axis(
        values, (last_index - 1).reshape(-1, 1, *([1] * (values.ndim - 2))).astype(mx.int32), axis=1
    )
    return mx.where(is_last, previous, values)


def global_root_to_local_root(root: mx.array, stats: RootStats, fps: float, last_index: mx.array) -> mx.array:
    """Port of MotionRepBase.global_root_to_local_root(normalized=True)."""
    root = root * stats.global_scale + stats.global_mean
    root_pos, cos, sin = root[..., :3], root[..., 3], root[..., 4]
    angle = mx.arctan2(sin, cos)
    cos_a, sin_a = mx.cos(angle), mx.sin(angle)
    cos_diff = cos_a[:, 1:] * cos_a[:, :-1] + sin_a[:, 1:] * sin_a[:, :-1]
    sin_diff = sin_a[:, 1:] * cos_a[:, :-1] - cos_a[:, 1:] * sin_a[:, :-1]
    rot_vel = fps * mx.arctan2(sin_diff, cos_diff)
    rot_vel = mx.concatenate([rot_vel, mx.zeros_like(rot_vel[:, :1])], axis=1)
    rot_vel = _repeat_last_valid(rot_vel, last_index)

    vel = fps * (root_pos[:, 1:] - root_pos[:, :-1])
    vel = mx.concatenate([vel, mx.zeros_like(vel[:, :1])], axis=1)
    vel = _repeat_last_valid(vel, last_index)

    local = mx.concatenate([rot_vel[..., None], vel[..., 0:1], vel[..., 2:3], root_pos[..., 1:2]], axis=-1)
    return (local - stats.local_mean) / stats.local_scale


class TwostageDenoiser:
    """Root stage then body stage; same inputs/outputs as kimodo's TwostageDenoiser.forward."""

    def __init__(self, params: dict, motion_rep, num_layers: int, num_heads: int, pe: np.ndarray):
        self.root = EncoderStack(_subtree(params, "root_model."), num_layers, num_heads)
        self.body = EncoderStack(_subtree(params, "body_model."), num_layers, num_heads)
        self.stats = RootStats.from_motion_rep(motion_rep)
        self.fps = float(motion_rep.fps)
        self.body_start = motion_rep.body_slice.start
        self.pe = mx.array(pe)  # [1, max_len, D]

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, motion_rep, num_layers: int, num_heads: int, pe: np.ndarray):
        weights = mx.load(ckpt_path)
        params = {k[len(PREFIX_KEY) :]: v for k, v in weights.items() if k.startswith(PREFIX_KEY)}
        if len(params) != len(weights):
            raise ValueError(
                f"Unexpected keys in {ckpt_path}: {sorted(set(weights) - set(PREFIX_KEY + k for k in params))}"
            )
        mx.eval(params)
        return cls(params, motion_rep, num_layers, num_heads, pe)

    def prepare(self, text_feat: mx.array, first_heading_angle: mx.array) -> dict:
        heading_feats = mx.stack([mx.cos(first_heading_angle), mx.sin(first_heading_angle)], axis=-1)
        return {
            "root": self.root.embed_prefix(text_feat, heading_feats),
            "body": self.body.embed_prefix(text_feat, heading_feats),
        }

    def __call__(
        self,
        x: mx.array,
        prefix: dict,
        timesteps: mx.array,
        motion_mask: mx.array,
        observed_motion: mx.array,
        key_mask: mx.array | None,
        last_index: mx.array,
    ) -> mx.array:
        x = x * (1 - motion_mask) + observed_motion * motion_mask
        pe_rows = self.pe[0][timesteps][:, None]  # TimestepEmbedder input, [B, 1, D]

        text, heading = prefix["root"]
        root_pred = self.root(
            mx.concatenate([x, motion_mask], axis=-1), text, self.root.embed_time(pe_rows), heading, self.pe, key_mask
        )
        local_root = global_root_to_local_root(root_pred, self.stats, self.fps, last_index)

        text, heading = prefix["body"]
        body_in = mx.concatenate([local_root, x[..., self.body_start :], motion_mask], axis=-1)
        body_pred = self.body(body_in, text, self.body.embed_time(pe_rows), heading, self.pe, key_mask)
        return mx.concatenate([root_pred, body_pred], axis=-1)


def _subtree(params: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}
