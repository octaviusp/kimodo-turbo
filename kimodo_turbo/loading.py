"""Build TurboKimodo from a kimodo checkpoint folder (config.yaml + model.safetensors + stats)."""

from pathlib import Path
from typing import Callable, Optional

from omegaconf import OmegaConf

from kimodo.model.backbone import PositionalEncoding
from kimodo.model.common import instantiate_from_dict
from kimodo.model.load_model import _resolve_hf_model_path
from kimodo.model.registry import resolve_model_name

from .denoiser import TwostageDenoiser
from .pipeline import TurboKimodo

DEFAULT_MODEL = "Kimodo-SOMA-RP-v1"

# Architecture facts the MLX port implements; anything else must fail loudly, not run silently.
SUPPORTED_DENOISER = {
    "_target_": "kimodo.model.twostage_denoiser.TwostageDenoiser",
    "motion_mask_mode": "concat",
    "use_text_mask": False,
    "activation": "gelu",
    "norm_first": False,
    "dropout": 0.0,
    "pe_dropout": 0.0,
    "input_first_heading_angle": True,
}


def resolve_checkpoint(modelname: Optional[str]) -> tuple[str, Path]:
    name = modelname or DEFAULT_MODEL
    short_key = resolve_model_name(name, "Kimodo")
    return short_key, _resolve_hf_model_path(short_key)


def load_config(checkpoint_dir: Path) -> dict:
    conf = OmegaConf.merge(
        OmegaConf.load(checkpoint_dir / "config.yaml"),
        OmegaConf.create({"checkpoint_dir": str(checkpoint_dir)}),
    )
    cfg = OmegaConf.to_container(conf, resolve=True)
    cfg.pop("checkpoint_dir")
    return cfg


def _check_supported(denoiser_cfg: dict) -> None:
    mismatched = {k: denoiser_cfg.get(k) for k, v in SUPPORTED_DENOISER.items() if denoiser_cfg.get(k) != v}
    if mismatched:
        raise NotImplementedError(f"Checkpoint uses denoiser settings the MLX port does not implement: {mismatched}")


def build_denoiser(cfg: dict):
    """Return (mlx denoiser, kimodo motion_rep) for a resolved checkpoint config."""
    denoiser_cfg = cfg["denoiser"]
    _check_supported(denoiser_cfg)
    motion_rep = instantiate_from_dict(denoiser_cfg["motion_rep"])
    pe = PositionalEncoding(denoiser_cfg["latent_dim"], denoiser_cfg["pe_dropout"]).pe.numpy()
    denoiser = TwostageDenoiser.from_checkpoint(
        denoiser_cfg["ckpt_path"],
        motion_rep,
        num_layers=denoiser_cfg["num_layers"],
        num_heads=denoiser_cfg["num_heads"],
        pe=pe,
    )
    return denoiser, motion_rep


def load_turbo_model(modelname: Optional[str], text_encoder: Callable) -> tuple[TurboKimodo, str]:
    short_key, checkpoint_dir = resolve_checkpoint(modelname)
    cfg = load_config(checkpoint_dir)
    denoiser, motion_rep = build_denoiser(cfg)
    model = TurboKimodo(
        denoiser,
        motion_rep,
        text_encoder,
        num_base_steps=cfg["num_base_steps"],
        cfg_type=cfg["cfg_type"],
        num_text_tokens=cfg["denoiser"]["num_text_tokens_override"] or cfg["denoiser"]["llm_shape"][0],
    )
    return model, short_key
