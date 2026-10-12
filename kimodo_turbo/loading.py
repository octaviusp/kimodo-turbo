"""Build TurboKimodo from a kimodo checkpoint folder (config.yaml + model.safetensors + stats)."""

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download
from kimodo.model.backbone import PositionalEncoding
from kimodo.model.common import instantiate_from_dict
from kimodo.model.registry import MODEL_NAMES, resolve_model_name
from omegaconf import OmegaConf

from .denoiser import TwostageDenoiser
from .paths import WEIGHTS
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


def resolve_checkpoint(modelname: str | None) -> tuple[str, Path]:
    """Use standalone local weights; download a selected model only when absent."""
    name = modelname or DEFAULT_MODEL
    short_key = resolve_model_name(name, "Kimodo")
    repo_id = MODEL_NAMES[short_key]
    checkpoint_dir = WEIGHTS / repo_id.split("/", 1)[1]
    if not (checkpoint_dir / "config.yaml").is_file() or not (checkpoint_dir / "model.safetensors").is_file():
        snapshot_download(repo_id=repo_id, local_dir=checkpoint_dir)
    return short_key, checkpoint_dir


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


def load_turbo_model(modelname: str | None, text_encoder: Callable) -> tuple[TurboKimodo, str]:
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
