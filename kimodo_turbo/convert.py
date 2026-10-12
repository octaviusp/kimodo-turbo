"""Export the in-memory reference LLM2Vec into streamable shards for StreamingLlm2Vec.

Weights are taken from the loaded reference model itself (after kimodo's own adapter loading
and merging), so the shards hold exactly what the reference computes with: base projections
in bf16 and the unmerged supervised LoRA in its runtime dtype.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .encoder import PROJECTIONS, SEPARATOR

PROBE = "probe text"


def load_reference_encoder():
    """kimodo's own LLM2Vec text encoder on CPU (bf16, ~17 GB RAM while loaded)."""
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS

    return LLM2VecEncoder(**{**TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"], "device": "cpu"})


def decoder_layers(llm2vec) -> list:
    return [m for m in llm2vec.model.modules() if type(m).__name__.endswith("DecoderLayer")]


def attention_is_causal(encoder) -> bool:
    """Run one probe and inspect what the first attention layer actually receives."""
    seen = []
    attn = decoder_layers(encoder.model)[0].self_attn
    handle = attn.register_forward_pre_hook(lambda m, a, kw: seen.append(kw.get("attention_mask")), with_kwargs=True)
    try:
        encoder([PROBE])
    finally:
        handle.remove()
    mask = seen[0]
    if mask is None:
        # SDPA then falls back to the module flag; kimodo's LLM2Vec layers set is_causal=False.
        return bool(getattr(attn, "is_causal", True))
    return bool((mask[0, 0].float().triu(1) < 0).any())


def _rope_theta(config) -> float:
    params = getattr(config, "rope_parameters", None) or {}
    if params.get("rope_type", "default") != "default":
        raise NotImplementedError(f"Unsupported RoPE scaling: {params}")
    return float(params.get("rope_theta", getattr(config, "rope_theta", None)))


def _template(llm2vec) -> dict | None:
    prepared = llm2vec.prepare_for_tokenization(SEPARATOR + PROBE)
    if prepared == SEPARATOR + PROBE:
        return None
    prefix, rest = prepared.split(SEPARATOR, 1)
    if not rest.startswith(PROBE):
        raise NotImplementedError(f"Unexpected prompt template: {prepared!r}")
    return {"prefix": prefix, "suffix": rest[len(PROBE) :]}


def _lora_parts(module, layer_index: int, name: str) -> tuple[dict, float | None]:
    if not hasattr(module, "lora_A"):
        return {f"{name}.weight": module.weight.detach().contiguous()}, None
    if module.merged_adapters or len(module.active_adapters) != 1:
        raise NotImplementedError(f"layer {layer_index} {name}: merged={module.merged_adapters} active={module.active_adapters}")
    adapter = module.active_adapters[0]
    if module.lora_dropout[adapter].training:
        raise RuntimeError("Reference model must be in eval mode (LoRA dropout active)")
    tensors = {
        f"{name}.weight": module.base_layer.weight.detach().contiguous(),
        f"{name}.lora_a": module.lora_A[adapter].weight.detach().contiguous(),
        f"{name}.lora_b": module.lora_B[adapter].weight.detach().contiguous(),
    }
    return tensors, float(module.scaling[adapter])


def export_encoder(llm2vec, out_dir: Path, causal: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = llm2vec.model
    named = dict(model.named_modules())
    layers = decoder_layers(llm2vec)
    embed = next(m for n, m in named.items() if n.endswith("embed_tokens"))
    norm = next(m for n, m in named.items() if n.endswith("norm") and ".layers." not in n)
    config = model.config

    embed.weight.detach().contiguous().view(torch.uint16).numpy().tofile(out_dir / "embed_tokens.bin")
    save_file({"weight": norm.weight.detach().contiguous()}, out_dir / "norm.safetensors")

    scaling = {}
    for index, layer in enumerate(layers):
        tensors = {
            "input_layernorm.weight": layer.input_layernorm.weight.detach().contiguous(),
            "post_attention_layernorm.weight": layer.post_attention_layernorm.weight.detach().contiguous(),
        }
        for name in PROJECTIONS:
            parent = layer.self_attn if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else layer.mlp
            parts, scale = _lora_parts(getattr(parent, name), index, name)
            tensors.update(parts)
            if scale is not None:
                if scaling.setdefault(name, scale) != scale:
                    raise NotImplementedError(f"{name}: per-layer LoRA scaling differs")
        save_file(tensors, out_dir / f"layer_{index:02d}.safetensors")

    llm2vec.tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    manifest = {
        "format": "kimodo-turbo/llm2vec-stream/1",
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": len(layers),
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
            "rope_theta": _rope_theta(config),
            "rms_norm_eps": config.rms_norm_eps,
        },
        "attention": "causal" if causal else "bidirectional",
        "template": _template(llm2vec),
        "pooling_mode": llm2vec.pooling_mode,
        "max_length": llm2vec.max_length,
        "doc_max_length": llm2vec.doc_max_length,
        "skip_instruction": llm2vec.skip_instruction,
        "lora_scaling": scaling,
        "reference": {"name_or_path": config._name_or_path, "attn_implementation": config._attn_implementation},
    }
    if manifest["pooling_mode"] != "mean" or not manifest["skip_instruction"]:
        raise NotImplementedError(f"Only mean pooling over the embedded text is ported: {manifest}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"exported {len(layers)} layers to {out_dir}")


def convert(out_dir: Path) -> None:
    torch.set_grad_enabled(False)
    encoder = load_reference_encoder()
    export_encoder(encoder.model, out_dir, causal=attention_is_causal(encoder))
