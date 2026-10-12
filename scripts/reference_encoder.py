"""Run kimodo's own LLM2Vec encoder (bf16, CPU) as the oracle for the streamed MLX encoder.

Records the loaded structure (adapters, dtypes, attention mask actually passed), token IDs and
pooling masks, per-layer hidden states for the first prompt, and pooled embeddings for every
prompt, to out/reference/encoder.npz + encoder.json. With --export DIR it also writes the
streamable shards from the same in-memory model (kimodo_turbo.convert).
Run with HF_HOME=out/hf HF_HUB_OFFLINE=1.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from kimodo.sanitize import sanitize_texts
from kimodo_turbo.convert import attention_is_causal, decoder_layers, export_encoder, load_reference_encoder

TOOL_ROOT = Path(__file__).resolve().parents[1]


def describe(llm2vec) -> dict:
    model = llm2vec.model
    lora = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and len(module.lora_A):
            adapter = next(iter(module.lora_A))
            lora[name] = {
                "adapters": list(module.lora_A.keys()),
                "active": list(module.active_adapters),
                "merged": list(getattr(module, "merged_adapters", [])),
                "scaling": {k: float(v) for k, v in module.scaling.items()},
                "lora_dtype": str(module.lora_A[adapter].weight.dtype),
                "base_dtype": str(module.base_layer.weight.dtype),
            }
    return {
        "model_class": type(model).__name__,
        "name_or_path": model.config._name_or_path,
        "attn_implementation": model.config._attn_implementation,
        "pooling_mode": llm2vec.pooling_mode,
        "max_length": llm2vec.max_length,
        "doc_max_length": llm2vec.doc_max_length,
        "skip_instruction": llm2vec.skip_instruction,
        "padding_side": llm2vec.tokenizer.padding_side,
        "num_lora_modules": len(lora),
        "lora_example": next(iter(lora.items())) if lora else None,
        "lora_modules_sample": sorted(lora)[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(TOOL_ROOT / "bench" / "prompts.json"))
    parser.add_argument("--out", default=str(TOOL_ROOT / "out" / "reference"))
    parser.add_argument("--export", default=None, help="also write MLX shards to this directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    prompts = sanitize_texts(raw_prompts)  # Kimodo sanitizes before calling the encoder.

    start = time.perf_counter()
    encoder = load_reference_encoder()
    load_seconds = time.perf_counter() - start
    llm2vec = encoder.model
    info = describe(llm2vec)
    info["load_seconds"] = load_seconds
    print(json.dumps(info, indent=2, default=str))

    # Capture what the first attention layer actually receives and every layer's output.
    attn_calls, hidden = [], []
    layers = decoder_layers(llm2vec)
    attn = [m for n, m in layers[0].named_modules() if n == "self_attn"][0]

    def attn_hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        attn_calls.append(None if mask is None else {"shape": list(mask.shape), "dtype": str(mask.dtype),
                                                     "upper_triangle_masked": bool((mask[0, 0].float().triu(1) < 0).any())})

    def layer_hook(module, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        hidden.append(tensor.detach().float().numpy()[0])

    handles = [attn.register_forward_pre_hook(attn_hook, with_kwargs=True)]
    handles += [layer.register_forward_hook(layer_hook) for layer in layers]

    embeddings, input_ids, embed_masks, seconds = [], [], [], []
    for index, prompt in enumerate(prompts):
        if index == 1:
            for handle in handles:
                handle.remove()
        sentence = llm2vec.prepare_for_tokenization(llm2vec._convert_to_str("", prompt))
        features = llm2vec.tokenize([sentence])
        input_ids.append(features["input_ids"][0].numpy())
        embed_masks.append(features["embed_mask"][0].numpy())
        start = time.perf_counter()
        emb, lengths = encoder([prompt])  # the exact call Kimodo makes
        seconds.append(time.perf_counter() - start)
        assert lengths == [1]
        embeddings.append(emb[0, 0].numpy())
        print(f"[{index}] {seconds[-1]:.2f}s tokens={len(input_ids[-1])} {prompt!r}")

    info.update(
        prompts=prompts,
        attention_mask_first_layer=attn_calls[0] if attn_calls else "not captured",
        encode_seconds=seconds,
        peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
    )
    (out_dir / "encoder.json").write_text(json.dumps(info, indent=2, default=str))
    np.savez(
        out_dir / "encoder.npz",
        embeddings=np.stack(embeddings),
        input_ids=np.array(input_ids, dtype=object),
        embed_masks=np.array(embed_masks, dtype=object),
        hidden_prompt0=np.stack(hidden),
    )
    print("attention mask seen by layer 0:", info["attention_mask_first_layer"])

    if args.export:
        export_encoder(llm2vec, Path(args.export), causal=attention_is_causal(encoder))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
