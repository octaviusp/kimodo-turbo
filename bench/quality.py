"""Task-level precision: is turbo's motion as close to an fp32 oracle as the original pipeline's?

REF    = original pipeline: kimodo torch fp32 denoiser + kimodo bf16 LLM2Vec embeddings (oracle npz).
TURBO  = kimodo-turbo: MLX fp32 denoiser + streamed MLX bf16 LLM2Vec.
ORACLE = kimodo torch fp32 denoiser + fp32-compute LLM2Vec embeddings (same weights).
All runs follow kimodo_gen's call (multi_prompt=True, post-processing on) with the same seed.
Writes out/bench/quality.json.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from kimodo.model.common import instantiate_from_dict
from kimodo.model.kimodo_model import Kimodo
from kimodo.tools import seed_everything
from kimodo_turbo.encoder import StreamingLlm2Vec
from kimodo_turbo.loading import load_config, load_turbo_model, resolve_checkpoint

TOOL_ROOT = Path(__file__).resolve().parents[1]
SHARDS = TOOL_ROOT / "out/weights/llm2vec-llama3-8b"
SECONDS, SEED, STEPS = 5.0, 0, 100


class TableEncoder:
    def __init__(self, table: dict):
        self.table = table

    def __call__(self, texts):
        return torch.from_numpy(np.stack([self.table[t] for t in texts]))[:, None], [1] * len(texts)


def generate(model, prompt: str) -> dict:
    seed_everything(SEED)
    return model([prompt], [int(SECONDS * model.fps)], num_denoising_steps=STEPS, num_samples=1,
                 multi_prompt=True, post_processing=True, return_numpy=True)


def compare(a: dict, b: dict) -> dict:
    joint_err = np.linalg.norm(a["posed_joints"] - b["posed_joints"], axis=-1)
    return {
        "mpjpe_cm": float(joint_err.mean() * 100),
        "max_joint_cm": float(joint_err.max() * 100),
        "root_max_cm": float(np.abs(a["root_positions"] - b["root_positions"]).max() * 100),
        "foot_contact_agreement": float((a["foot_contacts"] == b["foot_contacts"]).mean()),
    }


def main() -> None:
    oracle_info = json.loads((TOOL_ROOT / "out/reference/encoder.json").read_text())
    prompts = oracle_info["prompts"]
    ref_table = dict(zip(prompts, np.load(TOOL_ROOT / "out/reference/encoder.npz", allow_pickle=True)["embeddings"]))
    fp32_encoder = StreamingLlm2Vec(SHARDS, compute_dtype=mx.float32)
    oracle_table = {p: fp32_encoder.encode_one(p) for p in prompts}

    cfg = load_config(resolve_checkpoint(None)[1])
    torch_model = Kimodo(instantiate_from_dict(cfg["denoiser"]), TableEncoder(ref_table), cfg["num_base_steps"],
                         device="cpu", cfg_type=cfg["cfg_type"]).eval()
    turbo, _ = load_turbo_model(None, StreamingLlm2Vec(SHARDS))

    results = []
    for prompt in prompts:
        torch_model.text_encoder = TableEncoder(ref_table)
        ref = generate(torch_model, prompt)
        torch_model.text_encoder = TableEncoder(oracle_table)
        oracle = generate(torch_model, prompt)
        start = time.perf_counter()
        fast = generate(turbo, prompt)
        row = {
            "prompt": prompt,
            "ref_vs_oracle": compare(ref, oracle),
            "turbo_vs_oracle": compare(fast, oracle),
            "turbo_vs_ref": compare(fast, ref),
            "turbo_seconds": time.perf_counter() - start,
        }
        results.append(row)
        print(json.dumps(row))
        np.savez(TOOL_ROOT / f"out/bench/motions/{len(results) - 1}.npz", ref=ref["posed_joints"],
                 oracle=oracle["posed_joints"], turbo=fast["posed_joints"])

    summary = {
        key: {metric: float(np.mean([r[key][metric] for r in results])) for metric in results[0][key]}
        for key in ("ref_vs_oracle", "turbo_vs_oracle", "turbo_vs_ref")
    }
    summary["max_joint_cm_worst"] = {k: float(max(r[k]["max_joint_cm"] for r in results)) for k in ("ref_vs_oracle", "turbo_vs_oracle", "turbo_vs_ref")}
    (TOOL_ROOT / "out/bench/quality.json").write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    (TOOL_ROOT / "out/bench/motions").mkdir(parents=True, exist_ok=True)
    main()
