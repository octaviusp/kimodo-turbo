"""Fixed harness, part 0: frozen reference outputs and noise floors (run once; do not edit).

Reference = the validated exact pipeline (MLX fp32 denoiser, 100 DDIM steps, separated CFG,
bf16 streamed LLM2Vec, post-processing on), on the held-out eval prompts x seeds. Also records:
- numeric noise floor: the same pipeline with an fp32-compute text encoder (an equally valid
  implementation), compared pairwise to the reference;
- sample scale: seed 0 vs seed 1 of the same prompt (two different valid motions).
Writes out/autoresearch/reference/{reference.npz, reference.json}.
"""

import os

os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

TOOL_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(TOOL_ROOT / "out" / "hf"))

from edge_metrics import Evaluator, paired  # noqa: E402
from kimodo.tools import seed_everything  # noqa: E402
from kimodo_turbo.cli import SHARDS  # noqa: E402
from kimodo_turbo.encoder import StreamingLlm2Vec  # noqa: E402
from kimodo_turbo.loading import load_turbo_model  # noqa: E402

KEYS = ("posed_joints", "foot_contacts", "root_positions")
OUT = TOOL_ROOT / "out" / "autoresearch" / "reference"


class Table:
    def __init__(self, table):
        self.table = table

    def __call__(self, texts):
        import torch

        return torch.from_numpy(np.stack([self.table[t] for t in texts]))[:, None], [1] * len(texts)


def generate_all(model, items, seconds):
    outputs = {k: [] for k in KEYS}
    for prompt, seed in items:
        seed_everything(seed)
        out = model([prompt], [int(round(seconds * model.fps))], num_denoising_steps=100, num_samples=1,
                    multi_prompt=True, post_processing=True, return_numpy=True)
        for key in KEYS:
            outputs[key].append(np.asarray(out[key][0], dtype=np.float32))
    return {k: np.stack(v) for k, v in outputs.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workload = json.loads((TOOL_ROOT / "bench" / "edge_workload.json").read_text())
    prompts, seeds, seconds = workload["eval"], workload["seeds"], workload["seconds"]
    items = [(p, s) for p in prompts for s in seeds]

    from kimodo.sanitize import sanitize_texts

    sanitized = sanitize_texts(prompts)
    bf16 = StreamingLlm2Vec(SHARDS)
    fp32 = StreamingLlm2Vec(SHARDS, compute_dtype=mx.float32)
    feat_bf16 = {s: bf16.encode_one(s) for s in sanitized}
    feat_fp32 = {s: fp32.encode_one(s) for s in sanitized}

    # Generation is the slow part: checkpoint it so a scoring failure never forces a regeneration.
    raw = OUT / "generated.npz"
    if raw.exists():
        data = np.load(raw)
        reference = {k: data[f"ref_{k}"] for k in KEYS}
        numeric = {k: data[f"num_{k}"] for k in KEYS}
    else:
        model, _ = load_turbo_model(None, Table(feat_bf16))
        reference = generate_all(model, items, seconds)
        model.text_encoder = Table(feat_fp32)
        numeric = generate_all(model, items, seconds)
        np.savez(raw, **{f"ref_{k}": v for k, v in reference.items()}, **{f"num_{k}": v for k, v in numeric.items()})

    evaluator = Evaluator()
    text_feats = np.stack([feat_bf16[s] for s in sanitized for _ in seeds])
    text_emb = evaluator.text_embeddings(text_feats)
    ref_abs, ref_emb = evaluator.absolute(reference["posed_joints"], reference["foot_contacts"], text_emb)
    num_abs, num_emb = evaluator.absolute(numeric["posed_joints"], numeric["foot_contacts"], text_emb)

    # Seed scale: pair seed-0 and seed-1 outputs of the same prompt.
    first = {k: v[0::2] for k, v in reference.items()}
    second = {k: v[1::2] for k, v in reference.items()}
    seed_scale = paired(second, first, ref_emb[1::2], ref_emb[0::2])

    np.savez(OUT / "reference.npz", **reference, text_emb=text_emb, ref_emb=ref_emb)
    report = {
        "items": [{"prompt": p, "seed": s} for p, s in items],
        "reference_absolute": ref_abs,
        "numeric_floor": {"absolute": num_abs, "paired": paired(numeric, reference, num_emb, ref_emb)},
        "seed_scale_paired": seed_scale,
    }
    (OUT / "reference.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
