"""Streamed MLX LLM2Vec vs kimodo's torch bf16 encoder (oracle from scripts/reference_encoder.py)."""

import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from kimodo_turbo.encoder import StreamingLlm2Vec

TOOL_ROOT = Path(__file__).resolve().parents[1]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ref = np.load(TOOL_ROOT / "out/reference/encoder.npz", allow_pickle=True)
    prompts = json.loads((TOOL_ROOT / "out/reference/encoder.json").read_text())["prompts"]
    encoder = StreamingLlm2Vec(TOOL_ROOT / "out/weights/llm2vec-llama3-8b")

    failures = 0
    for i, prompt in enumerate(prompts):
        ids, embed_len = encoder.tokenize(prompt)
        if ids != ref["input_ids"][i].tolist() or embed_len != int(ref["embed_masks"][i].sum()):
            print(f"TOKEN MISMATCH [{i}] {ids} vs {ref['input_ids'][i].tolist()}")
            failures += 1

    # Layer-by-layer agreement on prompt 0, for both attention hypotheses.
    ids, _ = encoder.tokenize(prompts[0])
    oracle_hidden = ref["hidden_prompt0"]
    for causal in (False, True):
        encoder.causal = causal
        _, layers = encoder.hidden_states(ids, collect=True)
        rel = [np.abs(h - o).max() / np.abs(o).max() for h, o in zip(layers, oracle_hidden)]
        print(f"causal={causal}: layer max_rel first={rel[0]:.2e} mid={rel[15]:.2e} last={rel[-1]:.2e}")
    encoder.causal = encoder.manifest["attention"] == "causal"

    mx.reset_peak_memory()
    cosines, rels = [], []
    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        emb = encoder.encode_one(prompt)
        seconds = time.perf_counter() - start
        oracle = ref["embeddings"][i]
        cosines.append(cosine(emb, oracle))
        rels.append(np.abs(emb - oracle).max() / np.abs(oracle).max())
        print(f"[{i}] {seconds:.2f}s cos={cosines[-1]:.6f} max_rel={rels[-1]:.2e}")
    print(f"min cosine={min(cosines):.6f} max rel={max(rels):.2e}")
    print(f"mlx peak={mx.get_peak_memory() / 1e9:.2f} GB  process max RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
    failures += min(cosines) < 0.9999
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
