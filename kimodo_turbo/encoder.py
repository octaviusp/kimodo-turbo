"""LLM2Vec (Llama-3-8B, bf16) on MLX with layer-by-layer weight streaming.

Only one decoder layer (436 MB) is resident at a time, plus the next one being prefetched;
token embeddings are gathered row-by-row from a memory-mapped table. Numerics follow the
reference torch eager path op by op: bf16 projections, unmerged LoRA in the adapter dtype,
HF RMSNorm/RoPE rounding, bf16 SiLU-gated MLP, and bf16 mean pooling. compute_dtype=float32
mirrors kimodo's text_encoder_fp32 option (bf16 weights upcast, fp32 activations).
"""

import json
import threading
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import numpy as np
import torch
from tokenizers import Tokenizer

from .embedding_cache import EmbeddingCache

SEPARATOR = "!@#$%^&*()"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class ShardPrefetcher:
    """Loads layer shards in a background thread so SSD reads overlap GPU compute."""

    def __init__(self, paths: List[Path]):
        self.paths = paths
        self._next: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None

    def _load(self, index: int) -> None:
        weights = mx.load(str(self.paths[index]))
        mx.eval(weights)
        self._next = weights

    def __iter__(self):
        self._load(0)
        for index in range(len(self.paths)):
            weights = self._next
            if index + 1 < len(self.paths):
                self._thread = threading.Thread(target=self._load, args=(index + 1,))
                self._thread.start()
            yield weights
            del weights
            if self._thread is not None:
                self._thread.join()
                self._thread = None


class StreamingLlm2Vec:
    """Kimodo text-encoder contract: texts -> ([B, 1, 4096] fp32 tensor, [1] * B)."""

    def __init__(self, shard_dir: Path, cache: Optional[EmbeddingCache] = None, compute_dtype=mx.bfloat16):
        self.dir = Path(shard_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        cfg = self.manifest["config"]
        self.cfg = cfg
        self.tokenizer = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.embed = np.memmap(self.dir / "embed_tokens.bin", dtype=np.uint16, mode="r").reshape(
            cfg["vocab_size"], cfg["hidden_size"]
        )
        self.layer_paths = [self.dir / f"layer_{i:02d}.safetensors" for i in range(cfg["num_hidden_layers"])]
        self.dtype = compute_dtype
        self.final_norm = mx.load(str(self.dir / "norm.safetensors"))["weight"].astype(compute_dtype)
        self.lora_scaling = self.manifest["lora_scaling"]
        self.cache = cache
        self.causal = self.manifest["attention"] == "causal"

    # Tokenization mirrors LLM2Vec.encode -> _convert_to_str -> prepare_for_tokenization -> tokenize.
    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def tokenize(self, text: str) -> tuple[list[int], int]:
        doc_max = self.manifest["doc_max_length"]
        length = self._count(text)
        while length > doc_max:
            words = text.split()
            text = " ".join(words[: int(len(words) * (doc_max / length))])
            length = self._count(text)
        sentence = SEPARATOR + text
        template = self.manifest["template"]
        if template is not None:
            sentence = template["prefix"] + sentence.strip() + template["suffix"]
        parts = sentence.split(SEPARATOR)
        embed_text = parts[1] if len(parts) > 1 else ""
        max_length = self.manifest["max_length"]
        ids = self.tokenizer.encode("".join(parts), add_special_tokens=True).ids[:max_length]
        embed_len = min(len(self.tokenizer.encode(embed_text, add_special_tokens=False).ids[:max_length]), len(ids))
        return ids, embed_len

    def _rope(self, length: int) -> tuple[mx.array, mx.array]:
        """LlamaRotaryEmbedding.forward in fp32, cast to bf16 (computed with torch for identical libm)."""
        dim = self.cfg["head_dim"]
        inv_freq = 1.0 / (self.cfg["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        freqs = (inv_freq[:, None].float() @ torch.arange(length, dtype=torch.float32)[None, :]).T
        emb = torch.cat((freqs, freqs), dim=-1)
        to_mx = lambda t: mx.array(t.to(torch.bfloat16).view(torch.uint16).numpy()).view(mx.bfloat16)
        if self.dtype == mx.float32:
            return mx.array(emb.cos().numpy()), mx.array(emb.sin().numpy())
        return to_mx(emb.cos()), to_mx(emb.sin())

    def _rms_norm(self, x: mx.array, weight: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        var = mx.mean(x32 * x32, axis=-1, keepdims=True)
        return weight * (x32 * mx.rsqrt(var + self.cfg["rms_norm_eps"])).astype(self.dtype)

    def _proj(self, x: mx.array, w: dict, name: str) -> mx.array:
        out = x @ w[f"{name}.weight"].T
        lora_a = w.get(f"{name}.lora_a")
        if lora_a is None:
            return out
        xa = x.astype(lora_a.dtype)
        lora = ((xa @ lora_a.T) @ w[f"{name}.lora_b"].T) * self.lora_scaling[name]
        return (out + lora).astype(self.dtype)

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    def _layer(self, h: mx.array, w: dict, cos: mx.array, sin: mx.array, mask) -> mx.array:
        cfg = self.cfg
        length = h.shape[1]
        heads, kv_heads, head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        x = self._rms_norm(h, w["input_layernorm.weight"])
        q = self._proj(x, w, "q_proj").reshape(1, length, heads, head_dim).transpose(0, 2, 1, 3)
        k = self._proj(x, w, "k_proj").reshape(1, length, kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = self._proj(x, w, "v_proj").reshape(1, length, kv_heads, head_dim).transpose(0, 2, 1, 3)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(1, length, heads * head_dim)
        h = h + self._proj(attn, w, "o_proj")
        x = self._rms_norm(h, w["post_attention_layernorm.weight"])
        gate = self._proj(x, w, "gate_proj").astype(mx.float32)
        act = (gate * mx.sigmoid(gate)).astype(self.dtype) * self._proj(x, w, "up_proj")
        return h + self._proj(act, w, "down_proj")

    def hidden_states(self, ids: list[int], collect: bool = False):
        rows = np.ascontiguousarray(self.embed[np.asarray(ids)])
        h = mx.array(rows).view(mx.bfloat16)[None].astype(self.dtype)
        cos, sin = self._rope(len(ids))
        mask = "causal" if self.causal else None
        per_layer = []
        for weights in ShardPrefetcher(self.layer_paths):
            if self.dtype != mx.bfloat16:
                weights = {k: v.astype(self.dtype) for k, v in weights.items()}
            h = self._layer(h, weights, cos, sin, mask)
            mx.eval(h)
            if collect:
                per_layer.append(np.array(h.astype(mx.float32))[0])
        return h, per_layer

    def encode_one(self, text: str) -> np.ndarray:
        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        ids, embed_len = self.tokenize(text)
        h, _ = self.hidden_states(ids)
        h = self._rms_norm(h, self.final_norm)
        pooled = mx.mean(h[0, len(ids) - embed_len :].astype(mx.float32), axis=0).astype(self.dtype)
        embedding = np.array(pooled.astype(mx.float32))
        if self.cache is not None:
            self.cache.put(text, embedding)
        return embedding

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = np.stack([self.encode_one(text) for text in texts])
        return torch.from_numpy(embeddings)[:, None], [1] * len(texts)
