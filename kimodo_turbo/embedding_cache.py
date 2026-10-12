"""Exact on-disk cache of pooled text embeddings.

Keyed by the encoder shard identity (manifest + weight file sizes/mtimes), the compute dtype and
the exact prompt string, so a repeated prompt skips the 15 GB weight stream and returns identical bytes.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


class EmbeddingCache:
    def __init__(self, root: Path, shard_dir: Path, compute_dtype: str):
        identity = hashlib.sha256((Path(shard_dir) / "manifest.json").read_bytes() + compute_dtype.encode())
        for path in sorted(Path(shard_dir).iterdir()):
            stat = path.stat()
            identity.update(json.dumps([path.name, stat.st_size, stat.st_mtime_ns]).encode())
        self.dir = Path(root) / identity.hexdigest()[:16]
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str) -> Path:
        return self.dir / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.npy"

    def get(self, text: str) -> Optional[np.ndarray]:
        path = self._path(text)
        return np.load(path) if path.exists() else None

    def put(self, text: str, embedding: np.ndarray) -> None:
        path = self._path(text)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, embedding.astype(np.float32))
        tmp.replace(path)
