"""Folder-local paths shared by installation, inference and the UI."""

from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
OUT = TOOL_ROOT / "out"
WEIGHTS = OUT / "weights"
SHARDS = WEIGHTS / "llm2vec-llama3-8b"
EMBEDDING_CACHE = OUT / "cache" / "text-embeddings"
UI = TOOL_ROOT / "ui"
