"""kimodo-turbo: kimodo_gen on MLX in a few GB of RAM.

  kimodo-turbo gen "A person walks forward." [any kimodo_gen option]
  kimodo-turbo gym [--port N]   # local web gym: pick a .glb, type a prompt, watch it animate
  kimodo-turbo convert          # one-time: build the streamed text-encoder shards

`gen` runs kimodo's own generate script (same arguments, constraints, NPZ/BVH/CSV outputs)
with the model loader swapped for TurboKimodo.
"""

import os
import sys
import time
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
OUT = TOOL_ROOT / "out"
SHARDS = OUT / "weights" / "llm2vec-llama3-8b"
EMBEDDING_CACHE = OUT / "cache" / "text-embeddings"
USAGE = __doc__


def _turbo_load_model(modelname=None, device=None, default_family="Kimodo", return_resolved_name=False, **_):
    from .embedding_cache import EmbeddingCache
    from .encoder import StreamingLlm2Vec
    from .loading import load_turbo_model

    if not (SHARDS / "manifest.json").exists():
        sys.exit(f"Text-encoder shards not found in {SHARDS}. Run: kimodo-turbo convert")
    cache = None if os.environ.get("KIMODO_TURBO_NO_CACHE") else EmbeddingCache(EMBEDDING_CACHE, SHARDS, "bfloat16")
    model, resolved = load_turbo_model(modelname, StreamingLlm2Vec(SHARDS, cache=cache))
    return (model, resolved) if return_resolved_name else model


def generate(argv: list[str]) -> None:
    from kimodo.scripts import generate as kimodo_generate

    from .loading import DEFAULT_MODEL

    kimodo_generate.load_model = _turbo_load_model
    kimodo_generate.DEFAULT_MODEL = DEFAULT_MODEL
    sys.argv = ["kimodo-turbo gen", *argv]
    start = time.perf_counter()
    kimodo_generate.main()
    print(f"kimodo-turbo: done in {time.perf_counter() - start:.1f}s")


def main() -> None:
    os.environ.setdefault("HF_HOME", str(OUT / "hf"))
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return
    command, rest = args[0], args[1:]
    if command == "gen":
        generate(rest)
    elif command == "gym":
        from .gym_server import serve

        port = int(rest[rest.index("--port") + 1]) if "--port" in rest else 8765
        serve(port)
    elif command == "convert":
        from .convert import convert

        convert(SHARDS)
    else:
        sys.exit(f"Unknown command {command!r}\n{USAGE}")


if __name__ == "__main__":
    main()
