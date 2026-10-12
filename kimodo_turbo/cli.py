"""kimodo-turbo: kimodo_gen on MLX in a few GB of RAM.

  kimodo-turbo gen "A person walks forward." [any kimodo_gen option]
  kimodo-turbo ui [--port N]    # local motion preview at http://127.0.0.1:8765
  kimodo-turbo convert          # one-time: build the streamed text-encoder shards

`gen` runs kimodo's own generate script (same arguments, constraints, NPZ/BVH/CSV outputs)
with the model loader swapped for TurboKimodo.
"""

import os
import sys
import time

from .paths import EMBEDDING_CACHE, OUT, SHARDS

USAGE = __doc__


def _turbo_load_model(modelname=None, device=None, default_family="Kimodo", return_resolved_name=False, **_):
    from .embedding_cache import EmbeddingCache
    from .encoder import StreamingLlm2Vec
    from .loading import load_turbo_model

    if not (SHARDS / "manifest.json").exists():
        raise FileNotFoundError(f"Text-encoder shards not found in {SHARDS}. Run scripts/setup.zsh first.")
    cache = None if os.environ.get("KIMODO_TURBO_NO_CACHE") else EmbeddingCache(EMBEDDING_CACHE, SHARDS, "bfloat16")
    model, resolved = load_turbo_model(modelname, StreamingLlm2Vec(SHARDS, cache=cache))
    return (model, resolved) if return_resolved_name else model


def generate(argv: list[str]) -> None:
    from kimodo.scripts import generate as kimodo_generate

    from .loading import DEFAULT_MODEL

    kimodo_generate.load_model = _turbo_load_model
    kimodo_generate.DEFAULT_MODEL = DEFAULT_MODEL
    if not any(arg == "--output" or arg.startswith("--output=") for arg in argv):
        argv = [*argv, "--output", str(OUT / "motions" / "motion")]
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
    elif command == "ui":
        import argparse

        from .ui_server import serve

        parser = argparse.ArgumentParser(prog="kimodo-turbo ui", description="Local motion generation and preview.")
        parser.add_argument("--port", type=int, default=8765)
        options = parser.parse_args(rest)
        if not 1 <= options.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        serve(options.port)
    elif command == "convert":
        from .convert import convert

        convert(SHARDS)
    else:
        sys.exit(f"Unknown command {command!r}\n{USAGE}")


if __name__ == "__main__":
    main()
