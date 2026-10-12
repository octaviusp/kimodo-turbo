"""Fixed harness, part 1: run the edge engine on the workload in a fresh process (do not edit).

Times each generation end to end (prompt string -> numpy motion, text encoding included, no
embedding cache) after warm-up, and probes GPU contention with a fixed fp32 GEMM before/after.
Usage: edge_run.py OUT_DIR [--subset eval|calibration]
"""

import os

os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

TOOL_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(TOOL_ROOT / "out" / "hf"))
KEYS = ("posed_joints", "foot_contacts", "root_positions")


def gpu_probe(reps: int = 3) -> float:
    """Median ms of a fixed fp32 GEMM batch; rises when other processes use the GPU."""
    a = mx.random.normal((404, 1024), key=mx.random.key(0))
    b = mx.random.normal((1024, 3072), key=mx.random.key(1))
    mx.eval(a, b)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        mx.eval([a @ b for _ in range(64)])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main() -> None:
    out_dir = Path(sys.argv[1])
    subset = sys.argv[sys.argv.index("--subset") + 1] if "--subset" in sys.argv else "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    workload = json.loads((TOOL_ROOT / "bench" / "edge_workload.json").read_text())
    from kimodo_turbo.edge import build_engine

    probe_before = gpu_probe()
    start = time.perf_counter()
    engine = build_engine()
    load_s = time.perf_counter() - start
    for prompt in workload["warmup"]:
        engine.generate(prompt, workload["seconds"], 0)

    items = [(p, s) for p in workload[subset] for s in workload["seeds"]]
    outputs = {k: [] for k in KEYS}
    latencies = []
    for prompt, seed in items:
        start = time.perf_counter()
        out = engine.generate(prompt, workload["seconds"], seed)
        latencies.append((time.perf_counter() - start) * 1000)
        for key in KEYS:
            outputs[key].append(np.asarray(out[key], dtype=np.float32))
    probe_after = gpu_probe()

    np.savez(out_dir / "outputs.npz", **{k: np.stack(v) for k, v in outputs.items()})
    (out_dir / "timing.json").write_text(json.dumps({
        "engine": getattr(engine, "name", type(engine).__name__),
        "items": [{"prompt": p, "seed": s} for p, s in items],
        "latency_ms": latencies,
        "load_s": load_s,
        "mlx_peak_mb": mx.get_peak_memory() / 2**20,
        "gpu_probe_ms": [probe_before, probe_after],
    }, indent=1))


if __name__ == "__main__":
    main()
