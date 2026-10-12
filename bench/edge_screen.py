"""Fast screening of solver/precision configs on the CALIBRATION prompts (never the eval set).

Teacher = Kimodo's exact sampler (100-step DDIM, fp32) from fixed noise; candidates run from the
same noise. Metrics are in joint space after FK (no post-processing): MPJPE and root error vs the
teacher. Batched across prompts, so one process screens many configs quickly.

Usage: edge_screen.py "ddim:20:uniform:float32" "dpmpp2m:20:logsnr:bfloat16" ...
       (solver:steps:spacing:dtype[:rootfp32])
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
import torch  # noqa: E402

TOOL_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(TOOL_ROOT / "out" / "hf"))

from kimodo.sanitize import sanitize_texts  # noqa: E402
from kimodo_turbo.cli import SHARDS  # noqa: E402
from kimodo_turbo.edge import DTYPES, EdgeConfig, EdgeKimodo, LowPrecisionStacks  # noqa: E402
from kimodo_turbo.encoder import StreamingLlm2Vec  # noqa: E402
from kimodo_turbo.loading import load_turbo_model  # noqa: E402

CALIB = TOOL_ROOT / "out" / "autoresearch" / "calib"
FRAMES, SEEDS, BATCH = 150, 1, 24


def parse(spec: str) -> EdgeConfig:
    parts = spec.split(":")
    return EdgeConfig(steps=int(parts[1]), solver=parts[0], spacing=parts[2], dtype=parts[3],
                      root_fp32=len(parts) > 4 and parts[4] == "rootfp32")


def features() -> np.ndarray:
    path = CALIB / "feats.npz"
    if path.exists():
        return np.load(path)["feats"]
    prompts = sanitize_texts(json.loads((TOOL_ROOT / "bench" / "edge_workload.json").read_text())["calibration"])
    encoder = StreamingLlm2Vec(SHARDS)
    feats = np.stack([encoder.encode_one(p) for p in prompts])
    CALIB.mkdir(parents=True, exist_ok=True)
    np.savez(path, feats=feats, prompts=np.array(prompts))
    return feats


def run(model, feats, noise, config: EdgeConfig) -> tuple[np.ndarray, float]:
    outs, start = [], time.perf_counter()
    for b in range(0, len(noise), BATCH):
        text = torch.from_numpy(np.repeat(feats, SEEDS, axis=0)[b:b + BATCH])[:, None]
        n = noise[b:b + BATCH]
        cond = model.conditioning(text, torch.ones(len(n), FRAMES, dtype=torch.bool), torch.zeros(len(n)))
        x = model.solve(mx.array(n), cond, config.steps, config)
        outs.append(np.array(x.astype(mx.float32)))
    return np.concatenate(outs), time.perf_counter() - start


def joints(model, x: np.ndarray) -> np.ndarray:
    out = model.motion_rep.inverse(torch.from_numpy(x), is_normalized=True, return_numpy=True)
    return out["posed_joints"]


def main() -> None:
    feats = features()
    model, _ = load_turbo_model(None, None)
    model.__class__ = EdgeKimodo
    fp32_params = {name: dict(getattr(model.mlx_denoiser, name).p) for name in ("root", "body")}
    noise = np.random.default_rng(1234).standard_normal((len(feats) * SEEDS, FRAMES, model.motion_rep.motion_rep_dim)).astype(np.float32)

    teacher_path = CALIB / "teacher.npz"
    if teacher_path.exists():
        teacher = np.load(teacher_path)["x"]
    else:
        teacher, secs = run(model, feats, noise, EdgeConfig())
        np.savez(teacher_path, x=teacher)
        print(f"teacher: {secs:.1f}s")
    teacher_j = joints(model, teacher)

    results = []
    for spec in sys.argv[1:]:
        config = parse(spec)
        for name in ("root", "body"):  # restore fp32 before applying a precision policy
            stack = getattr(model.mlx_denoiser, name)
            stack.p, stack.dtype = dict(fp32_params[name]), mx.float32
        if config.dtype != "float32":
            LowPrecisionStacks(model.mlx_denoiser, DTYPES[config.dtype], config.root_fp32)
        x, secs = run(model, feats, noise, config)
        cand_j = joints(model, x)
        err = np.linalg.norm(cand_j - teacher_j, axis=-1)
        row = {
            "spec": spec,
            "mpjpe_cm": float(err.mean() * 100),
            "p90_sample_cm": float(np.percentile(err.mean(axis=(1, 2)), 90) * 100),
            "max_cm": float(err.max() * 100),
            "root_cm": float(np.linalg.norm(cand_j[:, :, 0] - teacher_j[:, :, 0], axis=-1).mean() * 100),
            "feat_rmse": float(np.sqrt(((x - teacher) ** 2).mean())),
            "batch_s": secs,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    log = CALIB / "screen.jsonl"
    with log.open("a") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
