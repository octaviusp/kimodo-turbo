"""Fixed harness, part 2: evaluate the current kimodo_turbo/edge.py (do not edit during the loop).

Usage: edge_eval.py RUN_NAME
Runs bench/edge_run.py in a fresh process under /usr/bin/time -l, then scores its outputs
against the frozen reference (bench/edge_reference.py) and prints an autoresearch summary:

  latency_ms:   median end-to-end generation time (5 s motion, text encoding included)
  peak_mem_mb:  peak physical footprint of the engine process (includes MLX GPU memory)
  gate:         A (near-lossless), B (ultra-minimal loss) or FAIL, from preregistered thresholds
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
REF = TOOL_ROOT / "out" / "autoresearch" / "reference"
RUNS = TOOL_ROOT / "out" / "autoresearch" / "runs"

# Preregistered gates (set from the reference noise floors before any candidate was run).
GATES = {
    "A": {"mpjpe_cm": 1.0, "tmr_pair_cos": 0.995, "t2m_drop": 0.003, "R3_drop": 0.03, "skate_rel": 0.05, "contacts": 0.98},
    "B": {"mpjpe_cm": 3.0, "tmr_pair_cos": 0.98, "t2m_drop": 0.008, "R3_drop": 0.06, "skate_rel": 0.10, "contacts": 0.95},
}
SKATE_KEY = "foot_skate_from_height"


def gate(paired: dict, cand: dict, ref: dict) -> str:
    skate_c = next((v for k, v in cand.items() if SKATE_KEY in k.lower()), None)
    skate_r = next((v for k, v in ref.items() if SKATE_KEY in k.lower()), None)
    for tier in ("A", "B"):
        g = GATES[tier]
        ok = (
            paired["mpjpe_cm"] <= g["mpjpe_cm"]
            and paired["tmr_pair_cos"] >= g["tmr_pair_cos"]
            and ref["t2m_cos"] - cand["t2m_cos"] <= g["t2m_drop"]
            and ref["R3"] - cand["R3"] <= g["R3_drop"] * 100
            and paired["contacts"] >= g["contacts"]
            and (skate_c is None or skate_c <= skate_r * (1 + g["skate_rel"]) + 1e-6)
        )
        if ok:
            return tier
    return "FAIL"


def main() -> None:
    name = sys.argv[1]
    run_dir = RUNS / name
    proc = subprocess.run(
        ["/usr/bin/time", "-l", str(TOOL_ROOT / ".venv/bin/python"), str(TOOL_ROOT / "bench/edge_run.py"), str(run_dir)],
        capture_output=True, text=True,
    )
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "run.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print("status: crash")
        print(proc.stderr[-3000:])
        sys.exit(1)
    peak = int(re.search(r"(\d+)\s+peak memory footprint", proc.stderr).group(1)) / 2**20

    sys.path.insert(0, str(TOOL_ROOT / "bench"))
    from edge_metrics import Evaluator, paired

    timing = json.loads((run_dir / "timing.json").read_text())
    cand = dict(np.load(run_dir / "outputs.npz"))
    ref_npz = np.load(REF / "reference.npz")
    ref_json = json.loads((REF / "reference.json").read_text())
    reference = {k: ref_npz[k] for k in ("posed_joints", "foot_contacts", "root_positions")}

    evaluator = Evaluator()
    cand_abs, cand_emb = evaluator.absolute(cand["posed_joints"], cand["foot_contacts"], ref_npz["text_emb"])
    pair = paired(cand, reference, cand_emb, ref_npz["ref_emb"])
    ref_abs = ref_json["reference_absolute"]
    tier = gate(pair, cand_abs, ref_abs)

    lat = np.array(timing["latency_ms"])
    summary = {
        "engine": timing["engine"],
        "latency_ms": float(np.median(lat)),
        "latency_p90_ms": float(np.percentile(lat, 90)),
        "load_s": timing["load_s"],
        "peak_mem_mb": peak,
        "mlx_peak_mb": timing["mlx_peak_mb"],
        "gpu_probe_ms": timing["gpu_probe_ms"],
        **pair,
        **{f"{k}": v for k, v in cand_abs.items()},
        **{f"ref_{k}": v for k, v in ref_abs.items() if k in ("t2m_cos", "R1", "R3")},
        "gate": tier,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("---")
    for key in ("engine", "latency_ms", "latency_p90_ms", "peak_mem_mb", "mlx_peak_mb", "load_s", "gpu_probe_ms",
                "mpjpe_cm", "mpjpe_p90_cm", "max_joint_cm", "contacts", "tmr_pair_cos", "tmr_pair_cos_min",
                "t2m_cos", "ref_t2m_cos", "R3", "ref_R3", "gate"):
        value = summary[key]
        print(f"{key + ':':18s}{value:.4f}" if isinstance(value, float) else f"{key + ':':18s}{value}")


if __name__ == "__main__":
    main()
