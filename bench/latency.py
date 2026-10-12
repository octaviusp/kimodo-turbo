"""End-to-end latency and peak memory of fresh `kimodo_gen` vs `kimodo-turbo gen` processes.

Each run is a new process timed by /usr/bin/time -l (wall time, max RSS, peak physical
footprint, which includes MLX GPU allocations). Variants are interleaved per repetition.
Usage: bench/latency.py [--reps N] [--baseline-reps M] [--duration S]
Writes out/bench/latency.json.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
BIN = TOOL_ROOT / ".venv" / "bin"
PROMPT = "A person jumps over an obstacle and lands softly."


def run(variant: str, duration: float, rep: int) -> dict:
    out = TOOL_ROOT / "out" / "bench" / "latency" / f"{variant}-{rep}"
    env = {**os.environ, "HF_HOME": str(TOOL_ROOT / "out" / "hf"), "HF_HUB_OFFLINE": "1", "TEXT_ENCODER_MODE": "local"}
    common = [PROMPT, "--duration", str(duration), "--seed", "0", "--output", str(out)]
    if variant == "baseline":
        cmd = [str(BIN / "kimodo_gen"), *common, "--model", "Kimodo-SOMA-RP-v1"]
    else:
        cmd = [str(BIN / "kimodo-turbo"), "gen", *common]
        if variant == "turbo-nocache":
            env["KIMODO_TURBO_NO_CACHE"] = "1"
    proc = subprocess.run(["/usr/bin/time", "-l", *cmd], env=env, capture_output=True, text=True, check=True)
    stats = proc.stderr
    return {
        "variant": variant,
        "wall_s": float(re.search(r"([\d.]+) real", stats).group(1)),
        "max_rss_gb": int(re.search(r"(\d+)\s+maximum resident set size", stats).group(1)) / 1e9,
        "peak_footprint_gb": int(re.search(r"(\d+)\s+peak memory footprint", stats).group(1)) / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--baseline-reps", type=int, default=2)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    run("turbo-cached", args.duration, -1)  # populate the embedding cache; not measured
    runs = []
    for rep in range(args.reps):
        variants = ["turbo-nocache", "turbo-cached"] + (["baseline"] if rep < args.baseline_reps else [])
        for variant in variants:
            result = run(variant, args.duration, rep)
            runs.append(result)
            print(json.dumps(result), flush=True)

    summary = {}
    for variant in sorted({r["variant"] for r in runs}):
        rows = [r for r in runs if r["variant"] == variant]
        summary[variant] = {
            "n": len(rows),
            "wall_s_median": statistics.median(r["wall_s"] for r in rows),
            "wall_s_max": max(r["wall_s"] for r in rows),
            "peak_footprint_gb_max": max(r["peak_footprint_gb"] for r in rows),
            "max_rss_gb_max": max(r["max_rss_gb"] for r in rows),
        }
    report = {"prompt": PROMPT, "duration_s": args.duration, "summary": summary, "runs": runs}
    (TOOL_ROOT / "out" / "bench" / "latency.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
