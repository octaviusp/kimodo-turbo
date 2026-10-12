#!/usr/bin/env python3
"""Check native skeleton/skin invariance to units, translation, and Z-up GLBs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import struct
import subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("autorig_validation", ROOT / "scripts/validate_autorig.py")
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def transformed_glb(source: pathlib.Path, target: pathlib.Path, transform: np.ndarray) -> None:
    original = source.read_bytes()
    data, binary = None, b""
    cursor = 12
    while cursor < len(original):
        length, kind = struct.unpack_from("<II", original, cursor)
        chunk = original[cursor + 8:cursor + 8 + length]
        if kind == 0x4E4F534A:
            data = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        cursor += 8 + length
    if data is None:
        raise ValueError("Missing GLB JSON chunk")
    scene = data["scenes"][data.get("scene", 0)]
    wrapper = {"name": "ValidationTransform", "matrix": transform.T.ravel().tolist(), "children": scene["nodes"]}
    scene["nodes"] = [len(data["nodes"])]
    data["nodes"].append(wrapper)
    encoded = json.dumps(data).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    size = 12 + 8 + len(encoded) + 8 + len(binary)
    target.write_bytes(struct.pack("<III", 0x46546C67, 2, size)
                       + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                       + struct.pack("<II", len(binary), 0x004E4942) + binary)


def native(binary: pathlib.Path, source: pathlib.Path, output: pathlib.Path, up: str, forward: str) -> dict:
    report = output.with_suffix(".json")
    command = [str(binary.resolve()), str(source), str(output), "--report", str(report), "--force",
               "--up", up, "--forward", forward]
    subprocess.run(command, check=True, capture_output=True, timeout=90)
    return json.loads(report.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "out/bin/kimodo-autorig")
    parser.add_argument("--manifest", type=pathlib.Path, default=ROOT / "out/autorig-validation/corpus.json")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "out/autorig-validation/qa-invariance")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_binary = args.binary.resolve()
    snapshot = args.out / f"native-cli-{ORACLE.digest(source_binary)[:12]}"
    if not snapshot.exists():
        shutil.copyfile(source_binary, snapshot)
        snapshot.chmod(source_binary.stat().st_mode)
    args.binary = snapshot
    corpus = json.loads(args.manifest.read_text())["cases"]
    sources = [case for case in corpus if case["id"] in ("tunic", "musk", "dwarf-blacksmith")]
    cases = []
    for case in sources:
        name = case["id"]
        source = pathlib.Path(case["path"])
        forward = case["forward_axis"].lower().removeprefix("+")
        base = native(args.binary, source, args.out / f"{name}-base.glb", "y", forward)
        base_points = np.array([joint["position"] for joint in base["joints"]])
        height = base["height"]
        for variation in ("translated", "small-units", "large-units", "z-up"):
            transform = np.eye(4)
            up, facing, scale = "y", forward, 1.0
            if variation == "translated":
                transform[:3, 3] = [1.7, -0.9, 4.5]
            elif variation == "small-units":
                scale = 0.001
                transform[:3, :3] *= scale
            elif variation == "large-units":
                scale = 100
                transform[:3, :3] *= scale
            else:
                # Rotating around X sends Y-up to Z-up. Keep facing expressible by the CLI.
                if forward == "z":
                    transform[:3, :3] = ORACLE.rotation(np.array([0, 0, 1]), 90) @ ORACLE.rotation(np.array([1, 0, 0]), 90)
                    facing = "x"
                else:
                    transform[:3, :3] = ORACLE.rotation(np.array([1, 0, 0]), 90)
                    facing = forward
                up = "z"
            fixture = args.out / f"{name}-{variation}-source.glb"
            output = args.out / f"{name}-{variation}-rigged.glb"
            transformed_glb(source, fixture, transform)
            actual = native(args.binary, fixture, output, up, facing)
            actual_points = np.array([joint["position"] for joint in actual["joints"]])
            expected = base_points @ transform[:3, :3].T + transform[:3, 3]
            errors = np.linalg.norm(actual_points - expected, axis=1) / (height * scale)
            oracle_report, _ = ORACLE.validate(ORACLE.Glb(output))
            proof = ORACLE.geometry_proof(ORACLE.Glb(fixture), ORACLE.Glb(output), bool(actual.get("covered_legs")))
            geometry_error = proof["max_face_corner_error_height"]
            record = {"id": f"{name}-{variation}", "joint_max_error_height": float(errors.max()),
                      "joint_p95_error_height": float(np.percentile(errors, 95)), "geometry_max_error_height": geometry_error,
                      "bind_correctness_pass": oracle_report["correctness_pass"],
                      "geometry_proof_pass": proof["pass"],
                      "pass": bool(errors.max() < 2e-4 and geometry_error < 1e-5 and proof["pass"] and oracle_report["correctness_pass"])}
            cases.append(record)
            print(json.dumps(record), flush=True)
    report = {"schema_version": 1, "binary_sha256": ORACLE.digest(args.binary), "cases": cases,
              "passed": sum(c["pass"] for c in cases), "total": len(cases),
              "joint_tolerance_character_heights": 2e-4, "geometry_tolerance_character_heights": 1e-5}
    ORACLE.json_write(args.out / "summary.json", report)
    return 0 if report["passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
