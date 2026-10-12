"""Auto-rig a GLB and render front/side views: vertices colored by dominant bone + joint markers.

Usage: tests/autorig_debug.py model.glb [out_dir]   (default out_dir: out/gym/debug)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from kimodo_turbo.autorig import JOINT_NAMES, PARENT, autorig
from kimodo_turbo.glb import Glb

TOOL_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else TOOL_ROOT / "out/gym/debug"
    out.mkdir(parents=True, exist_ok=True)
    rigged, report = autorig(src.read_bytes())
    (out / f"{src.stem}.rigged.glb").write_bytes(rigged)
    print(json.dumps({k: v for k, v in report.items() if k != "joints"}, indent=1))

    glb = Glb.parse(rigged)
    prim = glb.gltf["meshes"][0]["primitives"][0]
    pos = glb.read_accessor(prim["attributes"]["POSITION"])
    ids = glb.read_accessor(prim["attributes"]["JOINTS_0"])
    w = glb.read_accessor(prim["attributes"]["WEIGHTS_0"])
    dominant = ids[np.arange(len(ids)), w.argmax(1)]
    palette = (np.random.default_rng(3).random((len(JOINT_NAMES), 3)) * 200 + 40).astype(np.uint8)
    joints = {n: np.array(p) for n, p in report["joints"].items()}
    up, fwd = np.array(report["up"]), np.array(report["forward"])
    left = np.cross(up, fwd)

    for view, axis in (("front", left), ("side", fwd)):
        coords = np.stack([pos @ axis, pos @ up], 1)
        jc = {n: np.array([p @ axis, p @ up]) for n, p in joints.items()}
        lo, hi = coords.min(0), coords.max(0)
        scale = 760 / (hi - lo).max()
        to_px = lambda c: ((c[0] - lo[0]) * scale + 20 + (760 - (hi[0] - lo[0]) * scale) / 2, 780 - (c[1] - lo[1]) * scale)
        img = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(img)
        for c, d in zip(coords, dominant):
            x, y = to_px(c)
            draw.point((x, y), tuple(int(v) for v in palette[d]))
        for name in JOINT_NAMES:
            if PARENT[name]:
                draw.line([to_px(jc[PARENT[name]]), to_px(jc[name])], fill="black", width=2)
            x, y = to_px(jc[name])
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="red")
        img.save(out / f"{src.stem}.{view}.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
