"""Local web gym: pick a .glb, describe a motion, watch Kimodo animate it.

The model stays loaded between requests (one generation at a time on the GPU). Endpoints:
  GET  /api/status          readiness of the motion model
  GET  /api/default-model   the bundled character, auto-rigged if it has no skeleton
  POST /api/rig             body: .glb bytes -> rigged .glb (X-Rig: auto|existing, X-Rig-Report: json)
  POST /api/generate        {segments: [{prompt, duration}], seed, steps, postprocess} -> motion json
Everything else is served from gym/.
"""

import base64
import hashlib
import json
import re
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .autorig import autorig
from .glb import Glb

TOOL_ROOT = Path(__file__).resolve().parents[1]
STATIC = TOOL_ROOT / "gym"
DEFAULT_MODEL = STATIC / "assets" / "personaje-realista.glb"
OUT = TOOL_ROOT / "out" / "gym"
MAX_UPLOAD = 200 * 1024 * 1024
MAX_SECONDS_PER_SEGMENT = 10.0
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".glb": "model/gltf-binary",
                 ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json"}


class MotionService:
    """Owns the TurboKimodo model; loads it in the background and serializes generations."""

    def __init__(self):
        self.model = None
        self.error = None
        self.lock = threading.Lock()
        self.skeleton = None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            from .cli import _turbo_load_model

            model = _turbo_load_model()
            skel = model.output_skeleton
            self.skeleton = {
                "joint_names": list(skel.bone_order_names),
                "parents": [int(p) for p in np.asarray(skel.joint_parents)],
                "rest": skel.neutral_joints.numpy().astype(np.float32).ravel().tolist(),
            }
            self.model = model
        except Exception as error:  # surfaced to the UI via /api/status
            self.error = f"{type(error).__name__}: {error}"
            traceback.print_exc()

    def status(self) -> dict:
        return {"ready": self.model is not None, "error": self.error}

    def generate(self, request: dict) -> dict:
        if self.model is None:
            raise RuntimeError(self.error or "El modelo de movimiento todavía está cargando")
        segments = request.get("segments") or []
        prompts = [str(s.get("prompt", "")).strip() for s in segments]
        if not prompts or not all(prompts):
            raise ValueError("Escribí al menos una descripción de movimiento")
        durations = [float(s.get("duration", 5.0)) for s in segments]
        if not all(0.5 <= d <= MAX_SECONDS_PER_SEGMENT for d in durations):
            raise ValueError(f"Cada segmento debe durar entre 0.5 y {MAX_SECONDS_PER_SEGMENT:g} segundos")
        steps = int(request.get("steps", 100))
        seed = int(request.get("seed", 0))
        from kimodo.tools import seed_everything

        with self.lock:
            start = time.perf_counter()
            seed_everything(seed)
            out = self.model(
                prompts, [int(round(d * self.model.fps)) for d in durations], num_denoising_steps=steps,
                num_samples=1, multi_prompt=True, post_processing=bool(request.get("postprocess", True)),
                return_numpy=True, progress_bar=lambda x: x,
            )
            seconds = time.perf_counter() - start
        # multi_prompt outputs keep the sample axis: [num_samples, T, joints, ...]
        positions = out["posed_joints"][0].astype(np.float32)
        rotations = _quaternions(out["global_rot_mats"][0]).astype(np.float32)
        OUT.joinpath("motions").mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", prompts[0].lower()).strip("-")[:40] or "motion"
        np.savez(OUT / "motions" / f"{time.strftime('%Y-%m-%d-%H%M%S')}-{slug}.npz", **out, prompts=np.array(prompts))
        return {
            "fps": float(self.model.fps),
            "frames": int(positions.shape[0]),
            **self.skeleton,
            "positions": base64.b64encode(positions.tobytes()).decode(),
            "rotations": base64.b64encode(rotations.tobytes()).decode(),
            "seconds": seconds,
            "segments": [{"prompt": p, "duration": d} for p, d in zip(prompts, durations)],
            "seed": seed,
            "steps": steps,
        }


def _quaternions(matrices: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] -> unit quaternions [..., 4] in (x, y, z, w) order."""
    m = matrices.reshape(-1, 3, 3).astype(np.float64)
    q = np.empty((len(m), 4))
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i, r in enumerate(m):
        if trace[i] > 0:
            s = 0.5 / np.sqrt(trace[i] + 1.0)
            q[i] = [(r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s, (r[1, 0] - r[0, 1]) * s, 0.25 / s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*matrices.shape[:-2], 4)


def rig_model(data: bytes) -> tuple[bytes, dict]:
    """Rigged GLB for any upload; results of the auto-rigger are cached by content hash."""
    glb = Glb.parse(data)
    if glb.gltf.get("skins"):
        return data, {"rig": "existing"}
    cache = OUT / "rigged" / f"{hashlib.sha256(data).hexdigest()[:24]}"
    if cache.with_suffix(".glb").exists():
        return cache.with_suffix(".glb").read_bytes(), json.loads(cache.with_suffix(".json").read_text())
    rigged, report = autorig(data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.with_suffix(".glb").write_bytes(rigged)
    cache.with_suffix(".json").write_text(json.dumps(report))
    return rigged, report


def make_handler(service: MotionService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
            pass

        def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_UPLOAD:
                raise ValueError("El archivo está vacío o supera los 200 MB")
            return self.rfile.read(length)

        def _model(self, data: bytes) -> None:
            rigged, report = rig_model(data)
            self._send(HTTPStatus.OK, rigged, "model/gltf-binary",
                       {"X-Rig": report["rig"], "X-Rig-Report": json.dumps(report, separators=(",", ":"))})

        def do_GET(self):
            try:
                path = self.path.split("?", 1)[0]
                if path == "/api/status":
                    return self._json(HTTPStatus.OK, service.status())
                if path == "/api/default-model":
                    return self._model(DEFAULT_MODEL.read_bytes())
                target = (STATIC / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
                if not target.is_file() or STATIC.resolve() not in target.parents:
                    return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                self._send(HTTPStatus.OK, target.read_bytes(), CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
            except ValueError as error:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})

        def do_POST(self):
            try:
                if self.path == "/api/rig":
                    return self._model(self._body())
                if self.path == "/api/generate":
                    return self._json(HTTPStatus.OK, service.generate(json.loads(self._body())))
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (ValueError, RuntimeError) as error:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            except Exception as error:
                traceback.print_exc()
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def serve(port: int) -> None:
    service = MotionService()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    print(f"kimodo gym on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
