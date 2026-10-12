"""Loopback-only motion generation, character preview and NPZ downloads."""

import base64
import json
import math
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import numpy as np

from .character_library import character_catalog, character_source
from .paths import OUT, UI
from .rigging import MAX_CHARACTER_BYTES, rig_character

MOTIONS = OUT / "motions"
MAX_BODY = 16 * 1024
CONTENT_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".glb": "model/gltf-binary",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


class MotionService:
    """Load once in the background; serialize generation on the GPU."""

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
            skeleton = model.output_skeleton
            self.skeleton = {
                "joint_names": list(skeleton.bone_order_names),
                "parents": [int(p) for p in np.asarray(skeleton.joint_parents)],
                "rest": skeleton.neutral_joints.numpy().astype(np.float32).ravel().tolist(),
            }
            self.model = model
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            traceback.print_exc()

    def status(self) -> dict:
        return {"ready": self.model is not None, "error": self.error}

    def generate(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object.")
        prompt = request.get("prompt", "")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
            raise ValueError("Enter a motion description of 1–2000 characters.")
        prompt = prompt.strip()
        duration = float(request.get("duration", 5))
        if not math.isfinite(duration) or not 0.5 <= duration <= 10:
            raise ValueError("Duration must be between 0.5 and 10 seconds.")
        seed = request.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295.")
        if self.model is None:
            raise RuntimeError(self.error or "The motion model is still loading.")
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.tools import seed_everything

        with self.lock:
            start = time.perf_counter()
            seed_everything(seed)
            motion = self.model(
                [prompt],
                [int(round(duration * self.model.fps))],
                num_denoising_steps=100,
                num_samples=1,
                multi_prompt=True,
                post_processing=True,
                return_numpy=True,
                progress_bar=lambda frames: frames,
            )
            seconds = time.perf_counter() - start
            MOTIONS.mkdir(parents=True, exist_ok=True)
            filename = f"{time.strftime('%Y-%m-%d-%H%M%S')}-{uuid4().hex[:8]}.npz"
            # Export the single sample in the same shape as the CLI.
            save_kimodo_npz(str(MOTIONS / filename), {key: value[0] for key, value in motion.items()})
        positions = motion["posed_joints"][0].astype(np.float32)
        rotations = _quaternions(motion["global_rot_mats"][0]).astype(np.float32)
        return {
            "fps": float(self.model.fps),
            "frames": int(positions.shape[0]),
            **self.skeleton,
            "positions": base64.b64encode(positions.tobytes()).decode(),
            "rotations": base64.b64encode(rotations.tobytes()).decode(),
            "seconds": seconds,
            "prompt": prompt,
            "seed": seed,
            "download": f"/motions/{filename}",
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


def make_handler(service: MotionService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if int(args[1]) >= 400:
                super().log_message(fmt, *args)

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

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/status":
                return self._json(HTTPStatus.OK, service.status())
            if path == "/api/characters" or path.startswith("/api/characters/"):
                try:
                    if path == "/api/characters":
                        return self._json(HTTPStatus.OK, {"characters": [
                            {"id": entry["id"], "label": entry["label"], "url": f'/api/characters/{entry["id"]}'}
                            for entry in character_catalog()
                        ]})
                    return self._send(HTTPStatus.OK, character_source(path.removeprefix("/api/characters/")),
                                      "model/gltf-binary")
                except FileNotFoundError as error:
                    return self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                except (OSError, ValueError, RuntimeError) as error:
                    return self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            base = MOTIONS if path.startswith("/motions/") else UI
            relative = path.removeprefix("/motions/") if base == MOTIONS else path.lstrip("/") or "index.html"
            target = (base / relative).resolve()
            if base.resolve() not in target.parents or not target.is_file():
                return self._json(HTTPStatus.NOT_FOUND, {"error": "File not found."})
            if base == MOTIONS and target.suffix != ".npz":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "Motion not found."})
            self._send(HTTPStatus.OK, target.read_bytes(), CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

        def do_POST(self):
            if self.path == "/api/character":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= MAX_CHARACTER_BYTES:
                        raise ValueError("Choose a GLB character between 1 byte and 100 MB.")
                    path, report = rig_character(self.rfile.read(length))
                    self._send(HTTPStatus.OK, path.read_bytes(), "model/gltf-binary", {
                        "X-Rig-Report": json.dumps(report, separators=(",", ":"), ensure_ascii=True),
                    })
                except (ValueError, RuntimeError, OSError) as error:
                    self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
                return
            if self.path != "/api/generate":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send application/json."})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Request must contain 1–16384 bytes of JSON.")
                payload = json.loads(self.rfile.read(length))
                self._json(HTTPStatus.OK, service.generate(payload))
            except (ValueError, TypeError, RuntimeError) as error:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            except Exception as error:
                traceback.print_exc()
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def serve(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(MotionService()))
    print(f"Kimodo Turbo: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Kimodo Turbo.")
    finally:
        server.server_close()
