"""Content-addressed character cache around the native autorig executable."""

import hashlib
import json
import os
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

from .paths import OUT

MAX_CHARACTER_BYTES = 100 * 1024 * 1024
CHARACTERS = OUT / "characters"
AUTORIG = OUT / "bin" / "kimodo-autorig"
CACHE_VERSION = "character-rig-v1"
_LOCK = threading.Lock()


def _glb_document(data: bytes) -> dict:
    if len(data) < 20:
        raise ValueError("Choose a complete .glb character file.")
    magic, version, length, chunk_length, chunk_type = struct.unpack_from("<5I", data)
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError("The character must be a valid glTF 2 .glb file.")
    if chunk_type != 0x4E4F534A or chunk_length > len(data) - 20:
        raise ValueError("The character's GLB document is incomplete.")
    try:
        document = json.loads(data[20:20 + chunk_length])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("The character's GLB document is invalid.") from error
    if not isinstance(document, dict) or not document.get("meshes"):
        raise ValueError("The character file contains no mesh.")
    for kind in ("buffers", "images"):
        resources = document.get(kind, [])
        if not isinstance(resources, list):
            raise ValueError("The character's GLB resources are invalid.")
        for resource in resources:
            if not isinstance(resource, dict) or not isinstance(resource.get("uri", ""), str):
                raise ValueError("The character's GLB resources are invalid.")
            uri = resource.get("uri", "")
            if uri and not uri.startswith("data:"):
                raise ValueError("Use a self-contained .glb with embedded textures and geometry.")
    return document


def _validate_output(path: Path) -> str:
    if not path.is_file() or not 20 <= path.stat().st_size <= 2 * MAX_CHARACTER_BYTES:
        raise ValueError("The prepared character file is missing or too large.")
    data = path.read_bytes()
    document = _glb_document(data)
    skins, nodes = document.get("skins"), document.get("nodes", [])
    if not isinstance(skins, list) or not skins or not isinstance(nodes, list) or not any(
        isinstance(node, dict) and "skin" in node for node in nodes
    ):
        raise ValueError("The prepared character contains no usable skeleton.")
    return hashlib.sha256(data).hexdigest()


def rig_character(data: bytes) -> tuple[Path, dict]:
    """Validate a GLB, prepare it once, and return the cached model and rig report."""
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_CHARACTER_BYTES:
        raise ValueError("Choose a .glb character smaller than 100 MB.")
    _glb_document(data)
    if not AUTORIG.is_file() or not os.access(AUTORIG, os.X_OK):
        raise RuntimeError("Character preparation is unavailable. Run ./scripts/setup.zsh to build it.")
    source_sha = hashlib.sha256(data).hexdigest()
    binary_sha = hashlib.sha256(AUTORIG.read_bytes()).hexdigest()
    cache_key = hashlib.sha256(f"{CACHE_VERSION}:{binary_sha}:{source_sha}".encode()).hexdigest()
    model_path = CHARACTERS / f"{cache_key}.glb"
    metadata_path = CHARACTERS / f"{cache_key}.json"
    with _LOCK:
        CHARACTERS.mkdir(parents=True, exist_ok=True)
        if model_path.exists() or metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                report = metadata["report"]
                valid = (metadata["cache_version"] == CACHE_VERSION
                         and metadata["input_sha256"] == source_sha
                         and metadata["binary_sha256"] == binary_sha
                         and metadata["output_sha256"] == _validate_output(model_path)
                         and isinstance(report, dict))
                if not valid:
                    raise ValueError("Character cache verification failed.")
                return model_path, report
            except (OSError, ValueError, KeyError, TypeError) as error:
                model_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                raise RuntimeError("The cached character was damaged. Choose the file again to rebuild it.") from error
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=CHARACTERS) as temporary:
            scratch = Path(temporary)
            source = scratch / "source.glb"
            output = scratch / "character.glb"
            report_path = scratch / "report.json"
            source.write_bytes(data)
            try:
                result = subprocess.run(
                    [str(AUTORIG.resolve()), str(source), str(output), "--report", str(report_path)],
                    cwd=scratch, capture_output=True, text=True, timeout=120, check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("Character preparation took too long. Try a smaller character file.") from error
            except OSError as error:
                raise RuntimeError("Could not start character preparation. Run ./scripts/setup.zsh again.") from error
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-2000:]
                raise ValueError(detail or "The character could not be prepared. Use a standing humanoid model.")
            try:
                output_sha = _validate_output(output)
                report = json.loads(report_path.read_text())
                if not isinstance(report, dict):
                    raise ValueError("Invalid preparation report.")
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Character preparation produced invalid output: {error}") from error
            metadata = {
                "cache_version": CACHE_VERSION, "input_sha256": source_sha,
                "binary_sha256": binary_sha, "output_sha256": output_sha, "report": report,
            }
            report_path.write_text(json.dumps(metadata, indent=2) + "\n")
            output.replace(model_path)
            report_path.replace(metadata_path)
            return model_path, report
