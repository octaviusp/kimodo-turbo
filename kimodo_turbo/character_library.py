"""Local character presets; source assets stay outside Git under out/."""

import hashlib
import json

from .paths import OUT
from .rigging import MAX_CHARACTER_BYTES

LIBRARY = OUT / "characters" / "library"


def character_catalog() -> list[dict]:
    """Read the local preset index without exposing source paths to the browser."""
    index = LIBRARY / "index.json"
    if not index.exists():
        return []
    try:
        characters = json.loads(index.read_text())["characters"]
        if not isinstance(characters, list):
            raise ValueError("Expected a character list.")
        seen = set()
        for character in characters:
            identifier, label, filename, digest = (character[key] for key in ("id", "label", "file", "sha256"))
            if (not all(isinstance(value, str) and value for value in (identifier, label, filename, digest))
                    or not all(c.isascii() and (c.isalnum() or c == "-") for c in identifier)
                    or identifier in seen or filename != f"{identifier}.glb"
                    or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Invalid character entry.")
            seen.add(identifier)
        return characters
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("The local character library index is invalid.") from error


def character_source(identifier: str) -> bytes:
    """Only serve indexed, unchanged GLBs from the tool's own local library."""
    character = next((entry for entry in character_catalog() if entry["id"] == identifier), None)
    if character is None:
        raise FileNotFoundError("Character not found.")
    path = (LIBRARY / character["file"]).resolve()
    if path.parent != LIBRARY.resolve() or not path.is_file():
        raise FileNotFoundError("The local character file is missing.")
    if not 0 < path.stat().st_size <= MAX_CHARACTER_BYTES:
        raise ValueError("The local character file is too large or empty.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != character["sha256"]:
        raise ValueError("The local character file has changed. Restore its original library copy.")
    return data
