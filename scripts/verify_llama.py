"""Verify the ungated Llama-3 mirror is byte-identical to meta-llama's release, then alias it.

The reference kimodo pipeline resolves `meta-llama/Meta-Llama-3-8B-Instruct` (gated) through the
LLM2Vec adapter config. After the weights pass the SHA-256 check against Meta's published LFS
object IDs, the mirror snapshot is registered in the local HF cache under the gated repo ID so
the unmodified reference code loads exactly the same bytes (run it with HF_HUB_OFFLINE=1).
"""

import hashlib
import os
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
HUB = TOOL_ROOT / "out" / "hf" / "hub"
MIRROR = "NousResearch/Meta-Llama-3-8B-Instruct"
GATED = "meta-llama/Meta-Llama-3-8B-Instruct"
ALIAS_REVISION = "verified-mirror"

# LFS sha256 object IDs published by meta-llama/Meta-Llama-3-8B-Instruct.
OFFICIAL_SHA256 = {
    "model-00001-of-00004.safetensors": "d8cf9c4d0dd972e1a2131bfe656235ee98221679711a3beef6d46dadf0f20b5c",
    "model-00002-of-00004.safetensors": "8d4782b4a69ef03845159ce1a15e272aadaaf134dc138d68f616098e8531729c",
    "model-00003-of-00004.safetensors": "3acdd690e65c24f42a24581b8467af98bd3ca357444580f8012aacd2bd607921",
    "model-00004-of-00004.safetensors": "67e9ad31c8c32abf3a55ee7fc7217b3ecb35fd3c74d98a5bd233e0e4d6964f46",
}


def snapshot_dir(repo_id: str) -> Path:
    repo_dir = HUB / f"models--{repo_id.replace('/', '--')}"
    revision = (repo_dir / "refs" / "main").read_text().strip()
    return repo_dir / "snapshots" / revision


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    mirror = snapshot_dir(MIRROR)
    for name, expected in OFFICIAL_SHA256.items():
        actual = sha256(mirror / name)
        if actual != expected:
            print(f"MISMATCH {name}: {actual} != {expected}", file=sys.stderr)
            return 1
        print(f"ok {name}")

    alias_repo = HUB / f"models--{GATED.replace('/', '--')}"
    alias = alias_repo / "snapshots" / ALIAS_REVISION
    alias.mkdir(parents=True, exist_ok=True)
    for source in mirror.iterdir():
        target = alias / source.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(os.path.realpath(source))
    (alias_repo / "refs").mkdir(exist_ok=True)
    (alias_repo / "refs" / "main").write_text(ALIAS_REVISION)
    print(f"aliased {GATED} -> verified {MIRROR} snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
