"""Minimal glTF-binary (GLB) reader/writer: JSON + one BIN chunk, typed accessor access."""

import json
import struct
from dataclasses import dataclass

import numpy as np

COMPONENT_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
UNSUPPORTED_EXTENSIONS = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_mesh_quantization"}


@dataclass
class Glb:
    gltf: dict
    bin: bytearray

    @classmethod
    def parse(cls, data: bytes) -> "Glb":
        magic, version, _ = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67 or version != 2:
            raise ValueError("Not a glTF 2.0 binary (.glb) file")
        offset, gltf, binary = 12, None, bytearray()
        while offset < len(data):
            length, kind = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8 : offset + 8 + length]
            if kind == 0x4E4F534A:
                gltf = json.loads(chunk)
            elif kind == 0x004E4942:
                binary = bytearray(chunk)
            offset += 8 + length
        if gltf is None:
            raise ValueError("GLB has no JSON chunk")
        used = set(gltf.get("extensionsUsed", [])) & UNSUPPORTED_EXTENSIONS
        if used:
            raise ValueError(f"Compressed meshes are not supported ({', '.join(sorted(used))}); re-export uncompressed")
        if any("uri" in b for b in gltf.get("buffers", [])):
            raise ValueError("External buffers are not supported; use a self-contained .glb")
        return cls(gltf, binary)

    def to_bytes(self) -> bytes:
        payload = json.dumps(self.gltf, separators=(",", ":")).encode()
        payload += b" " * (-len(payload) % 4)
        binary = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        total = 12 + 8 + len(payload) + 8 + len(binary)
        return (
            struct.pack("<III", 0x46546C67, 2, total)
            + struct.pack("<II", len(payload), 0x4E4F534A) + payload
            + struct.pack("<II", len(binary), 0x004E4942) + binary
        )

    def read_accessor(self, index: int) -> np.ndarray:
        acc = self.gltf["accessors"][index]
        if "sparse" in acc:
            raise ValueError("Sparse accessors are not supported")
        dtype = np.dtype(COMPONENT_DTYPES[acc["componentType"]])
        width = TYPE_SIZES[acc["type"]]
        view = self.gltf["bufferViews"][acc["bufferView"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride", width * dtype.itemsize)
        count = acc["count"]
        raw = np.frombuffer(self.bin, dtype=np.uint8, count=stride * (count - 1) + width * dtype.itemsize, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, width * dtype.itemsize), strides=(stride, 1))
        values = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
        if acc.get("normalized"):
            values = values.astype(np.float32) / np.iinfo(dtype).max
        return values

    def add_accessor(self, array: np.ndarray, component_type: int, kind: str, target: int | None = None) -> int:
        array = np.ascontiguousarray(array, dtype=COMPONENT_DTYPES[component_type])
        self.bin += b"\0" * (-len(self.bin) % 4)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": array.nbytes}
        if target is not None:
            view["target"] = target
        self.bin += array.tobytes()
        self.gltf.setdefault("bufferViews", []).append(view)
        accessor = {
            "bufferView": len(self.gltf["bufferViews"]) - 1,
            "componentType": component_type,
            "count": int(array.shape[0]),
            "type": kind,
        }
        if kind == "VEC3" and component_type == 5126:
            accessor["min"] = array.min(0).tolist()
            accessor["max"] = array.max(0).tolist()
        self.gltf.setdefault("accessors", []).append(accessor)
        buffers = self.gltf.setdefault("buffers", [{"byteLength": 0}])
        buffers[0]["byteLength"] = len(self.bin)
        return len(self.gltf["accessors"]) - 1


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * s
    m[:3, 3] = t
    return m


def world_matrices(gltf: dict) -> dict[int, np.ndarray]:
    """World matrix of every node reachable from the default scene."""
    result = {}
    scene = gltf["scenes"][gltf.get("scene", 0)]

    def visit(index: int, parent: np.ndarray) -> None:
        world = parent @ node_matrix(gltf["nodes"][index])
        result[index] = world
        for child in gltf["nodes"][index].get("children", []):
            visit(child, world)

    for root in scene["nodes"]:
        visit(root, np.eye(4))
    return result
