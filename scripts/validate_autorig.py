#!/usr/bin/env python3
"""Independent GLB skinning oracle, deterministic stress tests, and browser evidence.

Run with the project's Python environment. Browser rendering additionally needs
Playwright; install it only in a scratch environment or the project's .venv.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import pathlib
import shutil
import statistics
import struct
import subprocess
import threading
import time
from collections import defaultdict
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
POSE_NAMES = ("rest", "arms_raised", "wide_stance", "crouch", "torso_twist", "asymmetric")
COMPONENT_TYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
ELEMENT_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quaternion_matrix(q: list[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    magnitude = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or magnitude < 1e-12:
        raise ValueError("Node quaternion must contain four finite values with nonzero length")
    x, y, z, w = q / magnitude
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-15)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    theta = math.radians(degrees)
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * k @ k


def swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    cosine = float(np.clip(a @ b, -1, 1))
    if np.linalg.norm(axis) < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.8 else [0, 1, 0])
    return rotation(axis, math.degrees(math.acos(cosine)))


def orthogonal(linear: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(linear)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    return r


def canonical(name: str) -> str | None:
    name = name.lower().replace("mixamorig", "")
    name = "".join(c for c in name if c.isalnum())
    core = {
        "hips": "Hips", "pelvis": "Hips", "hip": "Hips", "spine": "Spine1", "spine1": "Spine1",
        "spine01": "Spine1", "spine2": "Spine2", "spine02": "Spine2", "spine3": "Chest",
        "chest": "Chest", "upperchest": "Chest", "neck": "Neck1", "neck1": "Neck1", "head": "Head",
    }
    if name in core:
        return core[name]
    for side, abbreviations in (("Left", ("left", "l")), ("Right", ("right", "r"))):
        for short in abbreviations:
            rest = name[len(short):] if name.startswith(short) else name[:-len(short)] if name.endswith(short) else ""
            suffix = {
                "shoulder": "Shoulder", "clavicle": "Shoulder", "arm": "Arm", "upperarm": "Arm",
                "forearm": "ForeArm", "lowerarm": "ForeArm", "hand": "Hand", "upleg": "Leg", "leg": "Leg",
                "thigh": "Leg", "shin": "Shin", "calf": "Shin", "lowerleg": "Shin", "foot": "Foot",
                "toebase": "ToeBase", "toe": "ToeBase",
            }.get(rest)
            if suffix:
                return side + suffix
    return None


class Glb:
    """Small GLB reader deliberately independent from the native implementation."""

    def __init__(self, path: pathlib.Path, frame_hint: dict[str, Any] | None = None):
        self.path = path
        blob = path.read_bytes()
        magic, version, length = struct.unpack_from("<III", blob)
        if (magic, version, length) != (0x46546C67, 2, len(blob)):
            raise ValueError(f"Invalid GLB header: {path}")
        self.data: dict[str, Any] = {}
        self.binary = b""
        offset = 12
        while offset < length:
            size, kind = struct.unpack_from("<II", blob, offset)
            chunk = blob[offset + 8:offset + 8 + size]
            if kind == 0x4E4F534A:
                self.data = json.loads(chunk)
            elif kind == 0x004E4942:
                self.binary = chunk
            offset += 8 + size
        if not self.data:
            raise ValueError("GLB has no JSON chunk")
        for buffer in self.data.get("buffers", []):
            if buffer.get("uri"):
                raise ValueError("Validation expects self-contained GLB buffers")
        self.nodes = self.data.get("nodes", [])
        self.parent: dict[int, int] = {}
        for i, node in enumerate(self.nodes):
            for child in node.get("children", []):
                if child in self.parent:
                    raise ValueError("A glTF node has more than one parent")
                self.parent[child] = i
        self.local = np.stack([self.local_matrix(node) for node in self.nodes]) if self.nodes else np.zeros((0, 4, 4))
        self.world = self.world_matrices(self.local)
        self.names = {canonical(node.get("name", "")): i for i, node in enumerate(self.nodes) if canonical(node.get("name", ""))}
        self.primitives = self.read_primitives()
        positions = np.concatenate([p["rest"] for p in self.primitives])
        self.bounds = np.stack([positions.min(axis=0), positions.max(axis=0)])
        self.up, self.lateral, self.forward = self.frame()
        if frame_hint and "Hips" not in self.names:
            def axis(value: str) -> np.ndarray:
                value = value.lower().strip()
                return np.eye(3)["xyz".index(value[-1])] * (-1 if value.startswith("-") else 1)
            self.up = axis(frame_hint.get("up", frame_hint.get("up_axis", "y")))
            self.forward = axis(frame_hint.get("forward", frame_hint.get("forward_axis", "z")))
            self.lateral = np.cross(self.up, self.forward)
        self.height = float(np.ptp(positions @ self.up))

    def accessor(self, index: int) -> np.ndarray:
        accessor = self.data["accessors"][index]
        dtype = np.dtype(COMPONENT_TYPES[accessor["componentType"]])
        width, count = ELEMENT_SIZES[accessor["type"]], accessor["count"]
        if "bufferView" in accessor:
            view = self.data["bufferViews"][accessor["bufferView"]]
            if "EXT_meshopt_compression" in view.get("extensions", {}):
                raise ValueError("Compressed accessors require decompression before validation")
            offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            stride = view.get("byteStride", dtype.itemsize * width)
            result = np.ndarray((count, width), dtype=dtype, buffer=self.binary, offset=offset,
                                strides=(stride, dtype.itemsize)).copy()
        else:
            result = np.zeros((count, width), dtype=dtype)
        if "sparse" in accessor:
            sparse = accessor["sparse"]
            idx_spec, val_spec = sparse["indices"], sparse["values"]
            idx_view = self.data["bufferViews"][idx_spec["bufferView"]]
            val_view = self.data["bufferViews"][val_spec["bufferView"]]
            indices = np.frombuffer(self.binary, dtype=COMPONENT_TYPES[idx_spec["componentType"]], count=sparse["count"],
                                    offset=idx_view.get("byteOffset", 0) + idx_spec.get("byteOffset", 0))
            values = np.frombuffer(self.binary, dtype=dtype, count=sparse["count"] * width,
                                   offset=val_view.get("byteOffset", 0) + val_spec.get("byteOffset", 0)).reshape(-1, width)
            result[indices] = values
        if accessor.get("normalized") and dtype.kind in "iu":
            info = np.iinfo(dtype)
            result = result.astype(float) / info.max
            if dtype.kind == "i":
                result = np.maximum(result, -1)
        return result

    @staticmethod
    def local_matrix(node: dict[str, Any]) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=float).reshape(4, 4).T
        result = np.eye(4)
        result[:3, :3] = quaternion_matrix(node.get("rotation", [0, 0, 0, 1])) @ np.diag(node.get("scale", [1, 1, 1]))
        result[:3, 3] = node.get("translation", [0, 0, 0])
        return result

    def world_matrices(self, local: np.ndarray) -> np.ndarray:
        world = np.zeros_like(local)
        complete: set[int] = set()
        active: set[int] = set()

        def visit(i: int) -> np.ndarray:
            if i in active:
                raise ValueError("Cycle in glTF node hierarchy")
            if i not in complete:
                active.add(i)
                world[i] = visit(self.parent[i]) @ local[i] if i in self.parent else local[i]
                active.remove(i)
                complete.add(i)
            return world[i]

        for i in range(len(local)):
            visit(i)
        return world

    def read_primitives(self) -> list[dict[str, Any]]:
        primitives = []
        for node_id, node in enumerate(self.nodes):
            if "mesh" not in node:
                continue
            for primitive_id, primitive in enumerate(self.data["meshes"][node["mesh"]]["primitives"]):
                if primitive.get("mode", 4) != 4:
                    continue
                attributes = primitive.get("attributes", {})
                if "POSITION" not in attributes:
                    continue
                points = self.accessor(attributes["POSITION"]).astype(float)
                if not np.isfinite(points).all():
                    raise ValueError("Non-finite vertex positions")
                point4 = np.column_stack([points, np.ones(len(points))])
                indices = self.accessor(primitive["indices"]).ravel().astype(int) if "indices" in primitive else np.arange(len(points))
                if len(indices) % 3:
                    raise ValueError("Triangle indices are not a multiple of three")
                if np.any(indices < 0) or np.any(indices >= len(points)):
                    raise ValueError("Triangle indices out of range")
                item: dict[str, Any] = {
                    "node": node_id, "primitive": primitive_id, "points": point4,
                    "rest": (point4 @ self.world[node_id].T)[:, :3], "triangles": indices.reshape(-1, 3),
                    "material": primitive.get("material"),
                    "texcoords": {name: self.accessor(index).astype(float) for name, index in attributes.items()
                                  if name.startswith("TEXCOORD_")},
                }
                if "NORMAL" in attributes:
                    normals = self.accessor(attributes["NORMAL"]).astype(float)
                    normal_matrix = np.linalg.inv(self.world[node_id, :3, :3]).T
                    world_normals = normals @ normal_matrix.T
                    lengths = np.linalg.norm(world_normals, axis=1, keepdims=True)
                    item["normals"] = world_normals / np.maximum(lengths, 1e-20)
                    if not np.isfinite(item["normals"]).all():
                        raise ValueError("Non-finite vertex normals")
                if "skin" in node:
                    skin = self.data["skins"][node["skin"]]
                    joints = np.array(skin["joints"], dtype=int)
                    if "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
                        raise ValueError("A skinned primitive is missing joints/weights")
                    js, ws = [], []
                    for group in (0, 1):
                        if f"JOINTS_{group}" in attributes:
                            js.append(self.accessor(attributes[f"JOINTS_{group}"]).astype(int))
                            ws.append(self.accessor(attributes[f"WEIGHTS_{group}"]).astype(float))
                    joint_indices, weights = np.concatenate(js, axis=1), np.concatenate(ws, axis=1)
                    if joint_indices.shape != weights.shape or len(weights) != len(points):
                        raise ValueError("Skin attribute dimensions disagree")
                    if np.any(joint_indices < 0) or np.any(joint_indices >= len(joints)):
                        raise ValueError("Skin joint index out of range")
                    inverse_bind = self.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1) if "inverseBindMatrices" in skin else np.tile(np.eye(4), (len(joints), 1, 1))
                    if len(inverse_bind) != len(joints):
                        raise ValueError("Inverse bind matrix count disagrees with joints")
                    item.update(joints=joints, joint_indices=joint_indices, weights=weights, inverse_bind=inverse_bind)
                primitives.append(item)
        if not primitives:
            raise ValueError("No uncompressed triangle geometry found")
        return primitives

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if "Hips" in self.names:
            hips = self.world[self.names["Hips"], :3, 3]
            top = next((self.names[n] for n in ("Chest", "Spine2", "Spine1", "Head") if n in self.names), None)
            pair = next((n for n in ("Leg", "Arm", "Shoulder") if "Left" + n in self.names and "Right" + n in self.names), None)
            if top is not None and pair:
                up = self.world[top, :3, 3] - hips
                up /= np.linalg.norm(up)
                lateral = self.world[self.names["Left" + pair], :3, 3] - self.world[self.names["Right" + pair], :3, 3]
                lateral -= up * (up @ lateral)
                lateral /= np.linalg.norm(lateral)
                return up, lateral, np.cross(lateral, up)
        up = np.eye(3)[np.argmax(np.ptp(self.bounds, axis=0))]
        lateral = np.eye(3)[0 if abs(up[0]) < 0.8 else 2]
        return up, lateral, np.cross(lateral, up)

    def posed_local(self, pose: str) -> np.ndarray:
        local = self.local.copy()
        actions: dict[str, np.ndarray] = {}
        if pose in ("arms_raised", "asymmetric"):
            for side, sign in (("Left", 1), ("Right", -1)):
                arm, forearm = self.names.get(side + "Arm"), self.names.get(side + "ForeArm")
                if arm is not None and forearm is not None:
                    direction = self.world[forearm, :3, 3] - self.world[arm, :3, 3]
                    target = self.up * 0.96 + self.lateral * sign * 0.28
                    if pose == "asymmetric" and side == "Right":
                        target = -self.up * 0.65 + self.lateral * sign * 0.72
                    actions[side + "Arm"] = swing(direction, target)
                    actions[side + "ForeArm"] = rotation(self.lateral, -25 if pose == "arms_raised" else -55)
        if pose in ("wide_stance", "asymmetric"):
            angle = 35 if pose == "wide_stance" else 24
            actions.update(LeftLeg=rotation(self.forward, angle), RightLeg=rotation(self.forward, -angle))
        if pose == "crouch":
            actions.update(Hips=rotation(self.lateral, 15), LeftLeg=rotation(self.lateral, -85),
                           RightLeg=rotation(self.lateral, -85), LeftShin=rotation(self.lateral, 105),
                           RightShin=rotation(self.lateral, 105), LeftFoot=rotation(self.lateral, -20),
                           RightFoot=rotation(self.lateral, -20), Spine1=rotation(self.lateral, -25))
        if pose in ("torso_twist", "asymmetric"):
            actions.update(Spine1=rotation(self.up, 25), Spine2=rotation(self.up, 25), Chest=rotation(self.up, 20))
        for name, delta in actions.items():
            if name not in self.names:
                continue
            i = self.names[name]
            frame = orthogonal(self.world[i, :3, :3])
            local_delta = np.eye(4)
            local_delta[:3, :3] = frame.T @ delta @ frame
            local[i] = local[i] @ local_delta
        return local

    def deform(self, primitive: dict[str, Any], world: np.ndarray) -> np.ndarray:
        if "weights" not in primitive:
            return (primitive["points"] @ world[primitive["node"]].T)[:, :3]
        transforms = world[primitive["joints"]] @ primitive["inverse_bind"]
        output = np.zeros((len(primitive["points"]), 4))
        for k in range(primitive["weights"].shape[1]):
            moved = np.einsum("nij,nj->ni", transforms[primitive["joint_indices"][:, k]], primitive["points"])
            output += moved * primitive["weights"][:, k:k + 1]
        return output[:, :3]


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0 or float(weights.sum()) <= 0:
        return 0.0
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    i = min(int(np.searchsorted(cumulative, cumulative[-1] * q)), len(order) - 1)
    return float(values[order[i]])


def areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                        vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)


def metric_summary(values: np.ndarray, weights: np.ndarray, threshold: float) -> dict[str, float]:
    total = max(float(weights.sum()), 1e-20)
    return {"p95": weighted_percentile(values, weights, 0.95), "p99": weighted_percentile(values, weights, 0.99),
            "maximum": float(values.max(initial=0)), "area_fraction_above_threshold": float(weights[values > threshold].sum()) / total,
            "threshold": threshold}


def validate(glb: Glb) -> tuple[dict[str, Any], dict[str, Any]]:
    problems: list[str] = []
    weights_errors, rest_errors = [], []
    edge_weights, edge_ratios, area_weights, area_ratios = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    significant_edge_extension = defaultdict(list)
    leakage = defaultdict(list)
    vertex_areas, pelvis_weights = [], []
    total_vertices = sum(len(p["points"]) for p in glb.primitives)
    skinned_vertices = sum(len(p["points"]) for p in glb.primitives if "weights" in p)
    if not skinned_vertices:
        problems.append("No skinned triangle vertices")
    shape = np.concatenate([p["rest"] for p in glb.primitives])
    h = max(glb.height, 1e-15)
    all_local = {pose: glb.posed_local(pose) for pose in POSE_NAMES}
    all_world = {pose: glb.world_matrices(local) for pose, local in all_local.items()}
    moved = {pose: [] for pose in POSE_NAMES}
    signed_volumes = {pose: 0.0 for pose in POSE_NAMES}
    for p in glb.primitives:
        rest, faces = p["rest"], p["triangles"]
        face_areas = areas(rest, faces)
        mass = np.zeros(len(rest))
        for k in range(3):
            np.add.at(mass, faces[:, k], face_areas / 3)
        vertex_areas.append(mass)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edge_area = np.tile(face_areas / 3, 3)
        lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
        valid_edges = lengths > h * 1e-7
        edges, lengths, edge_area = edges[valid_edges], lengths[valid_edges], edge_area[valid_edges]
        if "weights" in p:
            weights = p["weights"]
            if not np.isfinite(weights).all() or not np.isfinite(p["inverse_bind"]).all():
                problems.append("Non-finite skin weights or inverse bind matrices")
            if np.any(weights < -1e-7) or np.any(weights > 1 + 1e-7):
                problems.append("Skin weights outside [0, 1]")
            weights_errors.extend(np.abs(weights.sum(axis=1) - 1))
            skin_names = [canonical(glb.nodes[j].get("name", "")) or glb.nodes[j].get("name", "") for j in p["joints"]]
            category = np.zeros((len(weights), 5))
            for k in range(weights.shape[1]):
                for j, name in enumerate(skin_names):
                    group = 0 if name.startswith("Left") and any(s in name for s in ("Arm", "Hand", "Shoulder")) else \
                            1 if name.startswith("Right") and any(s in name for s in ("Arm", "Hand", "Shoulder")) else \
                            2 if name.startswith("Left") and any(s in name for s in ("Leg", "Shin", "Foot", "Toe")) else \
                            3 if name.startswith("Right") and any(s in name for s in ("Leg", "Shin", "Foot", "Toe")) else 4
                    category[:, group] += weights[:, k] * (p["joint_indices"][:, k] == j)
            leakage["arm_torso_mix"].extend(np.minimum(category[:, 0] + category[:, 1], category[:, 4]))
            leakage["opposite_leg_mix"].extend(np.minimum(category[:, 2], category[:, 3]))
            leakage["opposite_arm_mix"].extend(np.minimum(category[:, 0], category[:, 1]))
            pelvis = np.zeros(len(rest))
            for k in range(weights.shape[1]):
                for j, name in enumerate(skin_names):
                    if name == "Hips":
                        pelvis += weights[:, k] * (p["joint_indices"][:, k] == j)
            pelvis_weights.append(pelvis)
            dense = np.zeros((len(weights), len(p["joints"])))
            for k in range(weights.shape[1]):
                np.add.at(dense, (np.arange(len(weights)), p["joint_indices"][:, k]), weights[:, k])
            leakage["edge_weight_total_variation"].extend(0.5 * np.abs(dense[edges[:, 0]] - dense[edges[:, 1]]).sum(axis=1))
        else:
            pelvis_weights.append(np.zeros(len(rest)))
        for pose, world in all_world.items():
            vertices = glb.deform(p, world)
            moved[pose].append(vertices)
            if not np.isfinite(vertices).all():
                problems.append(f"Non-finite deformed coordinates in {pose}")
            if pose == "rest":
                rest_errors.extend(np.linalg.norm(vertices - rest, axis=1) / h)
            ratio = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) / lengths
            significant_edge_extension[pose].extend(np.maximum(ratio - 1, 0) * lengths / h)
            edge_weights[pose].extend(edge_area)
            edge_ratios[pose].extend(ratio)
            valid_faces = face_areas > h * h * 1e-12
            area_weights[pose].extend(face_areas[valid_faces])
            area_ratios[pose].extend(areas(vertices, faces)[valid_faces] / face_areas[valid_faces])
            signed_volumes[pose] += float(np.einsum("ij,ij->i", vertices[faces[:, 0]],
                                                   np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6)
    rest_max = float(np.max(rest_errors, initial=0))
    weight_max = float(np.max(weights_errors, initial=0))
    if rest_max > 1e-5:
        problems.append(f"Rest pose does not recover the original geometry: {rest_max:.3g} character heights")
    if weight_max > 5e-4:
        problems.append(f"Weights do not sum to one: maximum error {weight_max:.3g}")
    if skinned_vertices != total_vertices:
        problems.append(f"Only {skinned_vertices}/{total_vertices} vertices are skinned")
    pose_stats = {}
    vertex_mass = np.concatenate(vertex_areas)
    pelvis_weight = np.concatenate(pelvis_weights)
    for pose in POSE_NAMES:
        vertices = np.concatenate(moved[pose])
        displacement = np.linalg.norm(vertices - shape, axis=1) / h
        er, ew = np.array(edge_ratios[pose]), np.array(edge_weights[pose])
        ar, aw = np.array(area_ratios[pose]), np.array(area_weights[pose])
        points_y = shape @ glb.up
        relative_y = (points_y - points_y.min()) / h
        relative_x = (shape - shape.mean(axis=0)) @ glb.lateral / h
        spatial_regions = {
            "left_outer_arm": (relative_x > 0.19) & (relative_y > 0.48) & (relative_y < 0.9),
            "right_outer_arm": (relative_x < -0.19) & (relative_y > 0.48) & (relative_y < 0.9),
            "left_lower_body": (relative_x > 0.05) & (relative_y < 0.42),
            "right_lower_body": (relative_x < -0.05) & (relative_y < 0.42),
            "left_foot_zone": (relative_x > 0.05) & (relative_y < 0.09),
            "right_foot_zone": (relative_x < -0.05) & (relative_y < 0.09),
            "pelvis_bound_lower_surface": (relative_y > 0.09) & (relative_y < 0.5) & (pelvis_weight > 0.8),
        }
        for side in ("Left", "Right"):
            hand = glb.names.get(side + "Hand")
            if hand is not None:
                distance = np.linalg.norm(shape - glb.world[hand, :3, 3], axis=1) / h
                spatial_regions[side.lower() + "_hand_zone"] = distance < 0.07
        coverage = {name: {"vertices": int(mask.sum()), "p95_displacement_height": float(np.percentile(displacement[mask], 95)) if mask.any() else None,
                           "fraction_moved_2pct_height": float(np.mean(displacement[mask] > 0.02)) if mask.any() else None,
                           "surface_area_fraction": float(vertex_mass[mask].sum() / max(vertex_mass.sum(), 1e-20)),
                           "area_fraction_moved_2pct_height": float(vertex_mass[mask & (displacement > 0.02)].sum() / max(vertex_mass[mask].sum(), 1e-20)) if mask.any() else None}
                    for name, mask in spatial_regions.items()}
        pose_stats[pose] = {
            "edge_stretch": metric_summary(er, ew, 2), "triangle_area_ratio": metric_summary(ar, aw, 4),
            "maximum_edge_extension_height": float(max(significant_edge_extension[pose], default=0)),
            "area_fraction_collapsed_below_10pct": float(aw[ar < 0.1].sum() / max(aw.sum(), 1e-20)),
            "movement_p95_height": float(np.percentile(displacement, 95)), "movement_coverage": coverage,
            "signed_volume_height_cubed": signed_volumes[pose] / h**3,
        }
    quality_flags = []
    for pose, stats in pose_stats.items():
        if pose == "rest":
            continue
        if stats["movement_p95_height"] < 1e-4:
            quality_flags.append(f"{pose}: the character remains effectively stationary")
        if stats["edge_stretch"]["area_fraction_above_threshold"] > 0.015:
            quality_flags.append(f"{pose}: more than 1.5% surface area has edges stretched above 2x")
        if stats["area_fraction_collapsed_below_10pct"] > 0.01:
            quality_flags.append(f"{pose}: more than 1% surface area collapses below 10% rest triangle area")
    if "LeftArm" not in glb.names or "RightArm" not in glb.names or "LeftLeg" not in glb.names or "RightLeg" not in glb.names:
        quality_flags.append("Canonical arm/leg joints missing; stress coverage incomplete")
    for pose, regions in (("arms_raised", ("left_outer_arm", "right_outer_arm")),
                          ("wide_stance", ("left_lower_body", "right_lower_body"))):
        for region in regions:
            coverage = pose_stats[pose]["movement_coverage"][region]
            if coverage["vertices"] >= 20 and coverage["fraction_moved_2pct_height"] < 0.5:
                quality_flags.append(f"{pose}: insufficient visible motion in {region}")
    for side in ("left", "right"):
        coverage = pose_stats["arms_raised"]["movement_coverage"].get(side + "_hand_zone")
        if coverage and coverage["vertices"] >= 20 and coverage["area_fraction_moved_2pct_height"] < 0.5:
            quality_flags.append(f"arms_raised: insufficient visible motion around {side} wrist")
    report = {
        "path": str(glb.path), "sha256": digest(glb.path), "vertices": total_vertices,
        "triangles": sum(len(p["triangles"]) for p in glb.primitives), "skinned_vertices": skinned_vertices,
        "canonical_bones": sorted(glb.names), "height": h, "rest_recovery_max_height": rest_max,
        "weight_sum_max_error": weight_max, "correctness_pass": not problems, "correctness_failures": sorted(set(problems)),
        "quality_review_required": bool(quality_flags), "quality_flags": quality_flags,
        "leakage": {name: {"p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
                           "maximum": float(max(v, default=0)), "fraction_above_0_2": float(np.mean(np.array(v) > 0.2))}
                    for name, v in leakage.items()}, "poses": pose_stats,
        "limitations": ["Linear blend skinning stress metrics cannot prove cloth realism or absence of self-intersection.",
                        "Signed volume is diagnostic only: garments and open meshes are not watertight.",
                        "Area-weighted stretch excludes zero-length edges and degenerate rest triangles.",
                        "Visible motion is checked spatially to expose stationary or pelvis-only bindings."],
    }
    samples = {pose: [] for pose in POSE_NAMES}
    for primitive_index, primitive in enumerate(glb.primitives):
        rest = primitive["rest"]
        indices = sorted({0, len(rest) // 2, len(rest) - 1,
                          *np.argmin(rest, axis=0).tolist(), *np.argmax(rest, axis=0).tolist()})
        for pose in POSE_NAMES:
            samples[pose].append({"node": primitive["node"], "primitive": primitive["primitive"],
                                  "indices": indices, "world_positions": moved[pose][primitive_index][indices].tolist()})
    viewer = {"path": str(glb.path), "up": glb.up.tolist(), "lateral": glb.lateral.tolist(), "height": h,
              "samples": samples,
              "pose_bounds": {pose: np.stack([np.concatenate(points).min(axis=0),
                                               np.concatenate(points).max(axis=0)]).tolist()
                              for pose, points in moved.items()},
              "poses": {name: local.transpose(0, 2, 1).reshape(-1, 16).tolist() for name, local in all_local.items()}}
    return report, viewer


def normalize_entries(manifest: pathlib.Path | None, inputs: list[str]) -> list[dict[str, Any]]:
    if manifest:
        data = json.loads(manifest.read_text())
        entries = data if isinstance(data, list) else data.get("characters", data.get("assets", data.get("cases", [])))
        if not entries:
            raise ValueError("Manifest must contain a characters, assets, or cases list")
        result = []
        for i, entry in enumerate(entries):
            entry = {"path": entry} if isinstance(entry, str) else dict(entry)
            entry["path"] = entry.get("path", entry.get("source", entry.get("source_path")))
            if not entry["path"]:
                raise ValueError(f"Corpus entry {i} has no path")
            entry["id"] = entry.get("id", entry.get("name", pathlib.Path(entry["path"]).stem))
            result.append(entry)
        return result
    return [{"id": pathlib.Path(path).stem, "path": path} for path in inputs]


def geometry_proof(source: Glb, output: Glb, allow_seam_split: bool) -> dict[str, Any]:
    failures, duplicates = [], []
    max_corner_error, max_prefix_error, max_uv_error, max_normal_error = 0.0, 0.0, 0.0, 0.0
    h = max(source.height, 1e-15)
    if len(source.primitives) != len(output.primitives):
        return {"pass": False, "failures": ["Primitive count changed"]}
    for original, rigged in zip(source.primitives, output.primitives, strict=True):
        a, b = original["rest"], rigged["rest"]
        fa, fb = original["triangles"], rigged["triangles"]
        if fa.shape != fb.shape:
            failures.append("Triangle count changed")
            continue
        max_corner_error = max(max_corner_error, float(np.linalg.norm(a[fa] - b[fb], axis=2).max(initial=0) / h))
        if len(b) < len(a):
            failures.append("Original vertices were removed")
            continue
        max_prefix_error = max(max_prefix_error, float(np.linalg.norm(a - b[:len(a)], axis=1).max(initial=0) / h))
        if not allow_seam_split and len(a) != len(b):
            failures.append("Vertex count changed for a character without a covered-leg garment")
        if original["material"] != rigged["material"]:
            failures.append("Primitive material changed")
        if ("normals" in original) != ("normals" in rigged):
            failures.append("Normal attribute set changed")
        elif "normals" in original:
            max_normal_error = max(max_normal_error, float(np.linalg.norm(original["normals"][fa]
                                                                           - rigged["normals"][fb], axis=2).max(initial=0)))
        if original["texcoords"].keys() != rigged["texcoords"].keys():
            failures.append("UV attribute set changed")
        else:
            for channel in original["texcoords"]:
                original_uv = original["texcoords"][channel][fa]
                rigged_uv = rigged["texcoords"][channel][fb]
                max_uv_error = max(max_uv_error, float(np.abs(original_uv - rigged_uv).max(initial=0)))
        flat_a, flat_b = fa.ravel(), fb.ravel()
        for vertex in range(len(a), len(b)):
            corners = flat_b == vertex
            mapped = np.unique(flat_a[corners])
            if len(mapped) != 1:
                failures.append("A duplicate vertex is unused or copies more than one source vertex")
                continue
            original_vertex = int(mapped[0])
            if not np.any(flat_b == original_vertex):
                failures.append("A duplicated source vertex has no retained incident faces")
            if np.linalg.norm(b[vertex] - a[original_vertex]) / h > 1e-5:
                failures.append("A seam duplicate changes the source geometry")
            duplicates.append({"primitive": original["primitive"], "source_vertex": original_vertex,
                               "new_vertex": vertex, "new_incident_faces": int(np.unique(np.nonzero(fb == vertex)[0]).size)})
    for field in ("materials", "textures", "samplers"):
        if source.data.get(field, []) != output.data.get(field, []):
            failures.append(f"{field.capitalize()} payload changed")
    def image_payloads(glb: Glb) -> list[str]:
        payloads = []
        for entry in glb.data.get("images", []):
            if "bufferView" in entry:
                view = glb.data["bufferViews"][entry["bufferView"]]
                start = view.get("byteOffset", 0)
                data = glb.binary[start:start + view["byteLength"]]
            else:
                data = entry.get("uri", "").encode()
            payloads.append(hashlib.sha256(data).hexdigest())
        return payloads
    if image_payloads(source) != image_payloads(output):
        failures.append("Embedded texture image bytes changed")
    if max_corner_error > 1e-5 or max_prefix_error > 1e-5:
        failures.append("World-space source geometry changed")
    if max_uv_error > 1e-7:
        failures.append("UV values changed at face corners")
    if max_normal_error > 1e-5:
        failures.append("World-space normal directions changed at face corners")
    return {"pass": not failures, "failures": sorted(set(failures)), "seam_split_allowed": allow_seam_split,
            "max_face_corner_error_height": max_corner_error, "max_original_prefix_error_height": max_prefix_error,
            "max_uv_corner_error": max_uv_error, "max_normal_corner_error": max_normal_error,
            "duplicated_seam_vertices": len(duplicates), "seam_duplicates": duplicates}


def run_case(entry: dict[str, Any], args: argparse.Namespace, qa: pathlib.Path) -> dict[str, Any]:
    name = "".join(c.lower() if c.isalnum() else "-" for c in entry["id"]).strip("-")
    folder = qa / name
    folder.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(entry["path"]).expanduser().resolve()
    result: dict[str, Any] = {"id": name, "source": str(source), "source_sha256": digest(source)}
    variants, viewer_variants = {}, []
    try:
        source_glb = Glb(source, entry)
        report, viewer = validate(source_glb)
        variants["source"] = report
        viewer_variants.append({"label": "Original rig" if report["skinned_vertices"] else "Unrigged source", **viewer})
    except (ValueError, KeyError, IndexError) as exc:
        variants["source"] = {"error": str(exc)}
    output = folder / "rigged.glb"
    if args.binary:
        times, hashes = [], []
        for run in range(args.runs):
            target = output if run == 0 else folder / f"repeat-{run}.glb"
            native_report = folder / ("native-report.json" if run == 0 else f"repeat-{run}.json")
            command = [str(args.binary), str(source), str(target), "--report", str(native_report)]
            if args.force:
                command.append("--force")
            for flag in ("up", "forward"):
                value = entry.get(flag, entry.get(f"{flag}_axis"))
                if value:
                    command.extend([f"--{flag}", str(value).lower().removeprefix("+")])
            start = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            times.append((time.perf_counter() - start) * 1000)
            if completed.returncode:
                result.update(error=completed.stderr.strip() or completed.stdout.strip(), command=command)
                return result
            hashes.append(digest(target))
        result["deterministic_bytes"] = len(set(hashes)) == 1
        result["repeat_sha256"] = hashes
        result["runtime_ms"] = {"runs": times, "median_all": statistics.median(times),
                                "median_warm": statistics.median(times[1:]) if len(times) > 1 else None}
        result["native_report"] = json.loads((folder / "native-report.json").read_text())
        for run in range(1, args.runs):
            (folder / f"repeat-{run}.glb").unlink()
            (folder / f"repeat-{run}.json").unlink(missing_ok=True)
    else:
        output = source
    output_glb = Glb(output)
    report, viewer = validate(output_glb)
    if args.binary and "source_glb" in locals():
        proof = geometry_proof(source_glb, output_glb, bool(result.get("native_report", {}).get("covered_legs")))
        report["source_geometry_proof"] = proof
        if not proof["pass"]:
            report["correctness_failures"].extend(proof["failures"])
            report["correctness_pass"] = False
    variants["autorig"] = report
    if args.binary:
        viewer_variants.append({"label": "Deterministic autorig", **viewer})
    else:
        viewer_variants = [{"label": "Existing rig", **viewer}]
    result["variants"] = variants
    result["output"] = str(output)
    result["correctness_pass"] = report["correctness_pass"] and result.get("deterministic_bytes", True)
    result["quality_review_required"] = report["quality_review_required"]
    source_stats = variants.get("source", {})
    if source_stats.get("skinned_vertices") and source_stats.get("correctness_pass"):
        result["reference_edge_stretch_p99"] = {
            pose: {"author_rig": source_stats["poses"][pose]["edge_stretch"]["p99"],
                   "autorig": report["poses"][pose]["edge_stretch"]["p99"]} for pose in POSE_NAMES
        }
    json_write(folder / "metrics.json", result)
    json_write(folder / "viewer.json", {"id": name, "variants": viewer_variants})
    return result


VIEWER_HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Autorig stress evidence</title>
<style>body{margin:0;background:#101318;color:#eef2f7;font:14px system-ui}h1{font-size:21px;margin:20px 24px 4px}
p{margin:0 24px 18px;color:#aeb8c7}.rows{display:grid;grid-template-columns:repeat(6,1fr);gap:2px;padding:0 20px 20px}
.tile{position:relative;background:#181e26}.tile canvas{display:block;width:100%;height:310px}.label{position:absolute;top:10px;left:10px;background:#101318dd;padding:5px 8px;border-radius:4px;font-size:12px}</style>
<script type="importmap">{"imports":{"three":"/ui/vendor/three.module.min.js","three/addons/":"/ui/vendor/addons/"}}</script>
<h1 id="heading">Loading…</h1><p>Actual GLB skinning · independent stress poses · check clothing and joint folds</p><div class="rows" id="rows"></div>
<script type="module" src="./viewer.js"></script></html>"""

VIEWER_JS = """import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/GLTFLoader.js';
const params=new URLSearchParams(location.search), id=params.get('id');
const data=await fetch(`./${id}/viewer.json`).then(r=>r.json());
document.querySelector('#heading').textContent=data.id.replaceAll('-',' ');
const poses=params.has('pose')?[params.get('pose')]:['rest','arms_raised','wide_stance','crouch','torso_twist','asymmetric'];
const detail=params.get('detail')==='1',width=detail?660:220,heightPx=detail?820:310;
if(detail){document.querySelector('#rows').style.gridTemplateColumns='repeat(2,1fr)';}
const loader=new GLTFLoader();
const labels={rest:'Rest',arms_raised:'Arms raised',wide_stance:'Wide stance',crouch:'Deep crouch',torso_twist:'Torso twist',asymmetric:'Asymmetric'};
// One detected frame and one pose-inclusive camera make source/output comparisons fair.
const frame=data.variants.at(-1),up=new THREE.Vector3(...frame.up),lat=new THREE.Vector3(...frame.lateral);
const front=new THREE.Vector3().crossVectors(lat,up),basis=new THREE.Matrix4().makeBasis(lat,up,front);
const displayMatrix=basis.clone().invert(),framingBox=new THREE.Box3();
for(const variant of data.variants)for(const pose of poses){
  const bounds=variant.pose_bounds[pose];
  framingBox.union(new THREE.Box3(new THREE.Vector3(...bounds[0]),new THREE.Vector3(...bounds[1])).applyMatrix4(displayMatrix));
}
window.qaErrors=[];window.qaOracleErrors=[];
for (const variant of data.variants) {
  for (const pose of poses) {
    const tile=document.createElement('div');tile.className='tile';
    const label=document.createElement('div');label.className='label';label.textContent=`${variant.label} · ${labels[pose]}`;
    tile.append(label);document.querySelector('#rows').append(tile);
    const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    renderer.setSize(width,heightPx);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;
    renderer.toneMapping=THREE.ACESFilmicToneMapping;tile.append(renderer.domElement);
    renderer.domElement.style.height=`${heightPx}px`;
    const scene=new THREE.Scene();scene.background=new THREE.Color(0x181e26);
    scene.add(new THREE.HemisphereLight(0xffffff,0x647085,2.7));
    const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,5,4);scene.add(light);
    const gltf=await loader.loadAsync(`/asset?path=${encodeURIComponent(variant.path)}`);
    const root=gltf.scene;scene.add(root);root.updateMatrixWorld(true);
    // Align the anatomical frame only for display; the tested joint matrices remain exact.
    const display=new THREE.Group();scene.remove(root);display.add(root);display.quaternion.setFromRotationMatrix(basis).invert();scene.add(display);
    display.updateMatrixWorld(true);
    const restBox=new THREE.Box3().setFromObject(display);
    const height=restBox.max.y-restBox.min.y;
    const camera=new THREE.PerspectiveCamera(34,width/heightPx,height/1000,height*100);
    const side=params.get('view')==='side';
    const target=framingBox.getCenter(new THREE.Vector3()),extent=framingBox.getSize(new THREE.Vector3());
    const horizontal=side?extent.z:extent.x,depth=side?extent.x:extent.z;
    const distance=1.13*Math.max(extent.y,horizontal/(width/heightPx))/(2*Math.tan(THREE.MathUtils.degToRad(17)))+depth*.5;
    camera.position.copy(target).add(side?new THREE.Vector3(distance,height*.04,0):new THREE.Vector3(0,height*.04,distance));camera.lookAt(target);
    const matrices=variant.poses[pose];
    for(let i=0;i<matrices.length;i++) {
      const node=await gltf.parser.getDependency('node',i);node.matrix.fromArray(matrices[i]);node.matrixAutoUpdate=false;
    }
    root.updateMatrixWorld(true);display.updateMatrixWorld(true);
    root.traverse(obj=>{if(obj.isSkinnedMesh){obj.frustumCulled=false;obj.skeleton.update();}});
    for(const sample of variant.samples?.[pose]??[]) {
      const node=await gltf.parser.getDependency('node',sample.node);
      const mesh=node.isMesh?node:node.children.filter(child=>child.isMesh)[sample.primitive];
      if(!mesh)throw new Error(`Missing sampled mesh ${sample.node}/${sample.primitive}`);
      for(let i=0;i<sample.indices.length;i++) {
        const actual=mesh.getVertexPosition(sample.indices[i],new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
        const expected=new THREE.Vector3(...sample.world_positions[i]).applyMatrix4(display.matrixWorld);
        const error=actual.distanceTo(expected)/variant.height;
        window.qaOracleErrors.push(error);
        if(error>1e-4)throw new Error(`Browser/oracle skin mismatch ${variant.label}/${pose}: ${error}`);
      }
    }
    const ground=new THREE.Mesh(new THREE.PlaneGeometry(height*10,height*10),new THREE.MeshStandardMaterial({color:0x232a33,roughness:1}));
    ground.rotation.x=-Math.PI/2;ground.position.y=restBox.min.y-height*.003;scene.add(ground);
    const grid=new THREE.GridHelper(height*5,20,0x526070,0x323e4e);grid.position.y=restBox.min.y;scene.add(grid);
    renderer.render(scene,camera);
  }
}
window.qaReady=true;
"""


def render_contact_sheets(qa: pathlib.Path, cases: list[dict[str, Any]], timeout: float, selected: list[str] | None = None,
                         details: bool = False, clean_pose_names: dict[str, str] | None = None) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Rendering requires Playwright in this interpreter; use a scratch uv environment") from exc
    (qa / "viewer.html").write_text(VIEWER_HTML, encoding="utf-8")
    (qa / "viewer.js").write_text(VIEWER_JS, encoding="utf-8")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def do_GET(self) -> None:
            from urllib.parse import parse_qs, urlsplit
            parsed = urlsplit(self.path)
            if parsed.path == "/asset":
                asset = pathlib.Path(parse_qs(parsed.query)["path"][0]).resolve()
                allowed = {pathlib.Path(v["path"]).resolve() for case in cases if "output" in case
                           for v in json.loads((qa / case["id"] / "viewer.json").read_text())["variants"]}
                if asset not in allowed:
                    self.send_error(403)
                    return
                blob = asset.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                return
            super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1360, "height": 740}, device_scale_factor=1)
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            for case in cases:
                if "output" not in case:
                    continue
                if selected and case["id"] not in selected:
                    continue
                errors.clear()
                viewer_path = qa / case["id"] / "viewer.json"
                viewer_data = json.loads(viewer_path.read_text())
                for variant in viewer_data["variants"]:
                    if "pose_bounds" not in variant:
                        glb = Glb(pathlib.Path(variant["path"]))
                        variant["pose_bounds"] = {}
                        for pose, matrices in variant["poses"].items():
                            local = np.array(matrices).reshape(-1, 4, 4).transpose(0, 2, 1)
                            world = glb.world_matrices(local)
                            points = np.concatenate([glb.deform(primitive, world) for primitive in glb.primitives])
                            variant["pose_bounds"][pose] = np.stack([points.min(axis=0), points.max(axis=0)]).tolist()
                json_write(viewer_path, viewer_data)
                count = len(viewer_data["variants"])
                page.set_viewport_size({"width": 1360, "height": 106 + count * 312})
                url = f"http://127.0.0.1:{server.server_port}/{qa.relative_to(ROOT).as_posix()}/viewer.html?id={case['id']}"
                page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                page.wait_for_function("window.qaReady === true", timeout=timeout * 1000)
                page.screenshot(path=str(qa / case["id"] / "contact-sheet.png"), full_page=True)
                oracle_errors = page.evaluate("window.qaOracleErrors")
                case["render"] = {"contact_sheet": str(qa / case["id"] / "contact-sheet.png"), "browser_errors": errors.copy(),
                                  "cpu_browser_samples": len(oracle_errors), "cpu_browser_max_error_height": max(oracle_errors, default=0)}
                if details:
                    page.set_viewport_size({"width": 1360, "height": 926})
                    for pose in ("wide_stance", "crouch", "arms_raised"):
                        for view in ("front", "side"):
                            page.goto(url + f"&detail=1&pose={pose}&view={view}", wait_until="networkidle", timeout=timeout * 1000)
                            page.wait_for_function("window.qaReady === true", timeout=timeout * 1000)
                            page.screenshot(path=str(qa / case["id"] / f"detail-{pose}-{view}.png"), full_page=True)
                if clean_pose_names:
                    page.set_viewport_size({"width": 1360, "height": 926})
                    for pose, suffix in clean_pose_names.items():
                        page.goto(url + f"&detail=1&pose={pose}&view=front", wait_until="networkidle", timeout=timeout * 1000)
                        page.wait_for_function("window.qaReady === true", timeout=timeout * 1000)
                        page.add_style_tag(content=".label { display: none; }")
                        for index, variant in enumerate(viewer_data["variants"]):
                            target = qa / f"{variant['proof_name']}-{suffix}.png"
                            page.locator("canvas").nth(index).screenshot(path=str(target))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path, help="Native autorig CLI; omit to inspect existing GLBs")
    parser.add_argument("--force", action="store_true", help="Rebind existing authored rigs for held-out algorithm testing")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "out/autorig-validation/qa")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-case", action="append", help="Render only this case ID; repeat for several cases")
    parser.add_argument("--details", action="store_true", help="Also render larger front/side raised-arm, stance, and crouch views")
    parser.add_argument("--render-existing", action="store_true", help="Render an existing --out/summary.json without rerunning native inference")
    args = parser.parse_args()
    if args.runs < 2 and args.binary:
        parser.error("At least two runs are required to verify deterministic output")
    if not args.manifest and not args.inputs and not args.render_existing:
        parser.error("Provide --manifest or input GLBs")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.render_existing:
        report = json.loads((args.out / "summary.json").read_text())
        render_contact_sheets(args.out, report["cases"], args.timeout, args.render_case, args.details)
        json_write(args.out / "summary.json", report)
        print(f"Rendered {len(report['cases'])} existing cases in {args.out}", flush=True)
        return 0
    if args.binary:
        original_binary = args.binary.resolve()
        binary_hash = digest(original_binary)
        snapshot = args.out / f"native-cli-{binary_hash[:12]}"
        if not snapshot.exists():
            temporary = snapshot.with_suffix(".prepare")
            shutil.copyfile(original_binary, temporary)
            temporary.chmod(original_binary.stat().st_mode)
            temporary.replace(snapshot)
        args.binary = snapshot
    cases = []
    for entry in normalize_entries(args.manifest, args.inputs):
        print(f"Validating {entry['id']}…", flush=True)
        try:
            case = run_case(entry, args, args.out)
        except (ValueError, KeyError, IndexError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            case = {"id": entry["id"], "source": entry["path"], "error": str(exc)}
        cases.append(case)
        print(f"  correctness={case.get('correctness_pass', False)} review={case.get('quality_review_required', True)}"
              f" warm_ms={case.get('runtime_ms', {}).get('median_warm')}", flush=True)
    if args.render:
        render_contact_sheets(args.out, cases, args.timeout, args.render_case, args.details)
    report = {
        "schema_version": 1, "command": [str(args.binary) if args.binary else "inspect", "--force" if args.force else "preserve"],
        "binary_sha256": digest(args.binary) if args.binary else None,
        "correctness_passed": sum(bool(c.get("correctness_pass")) for c in cases), "total_cases": len(cases),
        "quality_review_required": sum(bool(c.get("quality_review_required", True)) for c in cases),
        "cases": cases,
    }
    json_write(args.out / "summary.json", report)
    print(json.dumps({k: report[k] for k in ("correctness_passed", "total_cases", "quality_review_required")}), flush=True)
    return 0 if report["correctness_passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
