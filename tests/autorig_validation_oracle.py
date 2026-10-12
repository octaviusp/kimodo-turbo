"""Reference-oracle checks with analytically known skinning coordinates."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/validate_autorig.py"
SPEC = importlib.util.spec_from_file_location("autorig_validation", SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def fixture(path: pathlib.Path, weight_scale: float = 1, bind_offset: float = 0, frozen: bool = False) -> None:
    nodes = [
        {"mesh": 0, "skin": 0},
        {"name": "Hips", "translation": [0, 1, 0], "children": [2, 7, 8]},
        {"name": "Spine1", "translation": [0, 0.3, 0], "children": [3, 5]},
        {"name": "LeftArm", "translation": [0.2, 0, 0], "children": [4]},
        {"name": "LeftForeArm", "translation": [0.4, 0, 0]},
        {"name": "RightArm", "translation": [-0.2, 0, 0], "children": [6]},
        {"name": "RightForeArm", "translation": [-0.4, 0, 0]},
        {"name": "LeftLeg", "translation": [0.1, -0.1, 0]},
        {"name": "RightLeg", "translation": [-0.1, -0.1, 0]},
    ]
    positions = np.array([[0.7, 1.3, 0], [0.6, 1.4, 0], [0.6, 1.3, 0.1],
                          [0, 0, 0], [0, 2, 0], [-0.7, 1.3, 0]], dtype="<f4")
    joints = [3, 1, 4, 5, 6, 7, 8]
    worlds = {
        3: [0.2, 1.3, 0], 1: [0, 1, 0], 4: [0.6, 1.3, 0],
        5: [-0.2, 1.3, 0], 6: [-0.6, 1.3, 0], 7: [0.1, 0.9, 0], 8: [-0.1, 0.9, 0],
    }
    inverse_bind = np.tile(np.eye(4), (len(joints), 1, 1))
    for i, joint in enumerate(joints):
        inverse_bind[i, :3, 3] = -np.array(worlds[joint])
    inverse_bind[:, 0, 3] += bind_offset
    arrays = [positions, np.array([0, 1, 2, 3, 4, 5], dtype="<u2"),
              np.full((6, 4), 1 if frozen else 0, dtype="<u2"), np.array([[weight_scale, 0, 0, 0]] * 6, dtype="<f4"),
              inverse_bind.transpose(0, 2, 1).astype("<f4")]
    binary = bytearray()
    views, accessors = [], []
    for array, component, kind, count in zip(arrays, [5126, 5123, 5123, 5126, 5126],
                                            ["VEC3", "SCALAR", "VEC4", "VEC4", "MAT4"], [6, 6, 6, 6, 7], strict=True):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(array.tobytes())
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes})
        accessors.append({"bufferView": len(views) - 1, "componentType": component, "type": kind, "count": count})
    data = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}], "nodes": nodes,
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1}]}],
            "skins": [{"joints": joints, "inverseBindMatrices": 4}], "accessors": accessors,
            "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    encoded = json.dumps(data).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, total)
                     + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


class SkinningOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory(prefix="autorig-oracle-")
        self.path = pathlib.Path(self.scratch.name) / "analytic.glb"

    def tearDown(self) -> None:
        self.scratch.cleanup()

    def test_rest_bind_recovers_mesh(self) -> None:
        fixture(self.path)
        glb = ORACLE.Glb(self.path)
        actual = glb.deform(glb.primitives[0], glb.world)
        np.testing.assert_allclose(actual, glb.primitives[0]["rest"], atol=1e-7)
        report, _ = ORACLE.validate(glb)
        self.assertTrue(report["correctness_pass"])

    def test_known_ninety_degree_joint_rotation(self) -> None:
        fixture(self.path)
        glb = ORACLE.Glb(self.path)
        local = glb.local.copy()
        delta = np.eye(4)
        delta[:3, :3] = ORACLE.rotation(np.array([0, 0, 1]), 90)
        local[glb.names["LeftArm"]] = local[glb.names["LeftArm"]] @ delta
        actual = glb.deform(glb.primitives[0], glb.world_matrices(local))
        np.testing.assert_allclose(actual[0], [0.2, 1.8, 0], atol=1e-7)
        self.assertGreater(np.linalg.norm(actual[0] - glb.primitives[0]["rest"][0]), 0.6)

    def test_bad_bind_matrix_is_rejected(self) -> None:
        fixture(self.path, bind_offset=0.1)
        report, _ = ORACLE.validate(ORACLE.Glb(self.path))
        self.assertFalse(report["correctness_pass"])
        self.assertTrue(any("Rest pose" in message for message in report["correctness_failures"]))

    def test_bad_weight_sum_is_rejected(self) -> None:
        fixture(self.path, weight_scale=0.7)
        report, _ = ORACLE.validate(ORACLE.Glb(self.path))
        self.assertFalse(report["correctness_pass"])
        self.assertAlmostEqual(report["weight_sum_max_error"], 0.3, places=6)

    def test_extreme_poses_move_character(self) -> None:
        fixture(self.path)
        report, _ = ORACLE.validate(ORACLE.Glb(self.path))
        self.assertGreater(report["poses"]["arms_raised"]["movement_p95_height"], 0.1)

    def test_stationary_binding_cannot_hide_quality_failure(self) -> None:
        fixture(self.path, frozen=True)
        report, _ = ORACLE.validate(ORACLE.Glb(self.path))
        self.assertTrue(report["correctness_pass"])
        self.assertTrue(report["quality_review_required"])
        self.assertTrue(any("arms_raised: the character remains" in flag for flag in report["quality_flags"]))

    def seam_fixture(self):
        fixture(self.path)
        source = ORACLE.Glb(self.path)
        p = source.primitives[0]
        p["triangles"][1, 0] = 0
        p["texcoords"]["TEXCOORD_0"] = np.arange(12, dtype=float).reshape(6, 2) / 12
        p["normals"] = np.tile([0, 0, 1], (6, 1)).astype(float)
        output = copy.deepcopy(source)
        q = output.primitives[0]
        q["rest"] = np.concatenate([q["rest"], q["rest"][:1]])
        q["texcoords"]["TEXCOORD_0"] = np.concatenate([q["texcoords"]["TEXCOORD_0"], q["texcoords"]["TEXCOORD_0"][:1]])
        q["normals"] = np.concatenate([q["normals"], q["normals"][:1]])
        q["triangles"][1, 0] = 6
        return source, output

    def test_seam_split_preserves_face_corners_and_uv(self) -> None:
        source, output = self.seam_fixture()
        proof = ORACLE.geometry_proof(source, output, True)
        self.assertTrue(proof["pass"])
        self.assertEqual(proof["duplicated_seam_vertices"], 1)
        self.assertEqual(proof["max_face_corner_error_height"], 0)
        self.assertFalse(ORACLE.geometry_proof(source, output, False)["pass"])

    def test_seam_split_cannot_change_texture_coordinates(self) -> None:
        source, output = self.seam_fixture()
        output.primitives[0]["texcoords"]["TEXCOORD_0"][6, 0] += 0.1
        proof = ORACLE.geometry_proof(source, output, True)
        self.assertFalse(proof["pass"])
        self.assertIn("UV values changed at face corners", proof["failures"])

    def test_seam_split_cannot_change_normal_directions(self) -> None:
        source, output = self.seam_fixture()
        output.primitives[0]["normals"][6] = [0, 1, 0]
        proof = ORACLE.geometry_proof(source, output, True)
        self.assertFalse(proof["pass"])
        self.assertIn("World-space normal directions changed at face corners", proof["failures"])

    def test_seam_split_cannot_change_geometry_or_remove_faces(self) -> None:
        source, output = self.seam_fixture()
        output.primitives[0]["rest"][6, 0] += 0.1
        self.assertFalse(ORACLE.geometry_proof(source, output, True)["pass"])
        output.primitives[0]["triangles"] = output.primitives[0]["triangles"][:1]
        proof = ORACLE.geometry_proof(source, output, True)
        self.assertIn("Triangle count changed", proof["failures"])

    def test_unused_duplicate_is_not_a_seam(self) -> None:
        source, output = self.seam_fixture()
        output.primitives[0]["triangles"][0, 0] = 6
        proof = ORACLE.geometry_proof(source, output, True)
        self.assertFalse(proof["pass"])
        self.assertIn("A duplicated source vertex has no retained incident faces", proof["failures"])


if __name__ == "__main__":
    unittest.main()
