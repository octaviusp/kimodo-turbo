"""Automatic humanoid rig for an unskinned GLB, with Kimodo SOMA joint names.

Pipeline: canonical frame (up = +Y, forward from the feet, left = up x forward) -> anatomical
landmarks from horizontal bands of the welded mesh graph (legs merge at the crotch, separated
arms below the armpits, or lateral extremities for T/A poses) -> joint positions -> skin weights
(body-part labels + inverse distance to bone segments, smoothed over mesh adjacency) -> GLB
with a skin. Joint names match Kimodo's output skeleton, so retargeting is a direct mapping.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .glb import Glb, world_matrices

SIDES = {"Left": 1.0, "Right": -1.0}
# (joint, parent, bone end used for skin weights)
RIG = [
    ("Hips", None, "Spine1"),
    ("Spine1", "Hips", "Spine2"),
    ("Spine2", "Spine1", "Chest"),
    ("Chest", "Spine2", "Neck1"),
    ("Neck1", "Chest", "Head"),
    ("Head", "Neck1", "HeadEnd"),
    ("HeadEnd", "Head", None),
]
for _side in SIDES:
    RIG += [
        (f"{_side}Shoulder", "Chest", f"{_side}Arm"),
        (f"{_side}Arm", f"{_side}Shoulder", f"{_side}ForeArm"),
        (f"{_side}ForeArm", f"{_side}Arm", f"{_side}Hand"),
        (f"{_side}Hand", f"{_side}ForeArm", f"{_side}HandEnd"),
        (f"{_side}HandEnd", f"{_side}Hand", None),
        (f"{_side}Leg", "Hips", f"{_side}Shin"),
        (f"{_side}Shin", f"{_side}Leg", f"{_side}Foot"),
        (f"{_side}Foot", f"{_side}Shin", f"{_side}ToeBase"),
        (f"{_side}ToeBase", f"{_side}Foot", f"{_side}ToeEnd"),
        (f"{_side}ToeEnd", f"{_side}ToeBase", None),
    ]
JOINT_NAMES = [name for name, _, _ in RIG]
PARENT = {name: parent for name, parent, _ in RIG}
BONE_END = {name: end for name, _, end in RIG if end is not None}
TORSO_BONES = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head", "LeftShoulder", "RightShoulder", "LeftLeg", "RightLeg"]
BAND, STEP = 0.03, 0.01  # band thickness and step, as fractions of height


@dataclass
class Body:
    """Welded mesh in canonical, height-normalized coordinates: u (up), l (left), f (forward)."""

    ulf: np.ndarray
    edges: np.ndarray
    origin: np.ndarray
    axes: np.ndarray  # rows: up, left, forward (world unit vectors)
    height: float
    welded_index: np.ndarray  # original vertex -> welded vertex
    labels: dict = field(default_factory=dict)

    def to_world(self, ulf: np.ndarray) -> np.ndarray:
        return self.origin + self.height * (np.asarray(ulf) @ self.axes)

    def pieces(self) -> list[np.ndarray]:
        """Disconnected mesh pieces (body, separate hands/shoes/eyes...), largest first."""
        n, e = len(self.ulf), self.edges
        count, labels = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
        return sorted((np.flatnonzero(labels == k) for k in range(count)), key=len, reverse=True)

    def components(self, u0: float, u1: float) -> list[np.ndarray]:
        """Connected pieces of the mesh cross-section between heights u0 and u1.

        Uses every edge that overlaps the band (not only edges inside it), so long low-poly
        triangles still connect their cross-section; vertices are the endpoints of those edges.
        """
        eu = self.ulf[self.edges, 0]
        keep = (eu.min(1) <= u1) & (eu.max(1) >= u0)
        e = self.edges[keep]
        if len(e) == 0:
            return []
        index, local = np.unique(e, return_inverse=True)
        local = local.reshape(-1, 2)
        graph = coo_matrix((np.ones(len(local)), (local[:, 0], local[:, 1])), shape=(index.size, index.size))
        count, labels = connected_components(graph, directed=False)
        groups = [index[labels == k] for k in range(count)]
        return sorted((g for g in groups if g.size >= 3), key=len, reverse=True)


def load_mesh(glb: Glb) -> tuple[np.ndarray, np.ndarray, list]:
    """World-space positions and triangles of every mesh primitive, plus per-primitive vertex ranges."""
    gltf = glb.gltf
    worlds = world_matrices(gltf)
    positions, triangles, ranges, offset = [], [], [], 0
    for node_index, world in worlds.items():
        node = gltf["nodes"][node_index]
        if "mesh" not in node:
            continue
        for prim_index, prim in enumerate(gltf["meshes"][node["mesh"]]["primitives"]):
            if prim.get("mode", 4) != 4:
                continue
            pos = glb.read_accessor(prim["attributes"]["POSITION"]).astype(np.float64)
            pos = pos @ world[:3, :3].T + world[:3, 3]
            idx = glb.read_accessor(prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(pos))
            positions.append(pos)
            triangles.append(idx.reshape(-1, 3).astype(np.int64) + offset)
            ranges.append((node_index, prim_index, offset, len(pos)))
            offset += len(pos)
    if not positions:
        raise ValueError("GLB has no triangle meshes")
    return np.concatenate(positions), np.concatenate(triangles), ranges


def canonical_body(positions: np.ndarray, triangles: np.ndarray) -> Body:
    lo, hi = positions.min(0), positions.max(0)
    height = hi[1] - lo[1]
    key = np.round((positions - lo) / (height * 1e-4)).astype(np.int64)
    _, welded_index, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    welded = positions[welded_index]
    tri = inverse[triangles]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    up = np.array([0.0, 1.0, 0.0])
    extent = hi - lo
    lateral_axis, forward_axis = (0, 2) if extent[0] >= extent[2] else (2, 0)
    origin = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    u = (welded[:, 1] - lo[1]) / height
    f_raw = (welded[:, forward_axis] - origin[forward_axis]) / height
    forward = np.zeros(3)
    forward[forward_axis] = _facing_sign(u, f_raw, (welded[:, lateral_axis] - origin[lateral_axis]) / height)
    left = np.cross(up, forward)
    axes = np.stack([up, left, forward])
    ulf = (welded - origin) @ axes.T / height
    return Body(ulf, edges, origin, axes, height, inverse)


def _facing_sign(u: np.ndarray, f: np.ndarray, l: np.ndarray) -> float:
    """+1 if the character faces +f. The face's extreme is narrow (nose) while the back of the
    head is broad (skull, hair); feet are the tie-breaker (toes reach further from the body axis)."""
    head = (u > 0.84) & (u < 0.95)
    tip = 0.015
    front = np.ptp(l[head & (f > f[head].max() - tip)])
    back = np.ptp(l[head & (f < f[head].min() + tip)])
    if max(front, back) > 1.2 * min(front, back):
        return 1.0 if front < back else -1.0
    soles, pelvis = u < 0.025, (u > 0.45) & (u < 0.6)
    axis = np.median(f[pelvis])
    return 1.0 if f[soles].max() - axis >= axis - f[soles].min() else -1.0


def _centroid(body: Body, verts: np.ndarray) -> np.ndarray:
    return body.ulf[verts].mean(0)


def _track(body: Body, start: np.ndarray, u_from: float, direction: int, stop) -> list[tuple[float, np.ndarray]]:
    """Follow a body part band by band (bands overlap, so continuity = shared vertices)."""
    track, current, u = [], start, u_from
    while 0.0 <= u <= 1.0:
        u += direction * STEP
        nxt = next((c for c in body.components(u, u + BAND) if np.intersect1d(c, current).size), None)
        if nxt is None or stop(nxt, u):
            break
        track.append((u + BAND / 2, nxt))
        current = nxt
    return track


def find_legs(body: Body):
    """Track both legs upward from the shins; the crotch is where they join one band component."""
    start = body.components(0.08, 0.08 + BAND)[:2]
    if len(start) < 2:
        raise ValueError("Could not find two legs near the floor; is the model a standing humanoid?")
    start.sort(key=lambda c: -_centroid(body, c)[1])  # left (+l) first
    current = dict(zip(SIDES, start))
    tracks = {side: [(0.08 + BAND / 2, comp)] for side, comp in current.items()}
    u = 0.08
    while u < 0.75:
        u += STEP
        comps = body.components(u, u + BAND)
        nxt = {s: next((c for c in comps if np.intersect1d(c, current[s]).size), None) for s in SIDES}
        if any(c is None for c in nxt.values()):
            break
        if nxt["Left"] is nxt["Right"]:
            return tracks, u + BAND - STEP / 2
        for side, comp in nxt.items():
            tracks[side].append((u + BAND / 2, comp))
        current = nxt
    raise ValueError("Could not find where the legs join (crotch); is the model a standing humanoid?")


def _torso_component(body: Body, u: float):
    comps = body.components(u, u + BAND)
    return min(comps, key=lambda c: abs(_centroid(body, c)[1])) if comps else None


def find_arms(body: Body, crotch: float):
    """Separated hanging arms: outer band components below the armpit, tracked down to the fingertips."""
    for u in np.arange(0.86, crotch, -STEP):
        comps = body.components(u, u + BAND)
        torso = _torso_component(body, u)
        sides = {}
        for side, sign in SIDES.items():
            outer = [c for c in comps if c is not torso and sign * _centroid(body, c)[1] > 0.04 and len(c) >= 6]
            if outer:
                sides[side] = max(outer, key=len)
        if len(sides) == 2:
            arms = {}
            for side, comp in sides.items():
                sign = SIDES[side]
                # Stop if the hand merges with the thigh/clothing: the section drifts to the
                # midline or becomes much wider than an arm.
                track = [(u + BAND / 2, comp)] + _track(
                    body, comp, u, -1,
                    stop=lambda c, _u, sign=sign: sign * _centroid(body, c)[1] < 0.03 or np.ptp(body.ulf[c, 1]) > 0.15,
                )
                arms[side] = track
            if min(t[0][0] - t[-1][0] for t in arms.values()) > 0.12:
                return arms, u + BAND
    return None, None


def _polyline_point(points: np.ndarray, fraction: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    target = fraction * cum[-1]
    i = min(np.searchsorted(cum, target, side="right") - 1, len(seg) - 1)
    t = 0 if seg[i] == 0 else (target - cum[i]) / seg[i]
    return points[i] + t * (points[i + 1] - points[i])


def _arm_chain(shoulder: np.ndarray, centerline: np.ndarray) -> dict:
    """Shoulder->fingertip polyline -> (arm, forearm, hand, hand end) using Kimodo proportions."""
    chain = np.vstack([shoulder, centerline])
    return {
        "Arm": shoulder,
        "ForeArm": _polyline_point(chain, 0.39),
        "Hand": _polyline_point(chain, 0.75),
        "HandEnd": chain[-1],
    }


COVERED_CROTCH = 0.47  # typical crotch height / body height, used when clothing hides the legs
HIP_HALF_WIDTH = 0.057  # Kimodo hip-joint half spacing / body height


def _foot(body: Body, sign: float) -> tuple[np.ndarray, float, float, float]:
    foot = np.flatnonzero((body.ulf[:, 0] < 0.05) & (sign * body.ulf[:, 1] > 0.01))
    if foot.size < 3:
        raise ValueError("Could not find the feet near the floor")
    f = body.ulf[foot, 2]
    return foot, f.min(), f.max(), float(np.median(body.ulf[foot, 1]))


def locate_legs(body: Body, joints: dict, parts: dict) -> tuple[float, str]:
    """Leg joints. Separated legs are tracked up to the crotch; legs hidden by long clothing get
    anthropometric hips above the visible feet, and the clothing is weighted to both legs."""
    try:
        legs, crotch = find_legs(body)
        method = "separated"
    except ValueError:
        legs, crotch, method = None, COVERED_CROTCH, "covered"
    leg_joint_u = crotch + 0.035
    hips_u = leg_joint_u + 0.048
    torso = _torso_component(body, hips_u - BAND / 2)
    body_f = _centroid(body, torso)[2] if torso is not None else 0.0
    for side, sign in SIDES.items():
        foot, heel, toe, foot_l = _foot(body, sign)
        joints[f"{side}ToeBase"] = np.array([0.015, foot_l, heel + 0.72 * (toe - heel)])
        joints[f"{side}ToeEnd"] = np.array([0.01, foot_l, toe])
        if legs is not None:
            cl = np.array([_centroid(body, c) for _, c in legs[side]])
            at = lambda u: cl[np.argmin(np.abs(cl[:, 0] - u))]
            joints[f"{side}Leg"] = np.array([leg_joint_u, at(crotch - 0.04)[1], at(crotch - 0.04)[2]])
            joints[f"{side}Foot"] = np.array([0.05, at(0.09)[1], at(0.09)[2]])
            knee_u = (leg_joint_u + 0.05) / 2
            joints[f"{side}Shin"] = np.array([knee_u, at(knee_u)[1], at(knee_u)[2] + 0.005])
            parts[f"leg_{side}"] = np.unique(np.concatenate([c for _, c in legs[side]]))
        else:
            ankle = np.array([0.05, foot_l, heel + 0.3 * (toe - heel)])
            hip = np.array([leg_joint_u, sign * HIP_HALF_WIDTH, body_f])
            joints[f"{side}Foot"] = ankle
            joints[f"{side}Leg"] = hip
            joints[f"{side}Shin"] = (hip + ankle) / 2 + np.array([0.0, 0.0, 0.01])
            # Clothing below the hips follows the leg on its side; the midline blends both.
            parts[f"leg_{side}"] = np.flatnonzero((body.ulf[:, 0] < leg_joint_u) & (sign * body.ulf[:, 1] > -0.02))
    joints["Hips"] = np.array([hips_u, 0.0, (joints["LeftLeg"][2] + joints["RightLeg"][2]) / 2])
    return crotch, method


def locate_joints(body: Body) -> tuple[dict, dict, dict]:
    joints, parts = {}, {}
    crotch, leg_method = locate_legs(body, joints, parts)
    pieces = body.pieces()
    detached = pieces[1:]

    # Neck: narrowest torso band in the upper body.
    widths = []
    for u in np.arange(0.74, 0.93, STEP / 2):
        torso = _torso_component(body, u)
        if torso is not None:
            widths.append((np.ptp(body.ulf[torso, 1]), u, _centroid(body, torso)))
    _, neck_u, neck_c = min(widths, key=lambda w: w[0])
    top = body.ulf[:, 0].max()

    arms, armpit = find_arms(body, crotch)
    arm_method = "separated" if arms is not None else "lateral"
    if arms is not None:
        shoulder_u = min(armpit + 0.055, neck_u - 0.02)
        for side, track in arms.items():
            centers = np.array([_centroid(body, c) for _, c in track])
            end = centers[-1]
            # Separate hand meshes just past the sleeve extend the arm to the real fingertip.
            hands = [p for p in detached if np.linalg.norm(_centroid(body, p) - end) < 0.08]
            tip_verts = np.concatenate([track[-1][1], *hands])
            tip = body.ulf[tip_verts[np.argmin(body.ulf[tip_verts, 0])]]
            line = [centers] + [_centroid(body, h)[None] for h in hands] + [tip[None]]
            top_c = centers[0]
            shoulder = np.array([shoulder_u, top_c[1] - SIDES[side] * 0.01, top_c[2]])
            chain = _arm_chain(shoulder, np.vstack(line))
            for key, value in chain.items():
                joints[f"{side}{key}"] = value
            region = np.concatenate([c for _, c in track] + hands)
            deltoid = np.flatnonzero(
                (body.ulf[:, 0] >= armpit) & (body.ulf[:, 0] < shoulder_u + 0.05)
                & (SIDES[side] * body.ulf[:, 1] > SIDES[side] * shoulder[1] - 0.025)
            )
            parts[f"arm_{side}"] = np.unique(np.concatenate([region, deltoid]))
    else:
        # T/A pose: arms are whatever lies laterally beyond the chest outline above the crotch.
        chest = _torso_component(body, neck_u - 0.12)
        half_width = np.abs(body.ulf[chest, 1]).max() if chest is not None else 0.12
        shoulder_u = neck_u - 0.06
        for side, sign in SIDES.items():
            verts = np.flatnonzero((sign * body.ulf[:, 1] > half_width * 0.85) & (body.ulf[:, 0] > crotch + 0.05))
            if verts.size < 10:
                raise ValueError("Could not locate the arms (supported: arms hanging apart from the body, T-pose, A-pose)")
            shoulder = np.array([shoulder_u, sign * half_width * 0.8, neck_c[2]])
            dist = np.linalg.norm(body.ulf[verts] - shoulder, axis=1)
            tip = body.ulf[verts[np.argmax(dist)]]
            direction = (tip - shoulder) / np.linalg.norm(tip - shoulder)
            t = (body.ulf[verts] - shoulder) @ direction
            bins = np.linspace(0, t.max(), 12)
            centers = [body.ulf[verts[(t >= a) & (t < b)]].mean(0) for a, b in zip(bins[:-1], bins[1:]) if ((t >= a) & (t < b)).any()]
            chain = _arm_chain(shoulder, np.vstack(centers + [tip]))
            for key, value in chain.items():
                joints[f"{side}{key}"] = value
            parts[f"arm_{side}"] = verts

    for side, sign in SIDES.items():
        arm = joints[f"{side}Arm"]
        joints[f"{side}Shoulder"] = np.array([arm[0] - 0.005, sign * 0.012, neck_c[2]])

    neck1_u = max(joints["LeftArm"][0] + 0.015, neck_u - 0.03)
    for name, frac in (("Spine1", 0.11), ("Spine2", 0.26), ("Chest", 0.43)):
        u = joints["Hips"][0] + frac * (neck1_u - joints["Hips"][0])
        torso = _torso_component(body, u - BAND / 2)
        joints[name] = np.array([u, 0.0, _centroid(body, torso)[2] if torso is not None else 0.0])
    joints["Neck1"] = np.array([neck1_u, 0.0, neck_c[2]])
    head_u = neck1_u + 0.46 * (top - neck1_u)
    head = _torso_component(body, head_u - BAND / 2)
    head_f = _centroid(body, head)[2] if head is not None else neck_c[2]
    joints["Head"] = np.array([head_u, 0.0, head_f])
    joints["HeadEnd"] = np.array([top - 0.005, 0.0, head_f])
    # Fingertips just past the tracked sleeve/hand still belong to the hand.
    for side, sign in SIDES.items():
        near_hand = _segment_distance(body.ulf, joints[f"{side}Hand"], joints[f"{side}HandEnd"]) < 0.04
        parts[f"arm_{side}"] = np.union1d(parts[f"arm_{side}"], np.flatnonzero(near_hand & (sign * body.ulf[:, 1] > 0.06)))
    _grow_arm_labels(body, joints, parts)
    # Hands can hang below the hips: arm labels win over (clothing) leg labels.
    arm_verts = np.union1d(parts["arm_Left"], parts["arm_Right"])
    for side in SIDES:
        parts[f"leg_{side}"] = np.setdiff1d(parts[f"leg_{side}"], arm_verts)

    _label_detached(body, parts, detached)
    return joints, parts, {"legs": leg_method, "arms": arm_method, "pieces": len(pieces)}


def _grow_arm_labels(body: Body, joints: dict, parts: dict) -> None:
    """Flood the arm label over mesh edges into vertices whose nearest bone is on that arm
    (spread fingers, thumbs, cuffs the band tracking missed). Without this, such a vertex keeps
    torso weights and its triangles stretch from the hand back to the rest pose."""
    bones = list(BONE_END)
    dist = np.stack([_segment_distance(body.ulf, joints[b], joints[BONE_END[b]]) for b in bones], axis=1)
    nearest = np.array(bones)[dist.argmin(1)]
    e = np.concatenate([body.edges, body.edges[:, ::-1]])
    for side in SIDES:
        arm_bones = {f"{side}Arm", f"{side}ForeArm", f"{side}Hand"}
        eligible = np.isin(nearest, list(arm_bones))
        label = np.zeros(len(body.ulf), dtype=bool)
        label[parts[f"arm_{side}"]] = True
        while True:
            grow = np.zeros_like(label)
            grow[e[label[e[:, 0]] & ~label[e[:, 1]] & eligible[e[:, 1]], 1]] = True
            if not grow.any():
                break
            label |= grow
        parts[f"arm_{side}"] = np.flatnonzero(label)


def _label_detached(body: Body, parts: dict, detached: list[np.ndarray]) -> None:
    """A loose piece (shoe, eye, hand...) joins the body part of the nearest attached vertex."""
    owner = {}
    for name, verts in parts.items():
        owner.update(dict.fromkeys(verts.tolist(), name))
    attached = np.setdiff1d(np.arange(len(body.ulf)), np.concatenate(detached) if detached else [])
    for piece in detached:
        if any(np.isin(piece, verts).all() for verts in parts.values()):
            continue
        centre = _centroid(body, piece)
        nearest = attached[np.argmin(np.linalg.norm(body.ulf[attached] - centre, axis=1))]
        part = owner.get(int(nearest))
        if part is not None:
            parts[part] = np.union1d(parts[part], piece)


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip((points - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def skin_weights(body: Body, joints: dict, parts: dict) -> tuple[np.ndarray, np.ndarray]:
    bones = list(BONE_END)
    bone_index = {name: JOINT_NAMES.index(name) for name in bones}
    distances = np.stack([_segment_distance(body.ulf, joints[b], joints[BONE_END[b]]) for b in bones], axis=1)

    allowed = np.zeros((len(body.ulf), len(bones)), dtype=bool)
    labelled = np.zeros(len(body.ulf), dtype=bool)
    for side in SIDES:
        for key, names in (
            (f"arm_{side}", ["Chest", f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand"]),
            (f"leg_{side}", ["Hips", f"{side}Leg", f"{side}Shin", f"{side}Foot", f"{side}ToeBase"]),
        ):
            verts = parts.get(key, np.array([], dtype=np.int64))
            for name in names:
                allowed[verts, bones.index(name)] = True
            labelled[verts] = True
    for name in TORSO_BONES:
        allowed[~labelled, bones.index(name)] = True

    weights = np.where(allowed, 1.0 / (distances + 0.01) ** 4, 0.0)
    weights /= weights.sum(1, keepdims=True)
    # Laplacian smoothing over the welded mesh graph softens joint transitions without leaking
    # across parts that are not connected (e.g. hanging arm vs. torso).
    n = len(body.ulf)
    e = body.edges
    adjacency = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    degree = np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1)
    for _ in range(4):
        weights = 0.5 * weights + 0.5 * (adjacency @ weights) / degree[:, None]
        # Smoothing must not leak bones into parts that may not use them (hand <- leg).
        weights = np.where(allowed, weights, 0.0)
        weights /= weights.sum(1, keepdims=True)
    top4 = np.argsort(-weights, axis=1)[:, :4]
    top_w = np.take_along_axis(weights, top4, axis=1)
    top_w /= top_w.sum(1, keepdims=True)
    joint_ids = np.vectorize(lambda k: bone_index[bones[k]])(top4)
    return joint_ids.astype(np.uint16), top_w.astype(np.float32)


def autorig(data: bytes) -> tuple[bytes, dict]:
    """Return (rigged GLB bytes, report). Raises ValueError with a user-facing message on failure."""
    glb = Glb.parse(data)
    if glb.gltf.get("skins"):
        raise ValueError("Model is already rigged")
    positions, triangles, ranges = load_mesh(glb)
    body = canonical_body(positions, triangles)
    joints_ulf, parts, detection = locate_joints(body)
    joints_world = {name: body.to_world(joints_ulf[name]) for name in JOINT_NAMES}
    joint_ids, weights = skin_weights(body, joints_ulf, parts)
    gltf = glb.gltf

    # Skinned vertices are expressed in world space: bake node transforms into the primitives.
    worlds = world_matrices(gltf)
    for node_index, prim_index, offset, count in ranges:
        node = gltf["nodes"][node_index]
        prim = gltf["meshes"][node["mesh"]]["primitives"][prim_index]
        world = worlds[node_index]
        if not np.allclose(world, np.eye(4)):
            prim["attributes"]["POSITION"] = glb.add_accessor(positions[offset : offset + count], 5126, "VEC3", 34962)
            if "NORMAL" in prim["attributes"]:
                normals = glb.read_accessor(prim["attributes"]["NORMAL"]) @ np.linalg.inv(world[:3, :3])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
                prim["attributes"]["NORMAL"] = glb.add_accessor(normals, 5126, "VEC3", 34962)
            prim["attributes"].pop("TANGENT", None)
        welded = body.welded_index[offset : offset + count]
        prim["attributes"]["JOINTS_0"] = glb.add_accessor(joint_ids[welded], 5123, "VEC4", 34962)
        prim["attributes"]["WEIGHTS_0"] = glb.add_accessor(weights[welded], 5126, "VEC4", 34962)

    first = len(gltf["nodes"])
    node_of = {name: first + i for i, name in enumerate(JOINT_NAMES)}
    for name in JOINT_NAMES:
        parent = PARENT[name]
        local = joints_world[name] - (joints_world[parent] if parent else 0.0)
        gltf["nodes"].append({"name": name, "translation": [float(v) for v in local]})
    for name in JOINT_NAMES:
        children = [node_of[c] for c in JOINT_NAMES if PARENT[c] == name]
        if children:
            gltf["nodes"][node_of[name]]["children"] = children
    inverse_bind = np.stack([np.linalg.inv(_translation(joints_world[n])).T.reshape(-1) for n in JOINT_NAMES])
    gltf.setdefault("skins", []).append({
        "name": "kimodo-autorig",
        "joints": [node_of[n] for n in JOINT_NAMES],
        "skeleton": node_of["Hips"],
        "inverseBindMatrices": glb.add_accessor(inverse_bind, 5126, "MAT4"),
    })
    skin_index = len(gltf["skins"]) - 1
    scene = gltf["scenes"][gltf.get("scene", 0)]
    mesh_nodes = sorted({r[0] for r in ranges})
    for node_index in mesh_nodes:
        node = gltf["nodes"][node_index]
        for key in ("matrix", "translation", "rotation", "scale"):
            node.pop(key, None)
        node["skin"] = skin_index
    _reparent_to_scene(gltf, mesh_nodes)
    scene["nodes"] = list(dict.fromkeys(scene["nodes"] + mesh_nodes + [node_of["Hips"]]))

    report = {
        "rig": "auto",
        "height": body.height,
        "up": body.axes[0].tolist(),
        "forward": body.axes[2].tolist(),
        "detection": detection,
        "joints": {n: joints_world[n].tolist() for n in JOINT_NAMES},
    }
    return glb.to_bytes(), report


def _translation(t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def _reparent_to_scene(gltf: dict, nodes: list[int]) -> None:
    """Detach baked mesh nodes from parents so no transform applies twice."""
    for node in gltf["nodes"]:
        if "children" in node:
            node["children"] = [c for c in node["children"] if c not in nodes]
            if not node["children"]:
                del node["children"]
