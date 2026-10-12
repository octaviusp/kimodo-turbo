import * as THREE from 'three';

// Kimodo motion -> the bundled character skeleton.
//
// Kimodo's global joint rotations are relative to its T-pose rest (identity = rest). Calibration
// builds the source pose that matches the target's own rest pose: the hips frame (up + hip
// line) is aligned, then each bone is swung so its direction matches the target bone. With A
// the source rotation in that matched pose, each target bone gets a constant offset
// C = A^-1 * W_rest, and at runtime its world rotation is W(t) = Q_src(t) * C. This keeps
// twist, works for T-pose, A-pose or arms-down rigs, and needs no per-rig tables.

// Source joint -> candidate children that define its bone direction (first mapped one wins).
const DIRECTION = {
  Spine1: ['Spine2', 'Chest', 'Neck1'], Spine2: ['Chest', 'Neck1'], Chest: ['Neck1', 'Head'],
  Neck1: ['Head'], Head: ['HeadEnd'],
};
for (const s of ['Left', 'Right']) {
  Object.assign(DIRECTION, {
    [`${s}Shoulder`]: [`${s}Arm`], [`${s}Arm`]: [`${s}ForeArm`], [`${s}ForeArm`]: [`${s}Hand`],
    [`${s}Hand`]: [`${s}HandMiddle1`], [`${s}Leg`]: [`${s}Shin`], [`${s}Shin`]: [`${s}Foot`],
    [`${s}Foot`]: [`${s}ToeBase`, `${s}ToeEnd`], [`${s}ToeBase`]: [`${s}ToeEnd`],
  });
}

const v = (a, i) => new THREE.Vector3(a[3 * i], a[3 * i + 1], a[3 * i + 2]);

function frameQuaternion(up, lateral) {
  const x = lateral.clone().normalize();
  const y = up.clone().sub(x.clone().multiplyScalar(up.dot(x))).normalize();
  const z = new THREE.Vector3().crossVectors(x, y);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
}

export function decodeMotion(json) {
  const decode = (b64) => new Float32Array(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer);
  const motion = { ...json, positions: decode(json.positions), rotations: decode(json.rotations) };
  motion.index = Object.fromEntries(json.joint_names.map((n, i) => [n, i]));
  motion.jointCount = json.joint_names.length;
  return motion;
}

export class Retargeter {
  /** root: loaded glTF scene; bones: all skeleton bones parents-first; mapping: Map(bone -> joint). */
  constructor(motion, root, bones, mapping) {
    this.motion = motion;
    this.root = root;
    this.bones = bones;
    this.mapping = mapping;
    this.calibrate();
  }

  calibrate() {
    const { motion, mapping } = this;
    const J = motion.index;
    const rest = motion.rest;
    this.restLocal = new Map(this.bones.map((b) => [b, { q: b.quaternion.clone(), p: b.position.clone() }]));
    this.root.updateMatrixWorld(true);
    const restWorldQ = new Map(this.bones.map((b) => [b, b.getWorldQuaternion(new THREE.Quaternion())]));
    const restWorldP = new Map(this.bones.map((b) => [b, b.getWorldPosition(new THREE.Vector3())]));
    const boneOf = new Map([...mapping].map(([bone, joint]) => [joint, bone]));
    this.boneOf = boneOf;
    const hips = boneOf.get('Hips');
    if (!hips) throw new Error('The character must contain a Hips bone.');

    // Hips frame: up = towards the highest mapped spine joint, lateral = left minus right.
    const top = ['Chest', 'Spine2', 'Spine1', 'Neck1', 'Head'].find((j) => boneOf.has(j));
    const pair = ['Leg', 'Arm', 'Shoulder'].find((p) => boneOf.has(`Left${p}`) && boneOf.has(`Right${p}`));
    if (!top || !pair) throw new Error('The character is missing spine or limb bones.');
    const srcUp = v(rest, J[top]).sub(v(rest, J.Hips));
    const srcLat = v(rest, J[`Left${pair}`]).sub(v(rest, J[`Right${pair}`]));
    const tgtUp = restWorldP.get(boneOf.get(top)).clone().sub(restWorldP.get(hips));
    const tgtLat = restWorldP.get(boneOf.get(`Left${pair}`)).clone().sub(restWorldP.get(boneOf.get(`Right${pair}`)));

    const aligned = new Array(motion.jointCount);
    aligned[J.Hips] = frameQuaternion(tgtUp, tgtLat).multiply(frameQuaternion(srcUp, srcLat).invert());
    this.directionChild = new Map();
    for (let j = 0; j < motion.jointCount; j++) {
      if (j === J.Hips) continue;
      const parent = motion.parents[j];
      const base = aligned[parent] ?? aligned[J.Hips];
      const name = motion.joint_names[j];
      const child = (DIRECTION[name] || []).find((c) => boneOf.has(c));
      if (!boneOf.has(name) || !child) {
        aligned[j] = base.clone();
        continue;
      }
      this.directionChild.set(name, child);
      const src = v(rest, J[child]).sub(v(rest, j)).applyQuaternion(base).normalize();
      const tgt = restWorldP.get(boneOf.get(child)).clone().sub(restWorldP.get(boneOf.get(name))).normalize();
      aligned[j] = new THREE.Quaternion().setFromUnitVectors(src, tgt).multiply(base);
    }
    this.offset = new Map([...mapping].map(([bone, joint]) => [bone, aligned[J[joint]].clone().invert().multiply(restWorldQ.get(bone))]));
    this.jointOf = new Map(mapping);

    // Root motion scale from hip height above the floor (Kimodo motions stand on y = 0).
    let srcFloor = Infinity;
    for (let j = 0; j < motion.jointCount; j++) srcFloor = Math.min(srcFloor, rest[3 * j + 1]);
    const box = new THREE.Box3().setFromObject(this.root);
    this.floorY = box.min.y;
    this.hipsRest = restWorldP.get(hips).clone();
    this.scale = (this.hipsRest.y - this.floorY) / (rest[3 * J.Hips + 1] - srcFloor);
    this.start = v(motion.positions, J.Hips);
    this.staticParentQ = new Map();
    for (const bone of this.bones) {
      if (!this.restLocal.has(bone.parent)) this.staticParentQ.set(bone, bone.parent.getWorldQuaternion(new THREE.Quaternion()));
    }
    this.worldQ = new Map();
    this._q = new THREE.Quaternion();
    this._q2 = new THREE.Quaternion();
  }

  sourceQuat(frame, joint, out) {
    const { rotations, jointCount } = this.motion;
    const i0 = Math.floor(frame);
    const i1 = Math.min(i0 + 1, this.motion.frames - 1);
    const t = frame - i0;
    const a = 4 * (i0 * jointCount + joint);
    const b = 4 * (i1 * jointCount + joint);
    out.set(rotations[a], rotations[a + 1], rotations[a + 2], rotations[a + 3]);
    this._q2.set(rotations[b], rotations[b + 1], rotations[b + 2], rotations[b + 3]);
    return out.slerp(this._q2, t);
  }

  sourcePos(frame, joint, out = new THREE.Vector3()) {
    const { positions, jointCount } = this.motion;
    const i0 = Math.floor(frame);
    const i1 = Math.min(i0 + 1, this.motion.frames - 1);
    const t = frame - i0;
    const a = 3 * (i0 * jointCount + joint);
    const b = 3 * (i1 * jointCount + joint);
    return out.set(
      positions[a] + t * (positions[b] - positions[a]),
      positions[a + 1] + t * (positions[b + 1] - positions[a + 1]),
      positions[a + 2] + t * (positions[b + 2] - positions[a + 2]),
    );
  }

  /** World position where a source point is drawn next to/onto the character. */
  placeSource(p, inPlace, frame) {
    const out = new THREE.Vector3(
      this.hipsRest.x + this.scale * (p.x - this.start.x),
      this.floorY + this.scale * p.y,
      this.hipsRest.z + this.scale * (p.z - this.start.z),
    );
    if (inPlace) {
      const hips = this.sourcePos(frame, this.motion.index.Hips);
      out.x -= this.scale * (hips.x - this.start.x);
      out.z -= this.scale * (hips.z - this.start.z);
    }
    return out;
  }

  apply(frame, { inPlace = false } = {}) {
    const J = this.motion.index;
    for (const bone of this.bones) {
      const parentQ = this.worldQ.get(bone.parent) ?? this.staticParentQ.get(bone);
      const joint = this.jointOf.get(bone);
      if (joint) {
        const world = this.sourceQuat(frame, J[joint], new THREE.Quaternion()).multiply(this.offset.get(bone));
        bone.quaternion.copy(parentQ.clone().invert().multiply(world));
        this.worldQ.set(bone, world);
      } else {
        bone.quaternion.copy(this.restLocal.get(bone).q);
        this.worldQ.set(bone, parentQ.clone().multiply(bone.quaternion));
      }
    }
    const hips = this.boneOf.get('Hips');
    const target = this.placeSource(this.sourcePos(frame, J.Hips), inPlace, frame);
    hips.parent.updateWorldMatrix(true, false);
    hips.position.copy(hips.parent.worldToLocal(target));
    this.root.updateMatrixWorld(true);
  }

  reset() {
    for (const [bone, rest] of this.restLocal) {
      bone.quaternion.copy(rest.q);
      bone.position.copy(rest.p);
    }
    this.root.updateMatrixWorld(true);
  }
}
