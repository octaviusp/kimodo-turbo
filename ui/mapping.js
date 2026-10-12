// The bundled character uses Kimodo joint names; no per-rig mapping is needed.
export function mapBones(bones) {
  return new Map(bones.map((bone) => [bone, bone.name.replace(/HandEnd$/, 'HandMiddle1')]));
}
