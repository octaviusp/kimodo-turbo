// Deterministic bone-name adapter for Kimodo and common humanoid skeletons.

export const MAPPABLE_JOINTS = [
  'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Head', 'HeadEnd',
  ...['Left', 'Right'].flatMap((s) => [
    `${s}Shoulder`, `${s}Arm`, `${s}ForeArm`, `${s}Hand`, `${s}HandMiddle1`,
    `${s}Leg`, `${s}Shin`, `${s}Foot`, `${s}ToeBase`, `${s}ToeEnd`,
  ]),
];

const SKIP = /twist|roll|helper|ctrl|control|ik|pole|target|end_?site|nub|corrective|bend|scale|prop|weapon/i;
const FINGERS = /thumb|index|middle|ring|pinky|little|finger|digit|metacarpal/i;

function tokens(name) {
  return name
    .replace(/^(mixamorig\d*|cc_base_|def-|org-|bip0*1|armature|rig)[:_ ]?/i, '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
    .map((t) => t.toLowerCase());
}

function sideOf(parts) {
  if (parts.some((t) => t === 'left' || t === 'l' || t === 'lft')) return 'Left';
  if (parts.some((t) => t === 'right' || t === 'r' || t === 'rt' || t === 'rgt')) return 'Right';
  const joined = parts.join('');
  if (/^left|left$/.test(joined)) return 'Left';
  if (/^right|right$/.test(joined)) return 'Right';
  return null;
}

function limbPart(word, isMiddleFinger) {
  if (/toe.?end|toetip|toe.?nub/.test(word)) return 'ToeEnd';
  if (/toe|ball/.test(word)) return 'ToeBase';
  if (/foot|ankle/.test(word)) return 'Foot';
  if (/forearm|lowerarm|elbow/.test(word)) return 'ForeArm';
  if (/upperarm|^arm$|uparm/.test(word)) return 'Arm';
  if (/clavicle|shoulder|collar/.test(word)) return 'Shoulder';
  if (isMiddleFinger) return 'HandMiddle1';
  if (/hand.?end/.test(word)) return 'HandMiddle1';
  if (/hand|wrist/.test(word)) return 'Hand';
  if (/thigh|upleg|upperleg|^hip$/.test(word)) return 'Leg';
  if (/calf|shin|lowerleg|knee|^leg$/.test(word)) return 'Shin';
  return null;
}

/** Kimodo auto-rig output and Kimodo-named skeletons map 1:1 (except HandEnd -> HandMiddle1). */
function kimodoNative(bones) {
  const names = new Set(bones.map((b) => b.name));
  return names.has('Hips') && names.has('LeftShin') && names.has('RightShin');
}

export function mapBones(bones) {
  const mapping = new Map();
  if (kimodoNative(bones)) {
    for (const bone of bones) {
      const joint = bone.name.replace(/HandEnd$/, 'HandMiddle1');
      if (MAPPABLE_JOINTS.includes(joint)) mapping.set(bone, joint);
    }
    return mapping;
  }
  const spine = [];
  for (const bone of bones) {
    const parts = tokens(bone.name);
    const word = parts.filter((t) => !['left', 'right', 'l', 'r', 'lft', 'rt', 'rgt'].includes(t)).join('');
    if (!word || (SKIP.test(word) && !/^necktwist0?1$/.test(word))) continue;
    const finger = FINGERS.test(word);
    const middle1 = finger && /middle|^mid/.test(word) && /(^|[^0-9])(1|01)$/.test(word);
    if (finger && !middle1) continue;
    const side = sideOf(parts);
    if (side) {
      const part = limbPart(word, middle1);
      if (part && ![...mapping.values()].includes(`${side}${part}`)) mapping.set(bone, `${side}${part}`);
    } else if (/^(hips|pelvis|hip|root.?hips)$/.test(word) && ![...mapping.values()].includes('Hips')) {
      mapping.set(bone, 'Hips');
    } else if (/spine|chest|waist|torso|abdomen|back/.test(word)) {
      spine.push(bone);
    } else if (/neck/.test(word) && ![...mapping.values()].includes('Neck1')) {
      mapping.set(bone, 'Neck1');
    } else if (/head.?(top|end|nub)|headtop/.test(word)) {
      mapping.set(bone, 'HeadEnd');
    } else if (/^head$/.test(word)) {
      mapping.set(bone, 'Head');
    }
  }
  // Spine bones in hierarchy order (bones are visited parents-first): first -> Spine1,
  // last -> Chest, one in between -> Spine2.
  if (spine.length) {
    mapping.set(spine[0], 'Spine1');
    if (spine.length >= 2) mapping.set(spine[spine.length - 1], 'Chest');
    if (spine.length >= 3) mapping.set(spine[Math.floor(spine.length / 2)], 'Spine2');
  }
  return mapping;
}

/** Fail before replacing the preview when an existing skeleton cannot be animated safely. */
export function validateMapping(mapping) {
  const joints = new Set(mapping.values());
  if (joints.size !== mapping.size) throw new Error('Choose a character with a single humanoid skeleton.');
  const required = ['Hips', 'Head', ...['Left', 'Right'].flatMap((side) =>
    ['Arm', 'ForeArm', 'Hand', 'Leg', 'Shin', 'Foot'].map((part) => `${side}${part}`))];
  const missing = required.filter((joint) => !joints.has(joint));
  if (!['Spine1', 'Spine2', 'Chest'].some((joint) => joints.has(joint))) missing.push('Spine');
  if (missing.length) throw new Error(`The character skeleton is missing recognizable bones: ${missing.join(', ')}.`);
}
