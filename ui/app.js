import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/GLTFLoader.js';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { mapBones, validateMapping } from './mapping.js';
import { Retargeter, decodeMotion } from './retarget.js';

const $ = (id) => document.getElementById(id);
const PRESETS = [
  ['Run', 'A person runs forward at a steady pace.'],
  ['Walk', 'A person walks forward casually.'],
  ['Jump', 'A person jumps forward with both feet and lands softly.'],
  ['Dance', 'A person dances energetically, spinning around with arms raised.'],
  ['Kick', 'A person performs a roundhouse kick, then returns to a fighting stance.'],
  ['Sit', 'A person sits down on a chair and waves hello.'],
];

const state = {
  model: null,
  current: null,
  retargeter: null,
  time: 0,
  playing: false,
  ready: false,
  generating: false,
  loadingCharacter: false,
  characterId: 'tunic',
};

const characters = new Map();

// ---------- scene ----------
const viewport = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);
scene.fog = new THREE.Fog(0x101318, 12, 40);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a2420, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.4);
sun.position.set(3, 6, 4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
scene.add(sun, sun.target);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x1a1e24, roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(200, 200, 0x3a4150, 0x262b33);
grid.position.y = 0.001;
scene.add(grid);

const skeletonLines = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x5ac8fa, depthTest: false, transparent: true, opacity: 0.9 }),
);
skeletonLines.renderOrder = 10;
skeletonLines.visible = false;
scene.add(skeletonLines);

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

const DISPLAY_HEIGHT = 1.75; // metres; files arrive in any unit (Mixamo exports are often cm x 0.01)

function orientModel(root, report) {
  if (!report.up || !report.forward) return;
  const up = new THREE.Vector3(...report.up).normalize();
  const forward = new THREE.Vector3(...report.forward).normalize();
  const right = new THREE.Vector3().crossVectors(up, forward).normalize();
  if (!Number.isFinite(right.lengthSq()) || right.lengthSq() < 0.5) {
    throw new Error('The character orientation could not be determined.');
  }
  forward.crossVectors(right, up).normalize();
  const basis = new THREE.Matrix4().makeBasis(right, up, forward);
  root.quaternion.premultiply(new THREE.Quaternion().setFromRotationMatrix(basis).invert());
}

/** Scale the model to a person's height, feet on the floor, centred on the origin. */
function normalizeModel(root) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  root.scale.multiplyScalar(DISPLAY_HEIGHT / Math.max(box.max.y - box.min.y, 1e-6));
  root.updateMatrixWorld(true);
  box.setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  root.position.sub(new THREE.Vector3(center.x, box.min.y, center.z));
  root.updateMatrixWorld(true);
}

function frameModel(root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const h = Math.max(size.y, 0.1);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(h * 0.9, h * 0.25, h * 2.2));
  camera.near = h / 100;
  camera.far = h * 200;
  camera.updateProjectionMatrix();
  sun.position.copy(center).add(new THREE.Vector3(h * 2, h * 4, h * 2.5));
  sun.target.position.copy(center);
  const s = h * 2;
  Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: h / 10, far: h * 20 });
  sun.shadow.camera.updateProjectionMatrix();
}

// ---------- status helpers ----------
function setStatus(text, kind) {
  const el = $('status');
  el.textContent = text;
  el.className = `pill pill-${kind}`;
}
function busy(text) {
  $('busy').hidden = !text;
  $('busy-text').textContent = text || '';
}
function genStatus(text, isError = false) {
  const el = $('gen-status');
  el.textContent = text;
  el.style.color = isError ? 'var(--bad)' : '';
}

// ---------- model loading ----------
const loader = new GLTFLoader();

function disposeModel(root) {
  const geometries = new Set();
  const materials = new Set();
  const textures = new Set();
  root.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      if (!material) continue;
      materials.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    }
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
  for (const texture of textures) texture.dispose();
}

async function loadModel(buffer, report) {
  const gltf = await loader.parseAsync(buffer, '');
  const root = gltf.scene;
  const bones = [];
  let skinnedMeshes = 0;
  root.traverse((object) => {
    if (object.isMesh) {
      object.castShadow = true;
      object.frustumCulled = false;
    }
    if (object.isBone) {
      const node = gltf.parser.associations.get(object)?.nodes;
      object.name = gltf.parser.json.nodes[node]?.name || object.name;
      bones.push(object);
    }
    if (object.isSkinnedMesh) skinnedMeshes++;
  });
  try {
    if (!skinnedMeshes || !bones.length) throw new Error('The prepared character has no usable skeleton.');
    root.traverse((object) => { if (object.isSkinnedMesh) object.skeleton.pose(); });
    const mapping = mapBones(bones);
    validateMapping(mapping);
    orientModel(root, report);
    normalizeModel(root);
    if (state.model) {
      state.retargeter?.reset();
      scene.remove(state.model.root);
      disposeModel(state.model.root);
    }
    scene.add(root);
    state.model = { root, bones, mapping };
    frameModel(root);
    rebuildRetargeter();
  } catch (error) {
    disposeModel(root);
    throw error;
  }
}

function updateGenerateButton() {
  const unavailable = state.generating || state.loadingCharacter;
  $('generate').disabled = !state.ready || !state.model || unavailable;
  $('choose-character').disabled = unavailable;
  $('default-character').disabled = unavailable;
  $('character-select').disabled = unavailable || !characters.size;
}

function characterStatus(text, isError = false) {
  const element = $('character-status');
  element.textContent = text;
  element.style.color = isError ? 'var(--bad)' : '';
}

async function prepareCharacter(buffer, name) {
  const response = await fetch('/api/character', {
    method: 'POST', headers: { 'Content-Type': 'model/gltf-binary' }, body: buffer,
  });
  if (!response.ok) {
    const payload = await response.json();
    throw new Error(payload.error || 'The character could not be prepared.');
  }
  const report = JSON.parse(response.headers.get('X-Rig-Report') || '{}');
  await loadModel(await response.arrayBuffer(), report);
  state.model.report = report;
  characterStatus(`${name} is ready.`);
}

async function changeCharacter(read, name, identifier = '') {
  if (state.loadingCharacter || state.generating) return;
  state.loadingCharacter = true;
  updateGenerateButton();
  characterStatus('Preparing character…');
  busy('Preparing character…');
  try {
    await prepareCharacter(await read(), name);
    state.characterId = identifier;
  } catch (error) {
    characterStatus(error.message, true);
  } finally {
    state.loadingCharacter = false;
    $('character-select').value = state.characterId;
    updateGenerateButton();
    busy();
  }
}

async function loadDefault() {
  return changeCharacter(async () => {
    const response = await fetch('/assets/character.glb');
    if (!response.ok) throw new Error('Default character file not found.');
    return response.arrayBuffer();
  }, 'Default character', 'tunic');
}

async function loadCharacterLibrary() {
  try {
    const response = await fetch('/api/characters');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'The character library could not be loaded.');
    for (const character of payload.characters) {
      characters.set(character.id, character);
      $('character-select').add(new Option(character.label, character.id));
    }
    $('character-library').hidden = !characters.size;
    $('character-select').value = state.characterId;
    updateGenerateButton();
  } catch (error) {
    characterStatus(error.message, true);
  }
}

$('character-select').onchange = async (event) => {
  const character = characters.get(event.target.value);
  if (!character) return;
  await changeCharacter(async () => {
    const response = await fetch(character.url);
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || 'The character could not be loaded.');
    }
    return response.arrayBuffer();
  }, character.label, character.id);
};

$('choose-character').onclick = () => $('character-file').click();
$('default-character').onclick = loadDefault;
$('character-file').onchange = async (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  if (!/\.glb$/i.test(file.name)) return characterStatus('Choose a .glb character file.', true);
  if (!file.size || file.size > 100 * 1024 * 1024) {
    return characterStatus('Choose a .glb character smaller than 100 MB.', true);
  }
  await changeCharacter(() => file.arrayBuffer(), file.name);
};

function rebuildRetargeter() {
  state.retargeter?.reset();
  state.retargeter = null;
  if (!state.model || !state.current) return;
  try {
    state.retargeter = new Retargeter(state.current.motion, state.model.root, state.model.bones, state.model.mapping);
    updateSkeletonGeometry();
    applyFrame();
  } catch (error) {
    genStatus(error.message, true);
  }
}

// ---------- generation ----------
for (const [label, prompt] of PRESETS) {
  const chip = document.createElement('button');
  chip.className = 'chip';
  chip.textContent = label;
  chip.onclick = () => { $('prompt').value = prompt; };
  $('presets').append(chip);
}
$('prompt').value = PRESETS[1][1];

async function generate() {
  if (!state.ready || !state.model || state.generating || state.loadingCharacter) return;
  const prompt = $('prompt').value.trim();
  if (!prompt) return genStatus('Describe the motion first.', true);
  const seconds = Number($('duration').value);
  const seed = Number($('seed').value);
  if (!Number.isFinite(seconds) || seconds < 0.5 || seconds > 10) {
    return genStatus('Duration must be between 0.5 and 10 seconds.', true);
  }
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
    return genStatus('Seed must be an integer between 0 and 4294967295.', true);
  }
  state.generating = true;
  updateGenerateButton();
  const started = performance.now();
  const timer = setInterval(() => busy(`Generating… ${((performance.now() - started) / 1000).toFixed(1)} s`), 100);
  try {
    const response = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, duration: seconds, seed }),
    });
    const json = await response.json();
    if (!response.ok) throw new Error(json.error);
    selectMotion({ json, motion: decodeMotion(json) });
    $('download').href = json.download;
    $('download').hidden = false;
    genStatus(`Generated in ${json.seconds.toFixed(1)} s · ${json.frames} frames at ${json.fps} fps`);
  } catch (error) {
    genStatus(error.message, true);
  } finally {
    clearInterval(timer);
    busy();
    state.generating = false;
    updateGenerateButton();
  }
}
$('generate').onclick = generate;
$('prompt').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) generate();
});

function selectMotion(entry) {
  state.current = entry;
  state.time = 0;
  state.playing = true;
  $('play').disabled = false;
  $('scrub').disabled = false;
  $('play').textContent = '❚❚';
  $('play').setAttribute('aria-label', 'Pause');
  rebuildRetargeter();
}

// ---------- playback ----------
const duration = () => (state.current ? (state.current.json.frames - 1) / state.current.json.fps : 0);

function updateSkeletonGeometry() {
  const { motion } = state.current;
  const pairs = motion.parents.flatMap((p, j) => (p >= 0 ? [p, j] : []));
  skeletonLines.userData.pairs = pairs;
  skeletonLines.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs.length * 3), 3));
}

function applyFrame() {
  const rt = state.retargeter;
  if (!rt) return;
  const frame = Math.min(state.time * state.current.json.fps, state.current.json.frames - 1);
  const inPlace = $('inplace').checked;
  rt.apply(frame, { inPlace });
  if (skeletonLines.visible) {
    const attr = skeletonLines.geometry.attributes.position;
    const p = new THREE.Vector3();
    skeletonLines.userData.pairs.forEach((joint, k) => {
      attr.setXYZ(k, ...rt.placeSource(rt.sourcePos(frame, joint, p), inPlace, frame).toArray());
    });
    attr.needsUpdate = true;
    skeletonLines.geometry.computeBoundingSphere();
  }
  $('scrub').value = duration() ? state.time / duration() : 0;
  $('time').textContent = `${state.time.toFixed(1)} / ${duration().toFixed(1)} s`;
}

$('play').onclick = () => {
  if (!state.current) return;
  if (!state.playing && state.time >= duration()) state.time = 0;
  state.playing = !state.playing;
  $('play').textContent = state.playing ? '❚❚' : '▶';
  $('play').setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
};
$('scrub').oninput = (e) => {
  state.playing = false;
  $('play').textContent = '▶';
  state.time = Number(e.target.value) * duration();
  applyFrame();
};
$('skeleton').onchange = (e) => { skeletonLines.visible = e.target.checked; applyFrame(); };
$('inplace').onchange = applyFrame;
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !['TEXTAREA', 'INPUT', 'SELECT'].includes(document.activeElement.tagName)) {
    e.preventDefault();
    $('play').click();
  }
});

const clock = new THREE.Clock();
const followTarget = new THREE.Vector3();
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  if (state.playing && state.current) {
    state.time += dt * Number($('speed').value);
    if (state.time > duration()) {
      if ($('loop').checked) state.time %= Math.max(duration(), 1e-3);
      else { state.time = duration(); state.playing = false; $('play').textContent = '▶'; }
    }
    applyFrame();
  }
  const rt = state.retargeter;
  if (rt && $('follow').checked) {
    rt.boneOf.get('Hips').getWorldPosition(followTarget);
    followTarget.y = controls.target.y;
    const delta = followTarget.sub(controls.target).multiplyScalar(Math.min(1, dt * 6));
    controls.target.add(delta);
    camera.position.add(delta);
  }
  controls.update();
  renderer.render(scene, camera);
});

// ---------- boot ----------
async function pollStatus() {
  try {
    const status = await (await fetch('/api/status')).json();
    if (status.ready) {
      setStatus('Ready', 'ok');
      state.ready = true;
      updateGenerateButton();
      return;
    }
    if (status.error) {
      setStatus('Error', 'bad');
      genStatus(status.error, true);
      return;
    }
  } catch {
    setStatus('Connecting', 'wait');
  }
  setTimeout(pollStatus, 1000);
}
resize();
pollStatus();
loadCharacterLibrary();
loadDefault();
window.kimodoTurbo = state;
