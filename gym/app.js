import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/GLTFLoader.js';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { MAPPABLE_JOINTS, autoMap } from './mapping.js';
import { Retargeter, decodeMotion } from './retarget.js';

const $ = (id) => document.getElementById(id);
const PRESETS = [
  ['Correr', 'A person runs forward at a steady pace.'],
  ['Caminar', 'A person walks forward casually.'],
  ['Saltar', 'A person jumps forward with both feet and lands softly.'],
  ['Bailar', 'A person dances energetically, spinning around with arms raised.'],
  ['Patada', 'A person performs a roundhouse kick, then returns to a fighting stance.'],
  ['Sentarse', 'A person sits down on a chair and waves hello.'],
];

const state = {
  model: null, // { root, bones, mapping, name, rig }
  motions: [], // history: { json, motion, label }
  current: null,
  retargeter: null,
  time: 0,
  playing: false,
};

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

async function loadModel(buffer, name, rig, report) {
  const gltf = await loader.parseAsync(buffer, '');
  if (state.model) {
    state.retargeter?.reset();
    scene.remove(state.model.root);
  }
  const root = gltf.scene;
  const bones = [];
  root.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.frustumCulled = false; // skinned bounds do not follow the animation
    }
    if (o.isBone) bones.push(o);
  });
  root.traverse((o) => o.isSkinnedMesh && o.skeleton.pose());
  if (!bones.length) throw new Error('El modelo no tiene esqueleto');
  scene.add(root);
  normalizeModel(root);
  state.model = { root, bones, mapping: autoMap(bones), name, rig, report };
  $('model-name').textContent = name;
  $('model-rig').textContent = rig === 'auto'
    ? `Rig automático · ${report?.detection?.legs === 'covered' ? 'piernas cubiertas' : 'piernas separadas'} · ${bones.length} huesos`
    : `Rig propio · ${bones.length} huesos · ${state.model.mapping.size} mapeados`;
  frameModel(root);
  renderMapping();
  rebuildRetargeter();
}

let loadToken = 0;
async function fetchModel(request, name) {
  const token = ++loadToken; // the most recent request wins
  busy(`Preparando ${name}…`);
  try {
    const response = await fetch(request.url, request.init);
    if (!response.ok) throw new Error((await response.json()).error);
    const report = JSON.parse(response.headers.get('X-Rig-Report') || '{}');
    const buffer = await response.arrayBuffer();
    if (token !== loadToken) return;
    await loadModel(buffer, name, response.headers.get('X-Rig'), report);
  } catch (error) {
    genStatus(`No se pudo cargar ${name}: ${error.message}`, true);
  } finally {
    busy();
  }
}

const loadDefault = () => fetchModel({ url: '/api/default-model' }, 'Personaje realista');
const loadFile = (file) => fetchModel({ url: '/api/rig', init: { method: 'POST', body: file } }, file.name);

$('load-default').onclick = loadDefault;
$('file').onchange = (e) => e.target.files[0] && loadFile(e.target.files[0]);
viewport.addEventListener('dragover', (e) => { e.preventDefault(); viewport.classList.add('dragging'); });
viewport.addEventListener('dragleave', () => viewport.classList.remove('dragging'));
viewport.addEventListener('drop', (e) => {
  e.preventDefault();
  viewport.classList.remove('dragging');
  const file = [...e.dataTransfer.files].find((f) => f.name.toLowerCase().endsWith('.glb'));
  if (file) loadFile(file);
  else genStatus('Solo se aceptan archivos .glb', true);
});

// ---------- bone mapping ----------
function renderMapping() {
  const container = $('mapping');
  container.replaceChildren();
  const { bones, mapping } = state.model;
  for (const bone of bones) {
    const label = document.createElement('span');
    label.textContent = bone.name;
    label.title = bone.name;
    const select = document.createElement('select');
    select.append(new Option('—', ''), ...MAPPABLE_JOINTS.map((j) => new Option(j, j)));
    select.value = mapping.get(bone) || '';
    select.onchange = () => {
      if (select.value) mapping.set(bone, select.value);
      else mapping.delete(bone);
      rebuildRetargeter();
    };
    container.append(label, select);
  }
}

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

// ---------- prompt segments ----------
function addSegment(prompt = '', duration = 5) {
  const row = document.createElement('div');
  row.className = 'segment';
  row.innerHTML = `<textarea rows="2" placeholder="Describí el movimiento (en inglés)"></textarea>
    <input type="number" min="0.5" max="10" step="0.5" title="Duración en segundos">
    <button class="remove" title="Quitar segmento" aria-label="Quitar segmento">✕</button>`;
  row.querySelector('textarea').value = prompt;
  row.querySelector('input').value = duration;
  row.querySelector('.remove').onclick = () => {
    if ($('segments').children.length > 1) row.remove();
  };
  $('segments').append(row);
}
for (const [label, prompt] of PRESETS) {
  const chip = document.createElement('button');
  chip.className = 'chip';
  chip.textContent = label;
  chip.onclick = () => { $('segments').querySelector('textarea').value = prompt; };
  $('presets').append(chip);
}
addSegment(PRESETS[0][1], 5);
$('add-segment').onclick = () => addSegment('', 3);

async function generate() {
  const segments = [...$('segments').children].map((row) => ({
    prompt: row.querySelector('textarea').value.trim(),
    duration: Number(row.querySelector('input').value),
  }));
  if (segments.some((s) => !s.prompt)) return genStatus('Completá la descripción de cada segmento', true);
  const button = $('generate');
  button.disabled = true;
  const started = performance.now();
  const timer = setInterval(() => busy(`Generando… ${((performance.now() - started) / 1000).toFixed(1)} s`), 100);
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segments,
        seed: Number($('seed').value),
        steps: Number($('steps').value),
        postprocess: $('postprocess').checked,
      }),
    });
    const json = await response.json();
    if (!response.ok) throw new Error(json.error);
    const entry = { json, motion: decodeMotion(json), label: segments.map((s) => s.prompt).join(' → ') };
    state.motions.unshift(entry);
    selectMotion(entry);
    genStatus(`Generado en ${json.seconds.toFixed(1)} s en GPU · ${json.frames} frames a ${json.fps} fps`);
    renderHistory();
  } catch (error) {
    genStatus(error.message, true);
  } finally {
    clearInterval(timer);
    busy();
    button.disabled = false;
  }
}
$('generate').onclick = generate;
$('segments').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate();
});

function renderHistory() {
  const list = $('history');
  list.replaceChildren(...state.motions.map((entry) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.className = entry === state.current ? 'active' : '';
    const prompt = document.createElement('span');
    prompt.className = 'prompt';
    prompt.textContent = entry.label;
    const meta = document.createElement('span');
    meta.className = 'mono';
    meta.textContent = `${(entry.json.frames / entry.json.fps).toFixed(1)} s`;
    button.append(prompt, meta);
    button.title = entry.label;
    button.onclick = () => { selectMotion(entry); renderHistory(); };
    li.append(button);
    return li;
  }));
}

function selectMotion(entry) {
  state.current = entry;
  state.time = 0;
  state.playing = true;
  $('play').disabled = false;
  $('scrub').disabled = false;
  $('play').textContent = '❚❚';
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
let metricTimer = 0;
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
  if (rt && (metricTimer += dt) > 0.5) {
    metricTimer = 0;
    const frame = Math.min(state.time * state.current.json.fps, state.current.json.frames - 1);
    $('metric').textContent = `Error de dirección del retarget: ${rt.directionError(frame).toFixed(1)}° (huesos mapeados: ${state.model.mapping.size})`;
  }
  controls.update();
  renderer.render(scene, camera);
});

// ---------- boot ----------
async function pollStatus() {
  try {
    const status = await (await fetch('/api/status')).json();
    if (status.ready) {
      setStatus('Listo', 'ok');
      $('generate').disabled = false;
      return;
    }
    if (status.error) {
      setStatus('Error', 'bad');
      genStatus(status.error, true);
      return;
    }
  } catch {
    setStatus('Sin servidor', 'bad');
  }
  setTimeout(pollStatus, 1000);
}
resize();
pollStatus();
loadDefault();
window.gym = state; // inspection hook for QA scripts
