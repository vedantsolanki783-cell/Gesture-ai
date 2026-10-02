import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const WASM_PATH = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm');
const MODEL_PATH = httpsUrl('storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task');

const HOLD_DELAY_MS = 500;
const POST_EMIT_LOCK_MS = 1200;

const PINCH_THRESH = 0.058;
let filteredX = 0.5, filteredY = 0.5;
let prevTargetX = 0.5, prevTargetY = 0.5;

let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let mouselessMode = false;
let lastToggleTime = 0, lastClickTime = 0;

let candidateSign = '', latchedSign = '';
let candidateStartTime = 0, lastEmittedTime = 0;
let releaseFrameCount = 0;

interface MLEmbedding { label: string; vector: number[]; }
let mlDatabase: MLEmbedding[] = [];
let pendingTrainLabel: string | null = null;

try {
  const saved = localStorage.getItem('nova_ml_gestures');
  if (saved) mlDatabase = JSON.parse(saved);
} catch {}

export function learnSign(label: string) { pendingTrainLabel = label.trim().toUpperCase(); }
export function clearTrainedSigns() { mlDatabase = []; localStorage.removeItem('nova_ml_gestures'); }
export function getTrainedSignsCount() { return mlDatabase.length; }

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

function normalizeVector(lm: Landmark[]): number[] {
  const wrist = lm[0];
  let maxDist = 0.0001;
  const centered = lm.map(p => {
    const dx = p.x - wrist.x, dy = p.y - wrist.y, dz = (p.z || 0) - (wrist.z || 0);
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxDist) maxDist = dist;
    return { x: dx, y: dy, z: dz };
  });
  const vector: number[] = [];
  for (const p of centered) vector.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  return vector;
}

function classifyWithML(vector: number[]): string {
  if (!mlDatabase.length) return '';
  let bestLabel = '', minDist = Infinity;
  for (const item of mlDatabase) {
    let distSq = 0;
    for (let i = 0; i < 63; i++) distSq += Math.pow(vector[i] - item.vector[i], 2);
    if (distSq < minDist) { minDist = distSq; bestLabel = item.label; }
  }
  return minDist < 1.15 ? bestLabel : '';
}

function classifyASL(lm: Landmark[]): string {
  const wrist = lm[0], palm = Math.max(d(wrist, lm[9]), 0.05);
  const nd = (a: Landmark, b: Landmark) => d(a, b) / palm;
  
  const idxUp = lm[8].y < lm[6].y && nd(lm[8], wrist) > nd(lm[5], wrist);
  const midUp = lm[12].y < lm[10].y && nd(lm[12], wrist) > nd(lm[9], wrist);
  const rngUp = lm[16].y < lm[14].y && nd(lm[16], wrist) > nd(lm[13], wrist);
  const pnkUp = lm[20].y < lm[18].y && nd(lm[20], wrist) > nd(lm[17], wrist);
  const allClosed = !idxUp && !midUp && !rngUp && !pnkUp;

  if (lm[4].y > lm[3].y && lm[4].y > lm[5].y + palm * 0.3 && allClosed) return 'CLEAR';
  if (idxUp && !midUp && !rngUp && pnkUp) return 'SEND';
  if (idxUp && midUp && rngUp && pnkUp) return 'B';
  if (idxUp && midUp && rngUp && !pnkUp) return 'W';
  if (!idxUp && !midUp && !rngUp && pnkUp) return d(lm[4], lm[17]) > d(lm[2], lm[17]) * 1.1 ? 'Y' : 'I';
  if (idxUp && midUp && !rngUp && !pnkUp) return nd(lm[8], lm[12]) > 0.35 ? 'V' : 'U';
  if (idxUp && !midUp && !rngUp && !pnkUp) return nd(lm[4], lm[8]) > 0.9 ? 'L' : 'D';
  if (allClosed && lm[4].y < lm[6].y + palm * 0.1) return 'A';

  return '';
}

async function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_PATH).then(v => 
      HandLandmarker.createFromOptions(v, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
        runningMode: 'VIDEO', numHands: 2,
        minHandDetectionConfidence: 0.45, minTrackingConfidence: 0.45
      })
    );
  }
  return landmarkerPromise;
}

function updateMouselessCursor(visible: boolean, x=0.5, y=0.5, pinching=false) {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (bridge?.updateAirMouse) bridge.updateAirMouse(visible, x, y, pinching);
}

export async function localVision(video: HTMLVideoElement, _: number): Promise<VisionResult> {
  try {
    if (!video || video.readyState < 2) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    
    video.playsInline = true; video.muted = true;
    if (video.paused && video.srcObject) video.play().catch(()=>({}));

    const lm = await getLandmarker();
    const ts = performance.now();
    lastVideoTime = ts > lastVideoTime ? ts : lastVideoTime + 1;
    const res = lm.detectForVideo(video, lastVideoTime);
    const hands = (res.landmarks || []) as Landmark[][];
    const now = Date.now();

    if (!hands.length) {
      if (++releaseFrameCount >= 5) { latchedSign = ''; candidateSign = ''; }
      updateMouselessCursor(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    if (hands.length === 2 && Math.abs(hands[0][9].y - hands[1][9].y) > 0.1) {
      if (now - lastToggleTime > 1400) {
        mouselessMode = !mouselessMode; lastToggleTime = now;
        if (mouselessMode) (window as any).NovaAndroid?.enableOverlayBubble();
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    if (mouselessMode) {
      const h = hands[0];
      const rayX = (h[5].x * 0.65 + h[8].x * 0.35); 
      const rayY = (h[5].y * 0.65 + h[8].y * 0.35);
      
      const normX = Math.max(0.01, Math.min(0.99, (rayX - 0.15) / 0.7));
      const normY = Math.max(0.01, Math.min(0.99, (rayY - 0.15) / 0.7));

      const vel = Math.hypot(normX - prevTargetX, normY - prevTargetY);
      prevTargetX = normX; prevTargetY = normY;
      
      const alpha = vel < 0.005 ? 0.15 : vel > 0.05 ? 0.8 : 0.4;
      filteredX = filteredX * (1 - alpha) + normX * alpha;
      filteredY = filteredY * (1 - alpha) + normY * alpha;

      const pinching = d(h[8], h[4]) < PINCH_THRESH;
      updateMouselessCursor(true, filteredX, filteredY, pinching);
      window.dispatchEvent(new CustomEvent('nova-mouseless-toggle', { detail: true }));
      window.dispatchEvent(new CustomEvent('nova-mouseless-cursor', { detail: { x: filteredX * window.innerWidth, y: filteredY * window.innerHeight, pinching } }));

      const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
      if (filteredY < 0.12) { window.scrollBy(0, -15); bridge?.swipeScreen?.(500,450,500,1200,260); }
      if (filteredY > 0.88) { window.scrollBy(0, 15); bridge?.swipeScreen?.(500,1200,500,450,260); }

      if (pinching && now - lastClickTime > 600) {
        lastClickTime = now; bridge?.clickAirMouse?.(filteredX, filteredY);
        (document.elementFromPoint(filteredX * window.innerWidth, filteredY * window.innerHeight) as HTMLElement | null)?.click();
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    window.dispatchEvent(new CustomEvent('nova-mouseless-toggle', { detail: false }));

    const vector = normalizeVector(hands[0]);
    if (pendingTrainLabel) {
      mlDatabase.push({ label: pendingTrainLabel, vector });
      localStorage.setItem('nova_ml_gestures', JSON.stringify(mlDatabase));
      pendingTrainLabel = null;
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    let sign = classifyWithML(vector) || classifyASL(hands[0]);

    if (!sign) {
      if (++releaseFrameCount >= 5) { latchedSign = ''; candidateSign = ''; }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    releaseFrameCount = 0;
    if (now - lastEmittedTime < POST_EMIT_LOCK_MS) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    if (sign === latchedSign) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

    if (sign !== candidateSign) { candidateSign = sign; candidateStartTime = now; }
    else if (now - candidateStartTime > HOLD_DELAY_MS) {
      latchedSign = sign; lastEmittedTime = now;
      return { type: sign === 'CLEAR' || sign === 'SEND' ? 'GESTURE' : 'LETTER', value: sign, confidence: 0.95, source: 'local' };
    }
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  } catch { return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' }; }
}

export async function isLocalVisionModelAvailable() { return true; }
export function applyCustomGesture(res: VisionResult, _: CustomGesture[]) { return res; }
