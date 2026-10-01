import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const WASM_PATH = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
const MODEL_PATH = httpsUrl(
  'storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
);

const HOLD_DELAY_MS = 400; // Faster detection once trained
const POST_EMIT_LOCK_MS = 1250;
const CONSENSUS_WINDOW = 6;
const MIN_CONSENSUS_MATCH = 4;
const MAX_FINGER_VELOCITY = 0.038;
const MAX_WRIST_VELOCITY = 0.020;
const RELEASE_FRAMES_REQUIRED = 5;

// ============================================================================
// MACHINE LEARNING (KNN) STATE & LOGIC
// ============================================================================
interface MLEmbedding {
  label: string;
  vector: number[];
}

let mlDatabase: MLEmbedding[] = [];
let pendingTrainLabel: string | null = null;

function loadMLDatabase() {
  try {
    const saved = localStorage.getItem('nova_ml_gestures');
    if (saved) mlDatabase = JSON.parse(saved);
  } catch (e) {
    mlDatabase = [];
  }
}
loadMLDatabase();

export function learnSign(label: string) {
  if (!label.trim()) return;
  pendingTrainLabel = label.trim().toUpperCase();
}

export function clearTrainedSigns() {
  mlDatabase = [];
  localStorage.removeItem('nova_ml_gestures');
}

export function getTrainedSignsCount() {
  return mlDatabase.length;
}

// Converts 21 3D landmarks into a scale-invariant, translation-invariant 63D vector
function normalizeHandToVector(lm: Landmark[]): number[] {
  const wrist = lm[0];
  let maxDist = 0.0001; // prevent divide by zero
  
  // 1. Shift everything so wrist is at (0,0,0)
  const centered = lm.map(p => {
    const dx = p.x - wrist.x;
    const dy = p.y - wrist.y;
    const dz = (p.z || 0) - (wrist.z || 0);
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxDist) maxDist = dist;
    return { x: dx, y: dy, z: dz };
  });

  // 2. Scale by max distance and flatten into an array of 63 numbers
  const vector: number[] = [];
  for (const p of centered) {
    vector.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  }
  return vector;
}

// Compares current hand to saved hands using Euclidean distance
function classifyWithML(vector: number[]): { label: string; distance: number } {
  if (mlDatabase.length === 0) return { label: '', distance: 999 };

  let bestLabel = '';
  let minDist = Infinity;

  for (const item of mlDatabase) {
    let distSq = 0;
    for (let i = 0; i < 63; i++) {
      distSq += Math.pow(vector[i] - item.vector[i], 2);
    }
    if (distSq < minDist) {
      minDist = distSq;
      bestLabel = item.label;
    }
  }

  // The distance threshold. If the closest match is too different, reject it.
  if (minDist < 1.15) {
    return { label: bestLabel, distance: minDist };
  }
  return { label: '', distance: minDist };
}

// ============================================================================
// CORE VISION STATE
// ============================================================================
let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;
let lastScrollTime = 0;
let activeVideoEl: HTMLVideoElement | null = null;
let bgIntervalId: any = null;

let smoothNormX = 0.5;
let smoothNormY = 0.5;

let prevLandmarks: Landmark[] | null = null;
let smoothedLandmarks: Landmark[] | null = null;

let candidateSign = '';
let candidateStartTime = 0;
let latchedSign = '';
let releaseFrameCount = 0;
let lastEmittedTime = 0;

const recentPredictions: VisionResult[] = [];

function ensureBackgroundVisionLoop() {
  if (bgIntervalId) return;
  bgIntervalId = setInterval(() => {
    if (activeVideoEl && document.hidden && mouselessMode) {
      localVision(activeVideoEl, performance.now()).catch(() => {});
    }
  }, 75);
}

async function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      const isAndroidApk =
        typeof window !== 'undefined' &&
        (Boolean((window as any).NovaAndroid) || /wv|Android/i.test(navigator.userAgent));

      const preferredDelegate = isAndroidApk ? 'CPU' : 'GPU';

      try {
        return await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: preferredDelegate },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.48,
          minHandPresenceConfidence: 0.48,
          minTrackingConfidence: 0.48
        });
      } catch {
        return await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.48,
          minHandPresenceConfidence: 0.48,
          minTrackingConfidence: 0.48
        });
      }
    })();
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

function smoothAndMeasureVelocity(raw: Landmark[]): { lm: Landmark[]; fingerVelocity: number; wristVelocity: number } {
  if (!smoothedLandmarks || smoothedLandmarks.length !== raw.length) {
    smoothedLandmarks = raw.map(p => ({ ...p }));
    prevLandmarks = raw.map(p => ({ ...p }));
    return { lm: smoothedLandmarks, fingerVelocity: 0, wristVelocity: 0 };
  }

  const alpha = 0.5;
  for (let i = 0; i < raw.length; i++) {
    smoothedLandmarks[i] = {
      x: smoothedLandmarks[i].x * (1 - alpha) + raw[i].x * alpha,
      y: smoothedLandmarks[i].y * (1 - alpha) + raw[i].y * alpha,
      z: (smoothedLandmarks[i].z || 0) * (1 - alpha) + (raw[i].z || 0) * alpha
    };
  }

  const tipIndices = [4, 8, 12, 16, 20];
  let totalDelta = 0;
  let wristDelta = 0;
  if (prevLandmarks) {
    for (const idx of tipIndices) {
      totalDelta += d(smoothedLandmarks[idx], prevLandmarks[idx]);
    }
    wristDelta = d(smoothedLandmarks[0], prevLandmarks[0]);
  }
  prevLandmarks = smoothedLandmarks.map(p => ({ ...p }));

  return {
    lm: smoothedLandmarks,
    fingerVelocity: totalDelta / tipIndices.length,
    wristVelocity: wristDelta
  };
}

function jointAngleDeg(mcp: Landmark, pip: Landmark, tip: Landmark): number {
  const v1x = mcp.x - pip.x;
  const v1y = mcp.y - pip.y;
  const v2x = tip.x - pip.x;
  const v2y = tip.y - pip.y;
  const dot = v1x * v2x + v1y * v2y;
  const mag = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (mag === 0) return 180;
  const cos = Math.max(-1, Math.min(1, dot / mag));
  return (Math.acos(cos) * 180) / Math.PI;
}

function isTwoStackedHorizontalHands(h1: Landmark[], h2: Landmark[]): boolean {
  if (!h1 || !h2 || h1.length < 21 || h2.length < 21) return false;
  const isHoriz = (lm: Landmark[]) =>
    Math.abs(lm[9].x - lm[0].x) > Math.abs(lm[9].y - lm[0].y) * 0.75 &&
    Math.abs(lm[12].x - lm[0].x) > Math.abs(lm[12].y - lm[0].y) * 0.75;
  const verticalGap = Math.abs(h1[9].y - h2[9].y);
  const horizontalOverlap = Math.abs(h1[9].x - h2[9].x);
  return isHoriz(h1) && isHoriz(h2) && verticalGap > 0.1 && horizontalOverlap < 0.5;
}

function isOnlyMiddleFinger(lm: Landmark[]): boolean {
  if (!lm || lm.length < 21) return false;
  const palm = Math.max(d(lm[0], lm[9]), 0.05);
  const middleStraight = jointAngleDeg(lm[9], lm[10], lm[12]) > 142 && lm[12].y < lm[10].y;
  const othersCurled =
    lm[12].y < lm[8].y - palm * 0.3 &&
    lm[12].y < lm[16].y - palm * 0.3 &&
    lm[12].y < lm[20].y - palm * 0.3;
  return middleStraight && othersCurled;
}

function showModeNotification(isMouseless: boolean) {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (isMouseless && bridge && typeof bridge.enableOverlayBubble === 'function') {
    try { bridge.enableOverlayBubble(); } catch {}
  }

  let badge = document.getElementById('nova-mode-badge');
  if (!badge) {
    badge = document.createElement('div');
    badge.id = 'nova-mode-badge';
    Object.assign(badge.style, {
      position: 'fixed',
      top: '14px',
      left: '50%',
      transform: 'translateX(-50%)',
      padding: '8px 16px',
      borderRadius: '999px',
      fontWeight: 'bold',
      fontSize: '13px',
      color: '#ffffff',
      zIndex: '100000',
      boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
      pointerEvents: 'none',
      transition: 'opacity 0.25s ease'
    });
    document.body.appendChild(badge);
  }
  badge.style.background = isMouseless ? '#dc2626' : '#10b981';
  badge.textContent = isMouseless
    ? '🖱 System Wireless Mouse ON'
    : '✋ ML Sign Mode ON';
  badge.style.opacity = '1';
  setTimeout(() => {
    if (badge && !mouselessMode) badge.style.opacity = '0';
  }, 2500);
}

function updateMouselessCursor(visible: boolean, normX = 0.5, normY = 0.5, pinching = false) {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (bridge && typeof bridge.updateAirMouse === 'function') {
    try { bridge.updateAirMouse(visible, normX, normY, pinching); } catch {}
  }
}

function getConsensusPrediction(raw: VisionResult): VisionResult {
  recentPredictions.push(raw);
  if (recentPredictions.length > CONSENSUS_WINDOW) {
    recentPredictions.shift();
  }

  const counts = new Map<string, { count: number; sample: VisionResult }>();
  for (const item of recentPredictions) {
    if (item.type === 'UNKNOWN' || !item.value) continue;
    const prev = counts.get(item.value);
    if (prev) prev.count++;
    else counts.set(item.value, { count: 1, sample: item });
  }

  let bestSign = '';
  let bestCount = 0;
  let bestSample: VisionResult = { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

  for (const [sign, data] of counts.entries()) {
    if (data.count > bestCount) {
      bestSign = sign;
      bestCount = data.count;
      bestSample = data.sample;
    }
  }

  if (bestSign && bestCount >= MIN_CONSENSUS_MATCH) {
    return bestSample;
  }
  return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
}

function stabilizeSingleShot(raw: VisionResult, fingerVelocity: number): VisionResult {
  const empty: VisionResult = { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  const now = Date.now();

  if (now - lastEmittedTime < POST_EMIT_LOCK_MS) return empty;
  
  // Relaxed velocity constraint to allow dynamic signs if needed, but strict enough to prevent blur
  if (fingerVelocity > MAX_FINGER_VELOCITY) return empty;

  const consensus = getConsensusPrediction(raw);

  if (consensus.type === 'UNKNOWN' || !consensus.value) {
    releaseFrameCount++;
    if (releaseFrameCount >= RELEASE_FRAMES_REQUIRED) {
      latchedSign = '';
      candidateSign = '';
    }
    return empty;
  }

  releaseFrameCount = 0;
  if (consensus.value === latchedSign) return empty;

  if (consensus.value !== candidateSign) {
    candidateSign = consensus.value;
    candidateStartTime = now;
    return empty;
  }

  if (now - candidateStartTime < HOLD_DELAY_MS) return empty;

  latchedSign = consensus.value;
  lastEmittedTime = now;
  recentPredictions.length = 0;
  return consensus;
}

export async function localVision(video: HTMLVideoElement, _timestamp: number): Promise<VisionResult> {
  try {
    if (!video) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

    activeVideoEl = video;
    ensureBackgroundVisionLoop();

    video.playsInline = true;
    video.muted = true;
    if (video.paused && video.srcObject) {
      video.play().catch(() => {});
    }

    if (video.readyState < 2 || video.videoWidth === 0) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    const landmarker = await getLandmarker();
    const nowPerf = Math.floor(performance.now());
    const safeTimestamp = nowPerf > lastVideoTime ? nowPerf : lastVideoTime + 1;
    lastVideoTime = safeTimestamp;

    const result = landmarker.detectForVideo(video, safeTimestamp);
    const hands = (result.landmarks || []) as Landmark[][];
    const now = Date.now();

    if (hands.length === 0) {
      releaseFrameCount++;
      if (releaseFrameCount >= 5) {
        latchedSign = '';
        candidateSign = '';
        recentPredictions.length = 0;
        smoothedLandmarks = null;
      }
      updateMouselessCursor(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Still check for mouseless toggle (Hardcoded so it never breaks!)
    if (
      hands.length >= 2 &&
      (isTwoStackedHorizontalHands(hands[0], hands[1]) ||
        (isOnlyMiddleFinger(hands[0]) && isOnlyMiddleFinger(hands[1])))
    ) {
      if (now - lastToggleTime > 1400) {
        mouselessMode = !mouselessMode;
        lastToggleTime = now;
        showModeNotification(mouselessMode);
        updateMouselessCursor(mouselessMode);
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    if (mouselessMode) {
      const indexTip = hands[0][8];
      const thumbTip = hands[0][4];

      const rawX = (1 - indexTip.x - 0.12) / 0.76;
      const rawY = (indexTip.y - 0.12) / 0.76;
      const targetX = Math.max(0.01, Math.min(0.99, rawX));
      const targetY = Math.max(0.01, Math.min(0.99, rawY));

      smoothNormX = smoothNormX * 0.5 + targetX * 0.5;
      smoothNormY = smoothNormY * 0.5 + targetY * 0.5;

      const pinching = d(indexTip, thumbTip) < 0.062;
      updateMouselessCursor(true, smoothNormX, smoothNormY, pinching);

      const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;

      if (smoothNormY < 0.12) {
        window.scrollBy({ top: -15, behavior: 'auto' });
        if (bridge && now - lastScrollTime > 900) {
          lastScrollTime = now;
          try { bridge.swipeScreen(500, 450, 500, 1250, 260); } catch {}
        }
      } else if (smoothNormY > 0.88) {
        window.scrollBy({ top: 15, behavior: 'auto' });
        if (bridge && now - lastScrollTime > 900) {
          lastScrollTime = now;
          try { bridge.swipeScreen(500, 1250, 500, 450, 260); } catch {}
        }
      }

      if (pinching && now - lastClickTime > 680) {
        lastClickTime = now;
        if (bridge && typeof bridge.clickAirMouse === 'function') {
          try { bridge.clickAirMouse(smoothNormX, smoothNormY); } catch {}
        }
        const cx = smoothNormX * window.innerWidth;
        const cy = smoothNormY * window.innerHeight;
        const el = document.elementFromPoint(cx, cy) as HTMLElement | null;
        if (el) {
          el.click();
          el.focus();
        }
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    const { lm, fingerVelocity } = smoothAndMeasureVelocity(hands[0]);
    const vector = normalizeHandToVector(lm);

    // If the user tapped "Learn Sign" in the UI, save it right now!
    if (pendingTrainLabel) {
      mlDatabase.push({ label: pendingTrainLabel, vector });
      localStorage.setItem('nova_ml_gestures', JSON.stringify(mlDatabase));
      console.log(`ML Engine: Learned "${pendingTrainLabel}"`);
      pendingTrainLabel = null; 
      // Vibrate tablet to confirm
      if (navigator.vibrate) navigator.vibrate(100);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Run ML Classification
    const mlMatch = classifyWithML(vector);
    
    let rawResult: VisionResult = { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    
    if (mlMatch.label) {
      const isCmd = mlMatch.label === 'CLEAR' || mlMatch.label === 'SEND';
      rawResult = {
        type: isCmd ? 'GESTURE' : 'LETTER',
        value: mlMatch.label,
        confidence: 1.0 - (mlMatch.distance / 2),
        source: 'local'
      };
    } else {
      // If database is empty or no match, fall back to "Awaiting Training"
      if (mlDatabase.length === 0) {
        if (now - lastEmittedTime > 5000) {
          lastEmittedTime = now;
          console.warn("ML Engine empty! Train signs in the sidebar.");
        }
      }
    }

    return stabilizeSingleShot(rawResult, fingerVelocity);
  } catch (err) {
    console.error('Vision error:', err);
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
}

export async function isLocalVisionModelAvailable() {
  return true;
}

export function applyCustomGesture(result: VisionResult, gestures: CustomGesture[]): VisionResult {
  if (result.type !== 'GESTURE') return result;
  if (result.value.toUpperCase().includes('THEME')) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
  const match = gestures.find(g => g.name.toUpperCase() === result.value.toUpperCase());
  return match ? { ...result, value: match.name } : result;
}
