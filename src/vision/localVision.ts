import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const WASM_PATH = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
const MODEL_PATH = httpsUrl(
  'storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
);

// ============================================================================
// PALM-RAY AIR MOUSE TUNING CONSTANTS
// ============================================================================
const ROI_X_MIN = 0.15;
const ROI_X_MAX = 0.85;
const ROI_Y_MIN = 0.15;
const ROI_Y_MAX = 0.85;

const PINCH_DOWN_THRESH = 0.052;
const PINCH_UP_THRESH = 0.075;
const CLICK_DEBOUNCE_MS = 450;
const SCROLL_DEBOUNCE_MS = 280;

let mouselessMode = false;
let isPinching = false;
let lastClickTime = 0;
let lastScrollTime = 0;
let lastToggleTime = 0;

// Dynamic filter coordinates
let filteredX = 0.5;
let filteredY = 0.5;
let prevTargetX = 0.5;
let prevTargetY = 0.5;

// ML Database state
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
  } catch {
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

function normalizeHandToVector(lm: Landmark[]): number[] {
  const wrist = lm[0];
  let maxDist = 0.0001;
  const centered = lm.map(p => {
    const dx = p.x - wrist.x;
    const dy = p.y - wrist.y;
    const dz = (p.z || 0) - (wrist.z || 0);
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxDist) maxDist = dist;
    return { x: dx, y: dy, z: dz };
  });

  const vector: number[] = [];
  for (const p of centered) {
    vector.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  }
  return vector;
}

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

  if (minDist < 1.15) {
    return { label: bestLabel, distance: minDist };
  }
  return { label: '', distance: minDist };
}

// ============================================================================
// PALM-RAY VECTOR & DYNAMIC SMOOTHING CALCULATION
// ============================================================================
const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

function computePalmRayTarget(lm: Landmark[]): { targetX: number; targetY: number; isScrolling: boolean } {
  const wrist = lm[0];
  const indexMcp = lm[5];
  const indexPip = lm[6];
  const indexTip = lm[8];
  const middleTip = lm[12];
  const palmCenter = {
    x: (wrist.x + indexMcp.x + lm[17].x) / 3,
    y: (wrist.y + indexMcp.y + lm[17].y) / 3
  };

  // Two-finger scroll detection: index and middle fingers both extended upward
  const isIndexUp = indexTip.y < indexPip.y;
  const isMiddleUp = middleTip.y < lm[10].y;
  const isRingCurled = lm[16].y > lm[14].y;
  const isScrolling = isIndexUp && isMiddleUp && isRingCurled;

  // Use knuckle vector to stabilize against fingertip twitching during pinch
  const rayProjX = 1.0 - (indexMcp.x * 0.65 + indexTip.x * 0.35);
  const rayProjY = indexMcp.y * 0.65 + indexTip.y * 0.35;

  // Remap active camera ROI to tablet edges
  const normX = (rayProjX - ROI_X_MIN) / (ROI_X_MAX - ROI_X_MIN);
  const normY = (rayProjY - ROI_Y_MIN) / (ROI_Y_MAX - ROI_Y_MIN);

  return {
    targetX: Math.max(0.005, Math.min(0.995, normX)),
    targetY: Math.max(0.005, Math.min(0.995, normY)),
    isScrolling
  };
}

function updateAdaptiveMouseFilter(rawX: number, rawY: number): { x: number; y: number } {
  const velocity = Math.hypot(rawX - prevTargetX, rawY - prevTargetY);
  prevTargetX = rawX;
  prevTargetY = rawY;

  // Dynamic alpha: high speed = fast tracking; low speed = heavy jitter reduction
  let alpha = 0.55;
  if (velocity < 0.004) {
    alpha = 0.15; // Jitter suppression deadband
  } else if (velocity < 0.02) {
    alpha = 0.35;
  } else if (velocity > 0.08) {
    alpha = 0.85; // Low-latency fast tracking
  }

  filteredX = filteredX * (1 - alpha) + rawX * alpha;
  filteredY = filteredY * (1 - alpha) + rawY * alpha;

  return { x: filteredX, y: filteredY };
}

// ============================================================================
// MEDIAPIPE CORE
// ============================================================================
let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let activeVideoEl: HTMLVideoElement | null = null;
let bgIntervalId: any = null;

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
          minHandDetectionConfidence: 0.45,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45
        });
      } catch {
        return await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.45,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45
        });
      }
    })();
  }
  return landmarkerPromise;
}

function updateMouselessCursor(visible: boolean, normX = 0.5, normY = 0.5, pinching = false) {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (bridge && typeof bridge.updateAirMouse === 'function') {
    try {
      bridge.updateAirMouse(visible, normX, normY, pinching);
    } catch {}
  }
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
      }
      updateMouselessCursor(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Toggle Mouseless Mode: 2 Hands horizontal or middle finger gesture
    if (hands.length >= 2) {
      const h1 = hands[0];
      const h2 = hands[1];
      const isHoriz1 = Math.abs(h1[9].x - h1[0].x) > Math.abs(h1[9].y - h1[0].y) * 0.75;
      const isHoriz2 = Math.abs(h2[9].x - h2[0].x) > Math.abs(h2[9].y - h2[0].y) * 0.75;
      if (isHoriz1 && isHoriz2 && Math.abs(h1[9].y - h2[9].y) > 0.1) {
        if (now - lastToggleTime > 1400) {
          mouselessMode = !mouselessMode;
          lastToggleTime = now;
          const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
          if (mouselessMode && bridge && typeof bridge.enableOverlayBubble === 'function') {
            try { bridge.enableOverlayBubble(); } catch {}
          }
          updateMouselessCursor(mouselessMode);
        }
        return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
      }
    }

    // PALM-RAY ENGINE MOUSE EXECUTION
    if (mouselessMode) {
      const hand = hands[0];
      const { targetX, targetY, isScrolling } = computePalmRayTarget(hand);
      const filtered = updateAdaptiveMouseFilter(targetX, targetY);

      // Pinch distance with hysteresis
      const pinchDist = d(hand[8], hand[4]);
      if (!isPinching && pinchDist < PINCH_DOWN_THRESH) {
        isPinching = true;
      } else if (isPinching && pinchDist > PINCH_UP_THRESH) {
        isPinching = false;
      }

      updateMouselessCursor(true, filtered.x, filtered.y, isPinching);

      const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;

      // Two-Finger Scroll
      if (isScrolling && now - lastScrollTime > SCROLL_DEBOUNCE_MS) {
        lastScrollTime = now;
        const scrollDelta = filtered.y < 0.5 ? -350 : 350;
        window.scrollBy({ top: scrollDelta, behavior: 'smooth' });
        if (bridge && typeof bridge.swipeScreen === 'function') {
          try {
            if (filtered.y < 0.5) bridge.swipeScreen(500, 350, 500, 1200, 240);
            else bridge.swipeScreen(500, 1200, 500, 350, 240);
          } catch {}
        }
      }

      // Pinch Click Execution
      if (isPinching && now - lastClickTime > CLICK_DEBOUNCE_MS) {
        lastClickTime = now;
        if (bridge && typeof bridge.clickAirMouse === 'function') {
          try { bridge.clickAirMouse(filtered.x, filtered.y); } catch {}
        }
        const cx = filtered.x * window.innerWidth;
        const cy = filtered.y * window.innerHeight;
        const el = document.elementFromPoint(cx, cy) as HTMLElement | null;
        if (el) {
          el.click();
          el.focus();
        }
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // ML TRAINING & CLASSIFICATION
    const vector = normalizeHandToVector(hands[0]);

    if (pendingTrainLabel) {
      mlDatabase.push({ label: pendingTrainLabel, vector });
      localStorage.setItem('nova_ml_gestures', JSON.stringify(mlDatabase));
      pendingTrainLabel = null;
      if (navigator.vibrate) navigator.vibrate(100);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    const mlMatch = classifyWithML(vector);
    let rawResult: VisionResult = { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

    if (mlMatch.label) {
      rawResult = {
        type: mlMatch.label === 'CLEAR' || mlMatch.label === 'SEND' ? 'GESTURE' : 'LETTER',
        value: mlMatch.label,
        confidence: Math.max(0.5, 1.0 - mlMatch.distance / 2),
        source: 'local'
      };
    }

    // Consensus Single-Shot Latch
    recentPredictions.push(rawResult);
    if (recentPredictions.length > 6) recentPredictions.shift();

    const counts = new Map<string, number>();
    for (const r of recentPredictions) {
      if (r.value) counts.set(r.value, (counts.get(r.value) || 0) + 1);
    }

    let topSign = '';
    let topCount = 0;
    for (const [k, v] of counts.entries()) {
      if (v > topCount) {
        topSign = k;
        topCount = v;
      }
    }

    if (topCount >= 4 && topSign && topSign !== latchedSign) {
      if (topSign !== candidateSign) {
        candidateSign = topSign;
        candidateStartTime = now;
      } else if (now - candidateStartTime > 420 && now - lastEmittedTime > 1200) {
        latchedSign = topSign;
        lastEmittedTime = now;
        recentPredictions.length = 0;
        return {
          type: topSign === 'CLEAR' || topSign === 'SEND' ? 'GESTURE' : 'LETTER',
          value: topSign,
          confidence: 0.95,
          source: 'local'
        };
      }
    }

    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
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
  const match = gestures.find(g => g.name.toUpperCase() === result.value.toUpperCase());
  return match ? { ...result, value: match.name } : result;
}
