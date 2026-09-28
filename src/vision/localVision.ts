import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarkerPromise: Promise<HandLandmarker> | null = null;
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2, // Tracks up to 2 hands for the two-hand middle finger mode toggle
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      });
    })();
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

function extended(lm: Landmark[], tip: number, pip: number, mcp: number) {
  const wrist = lm[0];
  return d(lm[tip], wrist) > d(lm[pip], wrist) * 1.08 && d(lm[tip], lm[mcp]) > d(lm[pip], lm[mcp]);
}

function isOnlyMiddleFinger(lm: Landmark[]): boolean {
  if (!lm || lm.length < 21) return false;
  const index = extended(lm, 8, 6, 5);
  const middle = extended(lm, 12, 10, 9);
  const ring = extended(lm, 16, 14, 13);
  const pinky = extended(lm, 20, 18, 17);
  return middle && !index && !ring && !pinky;
}

// Manages the floating cursor in Mouseless Mode without changing your original UI
function updateMouselessCursor(visible: boolean, x = 0, y = 0, clicking = false) {
  let cursor = document.getElementById('nova-mouseless-cursor');
  if (!visible) {
    if (cursor) cursor.style.display = 'none';
    return;
  }
  if (!cursor) {
    cursor = document.createElement('div');
    cursor.id = 'nova-mouseless-cursor';
    Object.assign(cursor.style, {
      position: 'fixed',
      width: '22px',
      height: '22px',
      borderRadius: '50%',
      background: 'rgba(16, 185, 129, 0.85)',
      border: '2px solid #ffffff',
      pointerEvents: 'none',
      zIndex: '99999',
      transition: 'transform 0.08s ease'
    });
    document.body.appendChild(cursor);
  }
  cursor.style.display = 'block';
  cursor.style.left = `${x - 11}px`;
  cursor.style.top = `${y - 11}px`;
  cursor.style.transform = clicking ? 'scale(0.7)' : 'scale(1)';
}

// Full 26-Letter ASL Classifier (A-Z)
function classify(lm: Landmark[]): VisionResult {
  if (lm.length < 21) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

  // Normalize distances by palm size so signs work at any camera distance
  const palm = Math.max(d(lm[0], lm[9]), 0.05);
  const nd = (a: Landmark, b: Landmark) => d(a, b) / palm;

  const index = extended(lm, 8, 6, 5);
  const middle = extended(lm, 12, 10, 9);
  const ring = extended(lm, 16, 14, 13);
  const pinky = extended(lm, 20, 18, 17);

  const thumbOut = d(lm[4], lm[17]) > d(lm[2], lm[17]) * 1.18;
  const thumbUp = lm[4].y < lm[3].y && lm[3].y < lm[2].y;
  const thumbDown = lm[4].y > lm[0].y + palm * 0.45 && !index && !middle && !ring && !pinky;
  const isHorizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y) * 1.2;
  const indexHooked = !index && d(lm[8], lm[0]) > d(lm[12], lm[0]) * 1.15 && lm[6].y < lm[5].y;

  // Clear Gesture (Thumb pointing down)
  if (thumbDown) return { type: 'GESTURE', value: 'CLEAR', confidence: 0.9, source: 'local' };

  // 1. All 4 fingers extended -> B
  if (index && middle && ring && pinky) {
    return { type: 'LETTER', value: 'B', confidence: 0.88, source: 'local' };
  }

  // 2. 3 fingers extended -> W or F
  if (index && middle && ring && !pinky) {
    return { type: 'LETTER', value: 'W', confidence: 0.89, source: 'local' };
  }
  if (!index && middle && ring && pinky && nd(lm[4], lm[8]) < 0.55) {
    return { type: 'LETTER', value: 'F', confidence: 0.88, source: 'local' };
  }

  // 3. Pinky combinations -> Y, I, J
  if (!index && !middle && !ring && pinky) {
    if (thumbOut) return { type: 'LETTER', value: 'Y', confidence: 0.9, source: 'local' };
    const pinkySideways = Math.abs(lm[20].x - lm[17].x) > Math.abs(lm[20].y - lm[17].y) * 0.75;
    return {
      type: 'LETTER',
      value: pinkySideways ? 'J' : 'I',
      confidence: 0.86,
      source: 'local'
    };
  }

  // 4. Index + Middle extended -> H, P, R, K, V, U
  if (index && middle && !ring && !pinky) {
    if (isHorizontal) {
      const pointingDown = lm[8].y > lm[0].y;
      return { type: 'LETTER', value: pointingDown ? 'P' : 'H', confidence: 0.85, source: 'local' };
    }
    // R: Index and middle tips crossed or overlapping tightly
    if ((lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < 0 || nd(lm[8], lm[12]) < 0.18) {
      return { type: 'LETTER', value: 'R', confidence: 0.84, source: 'local' };
    }
    // K: Thumb extended upward between index and middle
    if (thumbOut && nd(lm[4], lm[10]) < 0.6) {
      return { type: 'LETTER', value: 'K', confidence: 0.85, source: 'local' };
    }
    // V vs U: Spread vs together
    if (nd(lm[8], lm[12]) > 0.36) {
      return { type: 'LETTER', value: 'V', confidence: 0.9, source: 'local' };
    }
    return { type: 'LETTER', value: 'U', confidence: 0.86, source: 'local' };
  }

  // 5. Only Index extended -> G, Q, L, D, Z
  if (index && !middle && !ring && !pinky) {
    if (isHorizontal) {
      const pointingDown = lm[8].y > lm[5].y;
      return { type: 'LETTER', value: pointingDown ? 'Q' : 'G', confidence: 0.85, source: 'local' };
    }
    if (thumbOut) {
      return { type: 'LETTER', value: 'L', confidence: 0.9, source: 'local' };
    }
    if (nd(lm[4], lm[12]) < 0.52) {
      return { type: 'LETTER', value: 'D', confidence: 0.86, source: 'local' };
    }
    return { type: 'LETTER', value: 'Z', confidence: 0.83, source: 'local' };
  }

  // 6. All 4 fingers curled (Fist / Curved shapes) -> X, C, O, A, E, T, N, M, S
  if (!index && !middle && !ring && !pinky) {
    if (indexHooked) {
      return { type: 'LETTER', value: 'X', confidence: 0.84, source: 'local' };
    }
    if (thumbOut && nd(lm[4], lm[8]) > 0.6 && d(lm[8], lm[0]) > d(lm[5], lm[0])) {
      return { type: 'LETTER', value: 'C', confidence: 0.82, source: 'local' };
    }
    if (nd(lm[4], lm[8]) < 0.42 && nd(lm[4], lm[12]) < 0.5) {
      return { type: 'LETTER', value: 'O', confidence: 0.85, source: 'local' };
    }
    if (thumbOut && thumbUp) {
      return { type: 'LETTER', value: 'A', confidence: 0.86, source: 'local' };
    }
    if (lm[4].y > lm[8].y && lm[4].y > lm[12].y) {
      return { type: 'LETTER', value: 'E', confidence: 0.82, source: 'local' };
    }
    // Distinguish T, N, M, S by where the thumb rests across the curled fingers
    if (nd(lm[4], lm[6]) < 0.42) {
      return { type: 'LETTER', value: 'T', confidence: 0.8, source: 'local' };
    }
    if (nd(lm[4], lm[10]) < 0.42) {
      return { type: 'LETTER', value: 'N', confidence: 0.8, source: 'local' };
    }
    if (nd(lm[4], lm[14]) < 0.48) {
      return { type: 'LETTER', value: 'M', confidence: 0.8, source: 'local' };
    }
    return { type: 'LETTER', value: 'S', confidence: 0.82, source: 'local' };
  }

  return { type: 'UNKNOWN', value: '', confidence: 0.2, source: 'local' };
}

export async function localVision(video: HTMLVideoElement, timestamp: number): Promise<VisionResult> {
  const landmarker = await getLandmarker();
  const result = landmarker.detectForVideo(video, timestamp);
  const hands = (result.landmarks || []) as Landmark[][];
  const now = Date.now();

  if (hands.length === 0) {
    updateMouselessCursor(false);
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  // Check Two-Hand Middle Finger Gesture to toggle Mouseless Control Mode
  if (hands.length >= 2 && isOnlyMiddleFinger(hands[0]) && isOnlyMiddleFinger(hands[1])) {
    if (now - lastToggleTime > 1500) {
      mouselessMode = !mouselessMode;
      lastToggleTime = now;
      updateMouselessCursor(mouselessMode);
    }
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  const primaryHand = hands[0];

  // If Mouseless Mode is active: move cursor, scroll up/down, pinch to click
  if (mouselessMode) {
    const indexTip = primaryHand[8];
    const thumbTip = primaryHand[4];
    const cx = (1 - indexTip.x) * window.innerWidth;
    const cy = indexTip.y * window.innerHeight;
    const pinching = d(indexTip, thumbTip) < 0.05;

    updateMouselessCursor(true, cx, cy, pinching);

    // Scroll page or chat container when pointing near top or bottom edges
    if (indexTip.y < 0.2) window.scrollBy({ top: -16, behavior: 'auto' });
    if (indexTip.y > 0.8) window.scrollBy({ top: 16, behavior: 'auto' });

    // Pinch thumb + index to click any button on screen
    if (pinching && now - lastClickTime > 750) {
      lastClickTime = now;
      const el = document.elementFromPoint(cx, cy) as HTMLElement | null;
      if (el) {
        el.click();
        el.focus();
      }
    }
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  return classify(primaryHand);
}

export async function isLocalVisionModelAvailable() {
  try {
    const response = await fetch(MODEL_PATH, { method: 'HEAD' });
    return response.ok;
  } catch {
    // Allow cached offline model to run even when internet is disconnected
    return true;
  }
}

export function applyCustomGesture(result: VisionResult, gestures: CustomGesture[]): VisionResult {
  if (result.type !== 'GESTURE') return result;
  // Permanently ignore any theme-switching gestures
  if (result.value.toUpperCase().includes('THEME')) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
  const match = gestures.find(
    g => g.name.toUpperCase() === result.value.toUpperCase() && !g.name.toUpperCase().includes('THEME')
  );
  return match ? { ...result, value: match.name } : result;
}
