import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarkerPromise: Promise<HandLandmarker> | null = null;
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;

// Short motion trail buffers for dynamic letters J and Z
const indexTrail: { x: number; y: number; t: number }[] = [];
const pinkyTrail: { x: number; y: number; t: number }[] = [];

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      });
    })();
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

// Rotation-independent finger extension check
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

function recordTrail(trail: { x: number; y: number; t: number }[], pt: Landmark) {
  const now = Date.now();
  trail.push({ x: pt.x, y: pt.y, t: now });
  while (trail.length > 0 && now - trail[0].t > 650) {
    trail.shift();
  }
}

function getTrailMovement(trail: { x: number; y: number; t: number }[]) {
  if (trail.length < 5) return { dx: 0, dy: 0, total: 0 };
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of trail) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const dx = maxX - minX;
  const dy = maxY - minY;
  return { dx, dy, total: Math.hypot(dx, dy) };
}

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

// Complete 26-Letter ASL Classifier Matched to the Standard ASL Alphabet Chart
function classify(lm: Landmark[]): VisionResult {
  if (lm.length < 21) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

  recordTrail(indexTrail, lm[8]);
  recordTrail(pinkyTrail, lm[20]);

  const palm = Math.max(d(lm[0], lm[9]), 0.05);
  const nd = (a: Landmark, b: Landmark) => d(a, b) / palm;

  const index = extended(lm, 8, 6, 5);
  const middle = extended(lm, 12, 10, 9);
  const ring = extended(lm, 16, 14, 13);
  const pinky = extended(lm, 20, 18, 17);

  const thumbOut = d(lm[4], lm[17]) > d(lm[2], lm[17]) * 1.16;
  const indexHorizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y) * 1.15;
  const indexPointingDown = lm[8].y > lm[5].y + palm * 0.25;
  const middlePointingDown = lm[12].y > lm[9].y + palm * 0.22;

  // B: 4 fingers straight up, thumb folded across palm
  if (index && middle && ring && pinky) {
    return { type: 'LETTER', value: 'B', confidence: 0.9, source: 'local' };
  }

  // W: Index, middle, ring straight up; pinky curled
  if (index && middle && ring && !pinky) {
    return { type: 'LETTER', value: 'W', confidence: 0.9, source: 'local' };
  }

  // F: Index + thumb touching in circle; middle, ring, pinky straight up
  if (!index && middle && ring && pinky && nd(lm[4], lm[8]) < 0.65) {
    return { type: 'LETTER', value: 'F', confidence: 0.9, source: 'local' };
  }

  // I, J, Y: Pinky extended while index, middle, ring are curled
  if (!index && !middle && !ring && pinky) {
    if (thumbOut) {
      return { type: 'LETTER', value: 'Y', confidence: 0.9, source: 'local' };
    }
    const pinkyMove = getTrailMovement(pinkyTrail);
    const pinkyTilted = Math.abs(lm[20].x - lm[17].x) > Math.abs(lm[20].y - lm[17].y) * 0.65;
    if (pinkyMove.total > 0.06 || pinkyTilted) {
      return { type: 'LETTER', value: 'J', confidence: 0.86, source: 'local' };
    }
    return { type: 'LETTER', value: 'I', confidence: 0.88, source: 'local' };
  }

  // P: Index pointing sideways while middle finger angles downward (as shown in chart)
  if (index && indexHorizontal && middlePointingDown && !ring && !pinky) {
    return { type: 'LETTER', value: 'P', confidence: 0.86, source: 'local' };
  }

  // H, K, P, R, U, V: Index + Middle extended, Ring + Pinky curled
  if (index && middle && !ring && !pinky) {
    // H or P when oriented sideways
    if (indexHorizontal) {
      const isP = lm[12].y > lm[8].y + palm * 0.25;
      return { type: 'LETTER', value: isP ? 'P' : 'H', confidence: 0.87, source: 'local' };
    }
    // R: Index and middle fingers crossed over each other
    const crossed = (lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < 0 || nd(lm[8], lm[12]) < 0.19;
    if (crossed) {
      return { type: 'LETTER', value: 'R', confidence: 0.86, source: 'local' };
    }
    // K: Index and middle open with thumb pointing up between them near middle base
    const spread = nd(lm[8], lm[12]);
    if (spread > 0.28 && lm[4].y < lm[5].y && nd(lm[4], lm[10]) < 0.58) {
      return { type: 'LETTER', value: 'K', confidence: 0.86, source: 'local' };
    }
    // V vs U: V has fingers spread apart, U has fingers held together
    if (spread > 0.35) {
      return { type: 'LETTER', value: 'V', confidence: 0.9, source: 'local' };
    }
    return { type: 'LETTER', value: 'U', confidence: 0.88, source: 'local' };
  }

  // Q: Hand arched downward with index and thumb pointing down
  if (indexPointingDown && !middle && !ring && !pinky && lm[4].y > lm[2].y) {
    return { type: 'LETTER', value: 'Q', confidence: 0.85, source: 'local' };
  }

  // D, G, L, Z: Only Index extended
  if (index && !middle && !ring && !pinky) {
    // G: Index pointing sideways horizontally
    if (indexHorizontal) {
      return { type: 'LETTER', value: 'G', confidence: 0.87, source: 'local' };
    }
    // L: Thumb extended outward forming 90-degree angle with index
    if (thumbOut && nd(lm[4], lm[8]) > 0.95) {
      return { type: 'LETTER', value: 'L', confidence: 0.9, source: 'local' };
    }
    // Z: Index moving in a zig-zag / horizontal stroke
    const idxMove = getTrailMovement(indexTrail);
    if (idxMove.dx > 0.065) {
      return { type: 'LETTER', value: 'Z', confidence: 0.85, source: 'local' };
    }
    // D: Thumb tip touching curled middle/ring fingers
    if (nd(lm[4], lm[12]) < 0.58 || nd(lm[4], lm[10]) < 0.58) {
      return { type: 'LETTER', value: 'D', confidence: 0.87, source: 'local' };
    }
    return { type: 'LETTER', value: 'Z', confidence: 0.82, source: 'local' };
  }

  // Fist / Curved shapes: A, C, E, M, N, O, S, T, X
  if (!index && !middle && !ring && !pinky) {
    // X: Index knuckle raised with index tip hooked downward, other fingers tightly curled
    const indexKnuckleRaised = lm[6].y < lm[10].y - palm * 0.15 && lm[8].y > lm[6].y;
    if (indexKnuckleRaised) {
      return { type: 'LETTER', value: 'X', confidence: 0.85, source: 'local' };
    }

    // O: Thumb tip touching curled index and middle fingertips
    if (nd(lm[4], lm[8]) < 0.42 && nd(lm[4], lm[12]) < 0.48) {
      return { type: 'LETTER', value: 'O', confidence: 0.87, source: 'local' };
    }

    // C: Fingers curved in an open arc with C-gap between thumb and index tips
    const fingersHalfOpen = nd(lm[8], lm[0]) > 1.15 && nd(lm[12], lm[0]) > 1.15;
    if (fingersHalfOpen && nd(lm[4], lm[8]) >= 0.42 && nd(lm[4], lm[8]) < 1.3) {
      return { type: 'LETTER', value: 'C', confidence: 0.84, source: 'local' };
    }

    // E: All 4 fingertips curled above the horizontal thumb
    const tipsAboveThumb =
      lm[8].y < lm[4].y &&
      lm[12].y < lm[4].y &&
      lm[16].y < lm[4].y &&
      nd(lm[4], lm[12]) < 0.65;
    if (tipsAboveThumb) {
      return { type: 'LETTER', value: 'E', confidence: 0.83, source: 'local' };
    }

    // A: Thumb resting vertically along the outside of the index finger (thumb tip high)
    if (thumbOut && lm[4].y < lm[6].y) {
      return { type: 'LETTER', value: 'A', confidence: 0.87, source: 'local' };
    }

    // T, N, M, S: Distinguish by where the thumb tip sits across the curled fingers
    const dIndex = nd(lm[4], lm[6]);
    const dMiddle = nd(lm[4], lm[10]);
    const dRing = nd(lm[4], lm[14]);
    const dPinky = nd(lm[4], lm[18]);

    // M: Thumb tucked furthest over near ring/pinky fingers
    if (dRing < dIndex && (dRing < 0.52 || dPinky < 0.6)) {
      return { type: 'LETTER', value: 'M', confidence: 0.81, source: 'local' };
    }
    // N: Thumb tucked between middle and ring fingers
    if (dMiddle < dIndex && dMiddle < 0.48) {
      return { type: 'LETTER', value: 'N', confidence: 0.81, source: 'local' };
    }
    // T: Thumb tucked between index and middle with tip pointing up
    if (dIndex < 0.46 && lm[4].y < lm[10].y) {
      return { type: 'LETTER', value: 'T', confidence: 0.82, source: 'local' };
    }

    // S: Clenched fist with thumb crossed in front of fingers
    return { type: 'LETTER', value: 'S', confidence: 0.83, source: 'local' };
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

  // Two-Hand Middle Finger Gesture toggles Mouseless Control Mode
  if (hands.length >= 2 && isOnlyMiddleFinger(hands[0]) && isOnlyMiddleFinger(hands[1])) {
    if (now - lastToggleTime > 1500) {
      mouselessMode = !mouselessMode;
      lastToggleTime = now;
      updateMouselessCursor(mouselessMode);
    }
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  const primaryHand = hands[0];

  if (mouselessMode) {
    const indexTip = primaryHand[8];
    const thumbTip = primaryHand[4];
    const cx = (1 - indexTip.x) * window.innerWidth;
    const cy = indexTip.y * window.innerHeight;
    const pinching = d(indexTip, thumbTip) < 0.05;

    updateMouselessCursor(true, cx, cy, pinching);

    if (indexTip.y < 0.2) window.scrollBy({ top: -16, behavior: 'auto' });
    if (indexTip.y > 0.8) window.scrollBy({ top: 16, behavior: 'auto' });

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
    return true;
  }
}

export function applyCustomGesture(result: VisionResult, gestures: CustomGesture[]): VisionResult {
  if (result.type !== 'GESTURE') return result;
  if (result.value.toUpperCase().includes('THEME')) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
  const match = gestures.find(
    g => g.name.toUpperCase() === result.value.toUpperCase() && !g.name.toUpperCase().includes('THEME')
  );
  return match ? { ...result, value: match.name } : result;
}
