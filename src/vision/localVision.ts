import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Timing tuned to work with your UI's frame buffer:
// 1. HOLD_DELAY_MS (450ms): Wait while you form the sign so it doesn't trigger too fast
// 2. EMIT_WINDOW_MS (280ms): Send the sign for ~5-6 frames so the UI types it exactly ONCE
// 3. REPEAT_COOLDOWN_MS (1800ms): Block repeats so holding a sign never types "AAAA..."
const HOLD_DELAY_MS = 450;
const EMIT_WINDOW_MS = 280;
const REPEAT_COOLDOWN_MS = 1800;

let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;
let lastActionTime = 0;

let candidateSign = '';
let candidateStartTime = 0;
let mismatchFrames = 0;

// Short motion history for dynamic ASL letters J and Z
const indexTrail: { x: number; y: number; t: number }[] = [];
const pinkyTrail: { x: number; y: number; t: number }[] = [];

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_PATH).then((vision) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.35
      })
    );
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

function recordTrail(trail: { x: number; y: number; t: number }[], pt: Landmark) {
  const now = Date.now();
  trail.push({ x: pt.x, y: pt.y, t: now });
  while (trail.length > 0 && now - trail[0].t > 550) {
    trail.shift();
  }
}

function getTrailSpan(trail: { x: number; y: number; t: number }[]) {
  if (trail.length < 4) return { dx: 0, dy: 0, total: 0 };
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of trail) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { dx: maxX - minX, dy: maxY - minY, total: Math.hypot(maxX - minX, maxY - minY) };
}

// 1. Detects the 2nd Image Gesture: Two horizontal hands held one above the other
function isTwoStackedHorizontalHands(h1: Landmark[], h2: Landmark[]): boolean {
  if (!h1 || !h2 || h1.length < 21 || h2.length < 21) return false;

  const isHoriz = (lm: Landmark[]) =>
    Math.abs(lm[9].x - lm[0].x) > Math.abs(lm[9].y - lm[0].y) * 0.75 &&
    Math.abs(lm[12].x - lm[0].x) > Math.abs(lm[12].y - lm[0].y) * 0.75;

  const verticalGap = Math.abs(h1[9].y - h2[9].y);
  const horizontalOverlap = Math.abs(h1[9].x - h2[9].x);

  return isHoriz(h1) && isHoriz(h2) && verticalGap > 0.1 && horizontalOverlap < 0.5;
}

// 2. Detects Two-Hand Middle Finger Gesture (works for both left & right hands, palm in or out)
function isOnlyMiddleFinger(lm: Landmark[]): boolean {
  if (!lm || lm.length < 21) return false;
  const palm = Math.max(d(lm[0], lm[9]), 0.05);
  const middleUp = lm[12].y < lm[10].y && d(lm[12], lm[0]) > d(lm[10], lm[0]) * 1.08;
  // Middle fingertip must be noticeably higher than index, ring, and pinky fingertips
  const highestTip =
    lm[12].y < lm[8].y - palm * 0.28 &&
    lm[12].y < lm[16].y - palm * 0.28 &&
    lm[12].y < lm[20].y - palm * 0.28;
  return middleUp && highestTip;
}

// Shows a visible floating cursor and mode badge when Mouseless Mode is toggled
function showModeNotification(isMouseless: boolean) {
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
    ? '🖱️ Mouseless Mode ON (Point to move/scroll, Pinch to click)'
    : '✋ ASL Sign Mode ON (26 Letters A–Z)';
  badge.style.opacity = '1';
  setTimeout(() => {
    if (badge && !mouselessMode) badge.style.opacity = '0';
  }, 2200);
}

function updateMouselessCursor(visible: boolean, x = 0, y = 0, pinching = false) {
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
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      background: 'rgba(239, 68, 68, 0.88)',
      border: '3px solid #ffffff',
      boxShadow: '0 0 12px rgba(239, 68, 68, 0.9)',
      pointerEvents: 'none',
      zIndex: '99999',
      transition: 'transform 0.08s ease'
    });
    document.body.appendChild(cursor);
  }
  cursor.style.display = 'block';
  cursor.style.left = `${x - 12}px`;
  cursor.style.top = `${y - 12}px`;
  cursor.style.transform = pinching ? 'scale(0.65)' : 'scale(1)';
}

function triggerInputAction(action: 'CLEAR' | 'SEND') {
  const now = Date.now();
  if (now - lastActionTime < 1200) return;
  lastActionTime = now;

  const inputEl = document.querySelector(
    'input[type="text"], input:not([type="password"]):not([type="file"]), textarea'
  ) as HTMLInputElement | HTMLTextAreaElement | null;

  if (action === 'CLEAR' && inputEl) {
    const proto =
      inputEl instanceof HTMLTextAreaElement
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
    const nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (nativeSetter) {
      nativeSetter.call(inputEl, '');
      inputEl.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
      inputEl.value = '';
    }
  }

  if (action === 'SEND') {
    if (inputEl) {
      inputEl.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true })
      );
    }
    const buttons = Array.from(document.querySelectorAll('button'));
    const sendBtn =
      buttons.find((b) => b.type === 'submit' || /send/i.test(b.textContent || '') || b.id === 'nova-send-btn') ||
      buttons[buttons.length - 1];
    if (sendBtn) sendBtn.click();
  }
}

// Complete 26-Letter ASL Classifier (A-Z) matching your ASL Alphabet Chart
function classifyRaw(lm: Landmark[]): VisionResult {
  if (!lm || lm.length < 21) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  recordTrail(indexTrail, lm[8]);
  recordTrail(pinkyTrail, lm[20]);

  const wrist = lm[0];
  const palm = Math.max(d(wrist, lm[9]), 0.05);
  const nd = (a: Landmark, b: Landmark) => d(a, b) / palm;

  // Upright finger checks
  const indexUp = lm[8].y < lm[6].y && nd(lm[8], wrist) > nd(lm[6], wrist) * 1.04;
  const middleUp = lm[12].y < lm[10].y && nd(lm[12], wrist) > nd(lm[10], wrist) * 1.04;
  const ringUp = lm[16].y < lm[14].y && nd(lm[16], wrist) > nd(lm[14], wrist) * 1.04;
  const pinkyUp = lm[20].y < lm[18].y && nd(lm[20], wrist) > nd(lm[18], wrist) * 1.04;

  // Orientation-independent extension checks (for sideways/downward signs G, H, P, Q)
  const indexExt = nd(lm[8], wrist) > nd(lm[6], wrist) * 1.06 && nd(lm[8], lm[5]) > nd(lm[6], lm[5]) * 1.04;
  const middleExt = nd(lm[12], wrist) > nd(lm[10], wrist) * 1.06 && nd(lm[12], lm[9]) > nd(lm[10], lm[9]) * 1.04;
  const ringExt = nd(lm[16], wrist) > nd(lm[14], wrist) * 1.06;
  const pinkyExt = nd(lm[20], wrist) > nd(lm[18], wrist) * 1.06;

  const thumbOut = d(lm[4], lm[17]) > d(lm[2], lm[17]) * 1.14;
  const thumbUp = lm[4].y < lm[3].y && lm[3].y < lm[2].y;
  const indexHalf = !indexUp && lm[6].y < lm[5].y && lm[8].y >= lm[6].y;
  const isHorizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y) * 1.15;

  // 1. Thumbs Down (👎) -> CLEAR ALL TEXT
  const allFingersClosed = !indexUp && !middleUp && !ringUp && !pinkyUp;
  const thumbPointingDown =
    lm[4].y > lm[3].y &&
    lm[4].y > lm[5].y + palm * 0.28 &&
    lm[4].y > wrist.y + palm * 0.32;
  if (thumbPointingDown && allFingersClosed && !indexExt) {
    return { type: 'GESTURE', value: 'CLEAR', confidence: 0.95, source: 'local' };
  }

  // 2. Yo-Yo Sign (🤘: Index + Pinky up, Middle + Ring curled) -> SEND MESSAGE
  if (indexUp && !middleUp && !ringUp && pinkyUp) {
    return { type: 'GESTURE', value: 'SEND', confidence: 0.95, source: 'local' };
  }

  // 3. Sideways & Downward ASL Signs: Q, P, H, G
  if (lm[8].y > lm[5].y + palm * 0.3 && lm[4].y > lm[2].y + palm * 0.2 && !middleUp && !ringUp && !pinkyUp) {
    return { type: 'LETTER', value: 'Q', confidence: 0.88, source: 'local' };
  }
  if (isHorizontal && indexExt && !ringExt && !pinkyExt) {
    if (middleExt) {
      const middleDropped = lm[12].y > lm[8].y + palm * 0.22 || lm[12].y > lm[9].y + palm * 0.18;
      return { type: 'LETTER', value: middleDropped ? 'P' : 'H', confidence: 0.89, source: 'local' };
    }
    if (lm[12].y > lm[9].y + palm * 0.25) {
      return { type: 'LETTER', value: 'P', confidence: 0.88, source: 'local' };
    }
    return { type: 'LETTER', value: 'G', confidence: 0.89, source: 'local' };
  }

  // 4. All 4 fingers straight up -> B
  if (indexUp && middleUp && ringUp && pinkyUp) {
    return { type: 'LETTER', value: 'B', confidence: 0.93, source: 'local' };
  }

  // 5. 3 fingers up -> W or F
  if (indexUp && middleUp && ringUp && !pinkyUp) {
    return { type: 'LETTER', value: 'W', confidence: 0.93, source: 'local' };
  }
  if (!indexUp && middleUp && ringUp && pinkyUp) {
    return { type: 'LETTER', value: 'F', confidence: 0.92, source: 'local' };
  }

  // 6. Only Pinky up -> Y, J, I
  if (!indexUp && !middleUp && !ringUp && (pinkyUp || pinkyExt)) {
    if (thumbOut) {
      return { type: 'LETTER', value: 'Y', confidence: 0.93, source: 'local' };
    }
    const pMove = getTrailSpan(pinkyTrail);
    const pinkyTilted = Math.abs(lm[20].x - lm[17].x) > Math.abs(lm[20].y - lm[17].y) * 0.62;
    if (pMove.total > 0.055 || pinkyTilted) {
      return { type: 'LETTER', value: 'J', confidence: 0.89, source: 'local' };
    }
    return { type: 'LETTER', value: 'I', confidence: 0.91, source: 'local' };
  }

  // 7. Index + Middle up -> R, K, V, U
  if (indexUp && middleUp && !ringUp && !pinkyUp) {
    const spread = nd(lm[8], lm[12]);
    // R: Index and middle crossed or overlapping tightly
    if ((lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < 0 || spread < 0.22) {
      return { type: 'LETTER', value: 'R', confidence: 0.89, source: 'local' };
    }
    // K: Thumb tucked up between index and middle
    if (thumbOut && nd(lm[4], lm[10]) < 0.72) {
      return { type: 'LETTER', value: 'K', confidence: 0.89, source: 'local' };
    }
    // V: Spread apart vs U: Held together
    if (spread > 0.42) {
      return { type: 'LETTER', value: 'V', confidence: 0.93, source: 'local' };
    }
    return { type: 'LETTER', value: 'U', confidence: 0.9, source: 'local' };
  }

  // 8. Only Index up -> L, Z, D
  if (indexUp && !middleUp && !ringUp && !pinkyUp) {
    if (thumbOut && nd(lm[4], lm[8]) > 0.9) {
      return { type: 'LETTER', value: 'L', confidence: 0.94, source: 'local' };
    }
    const iMove = getTrailSpan(indexTrail);
    if (iMove.dx > 0.055) {
      return { type: 'LETTER', value: 'Z', confidence: 0.89, source: 'local' };
    }
    if (nd(lm[4], lm[12]) < 0.68) {
      return { type: 'LETTER', value: 'D', confidence: 0.9, source: 'local' };
    }
    return { type: 'LETTER', value: 'Z', confidence: 0.87, source: 'local' };
  }

  // 9. Closed Fist / Curved Hand Shapes -> C, X, O, A, E, T, N, M, S
  if (allFingersClosed) {
    const thumbIndex = nd(lm[4], lm[8]);
    const thumbMiddle = nd(lm[4], lm[12]);

    if (indexHalf) {
      if (thumbOut && thumbIndex > 0.65) {
        return { type: 'LETTER', value: 'C', confidence: 0.88, source: 'local' };
      }
      return { type: 'LETTER', value: 'X', confidence: 0.88, source: 'local' };
    }
    if (thumbIndex < 0.52 && thumbMiddle < 0.65) {
      return { type: 'LETTER', value: 'O', confidence: 0.9, source: 'local' };
    }
    if (thumbOut && thumbIndex > 0.95 && nd(lm[8], wrist) > 1.05) {
      return { type: 'LETTER', value: 'C', confidence: 0.87, source: 'local' };
    }
    if (thumbOut && thumbUp) {
      return { type: 'LETTER', value: 'A', confidence: 0.93, source: 'local' };
    }
    if (lm[4].y > lm[8].y && lm[4].y > lm[12].y && thumbMiddle < 0.72) {
      return { type: 'LETTER', value: 'E', confidence: 0.88, source: 'local' };
    }
    if (nd(lm[4], lm[6]) < 0.48) {
      return { type: 'LETTER', value: 'T', confidence: 0.86, source: 'local' };
    }
    if (nd(lm[4], lm[10]) < 0.48) {
      return { type: 'LETTER', value: 'N', confidence: 0.86, source: 'local' };
    }
    if (nd(lm[4], lm[14]) < 0.55 || nd(lm[4], lm[18]) < 0.6) {
      return { type: 'LETTER', value: 'M', confidence: 0.86, source: 'local' };
    }
    return { type: 'LETTER', value: 'S', confidence: 0.88, source: 'local' };
  }

  return { type: 'UNKNOWN', value: '', confidence: 0.2, source: 'local' };
}

// Emits a steady 280ms window after a 450ms hold so your UI registers every letter once
function stabilizeAndGate(raw: VisionResult): VisionResult {
  const now = Date.now();
  const empty: VisionResult = { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

  if (raw.type === 'UNKNOWN' || !raw.value) {
    mismatchFrames++;
    if (mismatchFrames > 3) {
      candidateSign = '';
    }
    return empty;
  }

  if (raw.value !== candidateSign) {
    mismatchFrames++;
    if (mismatchFrames > 2 || !candidateSign) {
      candidateSign = raw.value;
      candidateStartTime = now;
      mismatchFrames = 0;
    }
    return empty;
  }

  mismatchFrames = 0;
  const elapsed = now - candidateStartTime;

  // 1. Wait 450ms while user forms the sign
  if (elapsed < HOLD_DELAY_MS) {
    return empty;
  }

  // 2. Emit sign for 280ms so the UI hook captures it cleanly once
  if (elapsed <= HOLD_DELAY_MS + EMIT_WINDOW_MS) {
    if (raw.value === 'CLEAR') triggerInputAction('CLEAR');
    if (raw.value === 'SEND') triggerInputAction('SEND');
    return raw;
  }

  // 3. Cooldown pause (1.8s) so holding the same sign doesn't type "AAAA..."
  if (elapsed < HOLD_DELAY_MS + EMIT_WINDOW_MS + REPEAT_COOLDOWN_MS) {
    return empty;
  }

  // Reset timer if user intentionally holds for >2.5 seconds to repeat a letter
  candidateStartTime = now - HOLD_DELAY_MS;
  return raw;
}

export async function localVision(video: HTMLVideoElement, timestamp: number): Promise<VisionResult> {
  try {
    if (!video || video.readyState < 2 || video.videoWidth === 0) {
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
      mismatchFrames++;
      if (mismatchFrames > 3) candidateSign = '';
      updateMouselessCursor(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // TOGGLE MOUSELESS MODE: Triggers on EITHER the 2nd Image (Two Stacked Horizontal Hands)
    // OR Two-Hand Middle Finger Gesture
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
      const cx = (1 - indexTip.x) * window.innerWidth;
      const cy = indexTip.y * window.innerHeight;
      const pinching = d(indexTip, thumbTip) < 0.055;

      updateMouselessCursor(true, cx, cy, pinching);

      if (indexTip.y < 0.2) window.scrollBy({ top: -15, behavior: 'auto' });
      if (indexTip.y > 0.8) window.scrollBy({ top: 15, behavior: 'auto' });

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

    const rawResult = classifyRaw(hands[0]);
    return stabilizeAndGate(rawResult);
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
