import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;
let lastActionTime = 0;

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
    })().catch((err) => {
      landmarkerPromise = null;
      throw err;
    });
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

// Works even when the hand is tilted sideways (such as during Thumbs Down)
function isFingerExtended(lm: Landmark[], tip: number, pip: number, mcp: number): boolean {
  const wrist = lm[0];
  return d(lm[tip], wrist) > d(lm[pip], wrist) * 1.08 && d(lm[tip], lm[mcp]) > d(lm[pip], lm[mcp]) * 1.05;
}

function getFingerStates(lm: Landmark[]) {
  const thumbOpen = d(lm[4], lm[17]) > d(lm[3], lm[17]) * 1.12;
  const thumbUp = lm[4].y < lm[3].y && lm[3].y < lm[2].y;
  const indexOpen = isFingerExtended(lm, 8, 6, 5) && lm[8].y < lm[6].y;
  const middleOpen = isFingerExtended(lm, 12, 10, 9) && lm[12].y < lm[10].y;
  const ringOpen = isFingerExtended(lm, 16, 14, 13) && lm[16].y < lm[14].y;
  const pinkyOpen = isFingerExtended(lm, 20, 18, 17) && lm[20].y < lm[18].y;
  const indexHalf = lm[8].y >= lm[6].y && lm[6].y < lm[5].y;
  return { thumbOpen, thumbUp, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf };
}

function isOnlyMiddleFinger(lm: Landmark[]): boolean {
  if (!lm || lm.length < 21) return false;
  const { indexOpen, middleOpen, ringOpen, pinkyOpen } = getFingerStates(lm);
  return middleOpen && !indexOpen && !ringOpen && !pinkyOpen;
}

// Directly clears the React input box or clicks the Send button when CLEAR or SEND is triggered
function triggerInputAction(action: 'CLEAR' | 'SEND') {
  const now = Date.now();
  if (now - lastActionTime < 900) return;
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

// Complete 26-Letter ASL Classifier (A-Z) + Thumbs Down (CLEAR) + Yo-Yo Sign (SEND)
function classify(lm: Landmark[]): VisionResult {
  if (!lm || lm.length < 21) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  const { thumbOpen, thumbUp, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf } =
    getFingerStates(lm);

  const thumbIndex = d(lm[4], lm[8]);
  const thumbMiddle = d(lm[4], lm[12]);
  const indexMiddle = d(lm[8], lm[12]);
  const horizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y) * 1.15;

  // 1. THUMBS DOWN GESTURE (👎) -> CLEAR ALL TEXT
  const fingersCurledForThumbDown =
    !isFingerExtended(lm, 8, 6, 5) &&
    !isFingerExtended(lm, 12, 10, 9) &&
    !isFingerExtended(lm, 16, 14, 13) &&
    !isFingerExtended(lm, 20, 18, 17);
  const thumbPointingDown =
    lm[4].y > lm[3].y &&
    lm[3].y > lm[2].y &&
    lm[4].y > lm[5].y + 0.04 &&
    lm[4].y > lm[9].y + 0.04;

  if (thumbPointingDown && fingersCurledForThumbDown) {
    triggerInputAction('CLEAR');
    return { type: 'GESTURE', value: 'CLEAR', confidence: 0.94, source: 'local' };
  }

  // 2. YO-YO / ROCK-ON SIGN (🤘: Index + Pinky extended, Middle + Ring curled) -> SEND MESSAGE
  if (indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    triggerInputAction('SEND');
    return { type: 'GESTURE', value: 'SEND', confidence: 0.94, source: 'local' };
  }

  // 3. All 4 fingers curled (A, E, M, N, O, S, T, C, X)
  if (!indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    if (indexHalf) {
      if (thumbOpen && thumbIndex > 0.08) {
        return { type: 'LETTER', value: 'C', confidence: 0.85, source: 'local' };
      }
      return { type: 'LETTER', value: 'X', confidence: 0.85, source: 'local' };
    }
    if (thumbIndex < 0.065 && thumbMiddle < 0.085) {
      return { type: 'LETTER', value: 'O', confidence: 0.88, source: 'local' };
    }
    if (thumbOpen && thumbIndex > 0.13) {
      return { type: 'LETTER', value: 'C', confidence: 0.84, source: 'local' };
    }
    if (thumbOpen && thumbUp) {
      return { type: 'LETTER', value: 'A', confidence: 0.9, source: 'local' };
    }
    if (lm[4].y > lm[8].y && lm[4].y > lm[12].y) {
      return { type: 'LETTER', value: 'E', confidence: 0.85, source: 'local' };
    }
    if (d(lm[4], lm[6]) < 0.06) {
      return { type: 'LETTER', value: 'T', confidence: 0.84, source: 'local' };
    }
    if (d(lm[4], lm[10]) < 0.06) {
      return { type: 'LETTER', value: 'N', confidence: 0.84, source: 'local' };
    }
    if (d(lm[4], lm[14]) < 0.07) {
      return { type: 'LETTER', value: 'M', confidence: 0.84, source: 'local' };
    }
    return { type: 'LETTER', value: 'S', confidence: 0.85, source: 'local' };
  }

  // 4. All 4 fingers open -> B
  if (indexOpen && middleOpen && ringOpen && pinkyOpen) {
    return { type: 'LETTER', value: 'B', confidence: 0.9, source: 'local' };
  }

  // 5. Three fingers open -> W or F
  if (indexOpen && middleOpen && ringOpen && !pinkyOpen) {
    return { type: 'LETTER', value: 'W', confidence: 0.9, source: 'local' };
  }
  if (!indexOpen && middleOpen && ringOpen && pinkyOpen) {
    return { type: 'LETTER', value: 'F', confidence: 0.88, source: 'local' };
  }

  // 6. Only Pinky open -> Y, J, I
  if (!indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    if (thumbOpen) {
      return { type: 'LETTER', value: 'Y', confidence: 0.9, source: 'local' };
    }
    const pinkyTilted = Math.abs(lm[20].x - lm[17].x) > Math.abs(lm[20].y - lm[17].y) * 0.65;
    return { type: 'LETTER', value: pinkyTilted ? 'J' : 'I', confidence: 0.87, source: 'local' };
  }

  // 7. Index + Middle open -> H, P, R, K, V, U
  if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) {
      const pointingDown = lm[12].y > lm[8].y + 0.04 || lm[8].y > lm[0].y;
      return { type: 'LETTER', value: pointingDown ? 'P' : 'H', confidence: 0.86, source: 'local' };
    }
    if ((lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < 0 || indexMiddle < 0.03) {
      return { type: 'LETTER', value: 'R', confidence: 0.86, source: 'local' };
    }
    if (thumbOpen && thumbMiddle < 0.085) {
      return { type: 'LETTER', value: 'K', confidence: 0.86, source: 'local' };
    }
    if (indexMiddle > 0.055) {
      return { type: 'LETTER', value: 'V', confidence: 0.9, source: 'local' };
    }
    return { type: 'LETTER', value: 'U', confidence: 0.87, source: 'local' };
  }

  // 8. Only Index open -> G, Q, L, D, Z
  if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) {
      const pointingDown = lm[8].y > lm[0].y;
      return { type: 'LETTER', value: pointingDown ? 'Q' : 'G', confidence: 0.86, source: 'local' };
    }
    if (thumbOpen) {
      return { type: 'LETTER', value: 'L', confidence: 0.9, source: 'local' };
    }
    if (thumbMiddle < 0.075) {
      return { type: 'LETTER', value: 'D', confidence: 0.87, source: 'local' };
    }
    return { type: 'LETTER', value: 'Z', confidence: 0.84, source: 'local' };
  }

  return { type: 'UNKNOWN', value: '', confidence: 0.2, source: 'local' };
}

export async function localVision(video: HTMLVideoElement, timestamp: number): Promise<VisionResult> {
  try {
    if (!video || video.readyState < 2 || video.videoWidth === 0) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    const landmarker = await getLandmarker();
    const safeTimestamp = Math.max(Math.floor(timestamp || performance.now()), lastVideoTime + 1);
    lastVideoTime = safeTimestamp;

    const result = landmarker.detectForVideo(video, safeTimestamp);
    const hands = (result.landmarks || []) as Landmark[][];
    const now = Date.now();

    if (hands.length === 0) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Two-Hand Middle Finger Gesture toggles Mouseless Scroll/Click Mode
    if (hands.length >= 2 && isOnlyMiddleFinger(hands[0]) && isOnlyMiddleFinger(hands[1])) {
      if (now - lastToggleTime > 1500) {
        mouselessMode = !mouselessMode;
        lastToggleTime = now;
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    if (mouselessMode) {
      const indexTip = hands[0][8];
      const thumbTip = hands[0][4];
      const cx = (1 - indexTip.x) * window.innerWidth;
      const cy = indexTip.y * window.innerHeight;

      if (indexTip.y < 0.2) window.scrollBy({ top: -15, behavior: 'auto' });
      if (indexTip.y > 0.8) window.scrollBy({ top: 15, behavior: 'auto' });

      if (d(indexTip, thumbTip) < 0.05 && now - lastClickTime > 750) {
        lastClickTime = now;
        const el = document.elementFromPoint(cx, cy) as HTMLElement | null;
        if (el) el.click();
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    return classify(hands[0]);
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
