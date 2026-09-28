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

// Anti-repeat lock so 1 sign = 1 single letter typed
let lockedSign = '';
let signFirstSeenAt = 0;
let lockCooldownUntil = 0;
let framesWithoutHand = 0;

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        // Lowered to 0.35 so the camera detects hands reliably in low/indoor light
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.35
      });
    })().catch((err) => {
      landmarkerPromise = null;
      throw err;
    });
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

function getFingerStates(lm: Landmark[]) {
  const wrist = lm[0];
  const thumbOpen = d(lm[4], lm[17]) > d(lm[3], lm[17]) * 1.08;
  const thumbUp = lm[4].y < lm[3].y && lm[3].y < lm[2].y;

  // Forgiving extension checks that work even if your hand is slightly tilted
  const indexOpen = lm[8].y < lm[6].y && d(lm[8], wrist) > d(lm[6], wrist);
  const middleOpen = lm[12].y < lm[10].y && d(lm[12], wrist) > d(lm[10], wrist);
  const ringOpen = lm[16].y < lm[14].y && d(lm[16], wrist) > d(lm[14], wrist);
  const pinkyOpen = lm[20].y < lm[18].y && d(lm[20], wrist) > d(lm[18], wrist);
  const indexHalf = lm[8].y >= lm[6].y && lm[6].y < lm[5].y;

  return { thumbOpen, thumbUp, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf };
}

function isOnlyMiddleFinger(lm: Landmark[]): boolean {
  if (!lm || lm.length < 21) return false;
  const { indexOpen, middleOpen, ringOpen, pinkyOpen } = getFingerStates(lm);
  return middleOpen && !indexOpen && !ringOpen && !pinkyOpen;
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

// Raw 26-Letter ASL Classifier (A-Z) + Thumbs Down (CLEAR) + Yo-Yo (SEND)
function classifyRaw(lm: Landmark[]): VisionResult {
  if (!lm || lm.length < 21) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  const { thumbOpen, thumbUp, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf } =
    getFingerStates(lm);

  const thumbIndex = d(lm[4], lm[8]);
  const thumbMiddle = d(lm[4], lm[12]);
  const indexMiddle = d(lm[8], lm[12]);
  const horizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y) * 1.12;

  // 1. THUMBS DOWN (👎) -> CLEAR ALL TEXT
  const allFingersClosed = !indexOpen && !middleOpen && !ringOpen && !pinkyOpen;
  const thumbPointingDown = lm[4].y > lm[3].y && lm[4].y > lm[5].y + 0.03 && lm[4].y > lm[0].y + 0.04;
  if (thumbPointingDown && allFingersClosed) {
    triggerInputAction('CLEAR');
    return { type: 'GESTURE', value: 'CLEAR', confidence: 0.95, source: 'local' };
  }

  // 2. YO-YO SIGN (🤘: Index + Pinky up) -> SEND MESSAGE
  if (indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    triggerInputAction('SEND');
    return { type: 'GESTURE', value: 'SEND', confidence: 0.95, source: 'local' };
  }

  // 3. All 4 fingers curled (A, E, M, N, O, S, T, C, X)
  if (allFingersClosed) {
    if (indexHalf) {
      if (thumbOpen && thumbIndex > 0.08) {
        return { type: 'LETTER', value: 'C', confidence: 0.86, source: 'local' };
      }
      return { type: 'LETTER', value: 'X', confidence: 0.86, source: 'local' };
    }
    if (thumbIndex < 0.07 && thumbMiddle < 0.09) {
      return { type: 'LETTER', value: 'O', confidence: 0.88, source: 'local' };
    }
    if (thumbOpen && thumbIndex > 0.13) {
      return { type: 'LETTER', value: 'C', confidence: 0.85, source: 'local' };
    }
    if (thumbOpen && thumbUp) {
      return { type: 'LETTER', value: 'A', confidence: 0.92, source: 'local' };
    }
    if (lm[4].y > lm[8].y && lm[4].y > lm[12].y) {
      return { type: 'LETTER', value: 'E', confidence: 0.86, source: 'local' };
    }
    if (d(lm[4], lm[6]) < 0.065) {
      return { type: 'LETTER', value: 'T', confidence: 0.85, source: 'local' };
    }
    if (d(lm[4], lm[10]) < 0.065) {
      return { type: 'LETTER', value: 'N', confidence: 0.85, source: 'local' };
    }
    if (d(lm[4], lm[14]) < 0.075) {
      return { type: 'LETTER', value: 'M', confidence: 0.85, source: 'local' };
    }
    return { type: 'LETTER', value: 'S', confidence: 0.86, source: 'local' };
  }

  // 4. All 4 fingers open -> B
  if (indexOpen && middleOpen && ringOpen && pinkyOpen) {
    return { type: 'LETTER', value: 'B', confidence: 0.92, source: 'local' };
  }

  // 5. Three fingers open -> W or F
  if (indexOpen && middleOpen && ringOpen && !pinkyOpen) {
    return { type: 'LETTER', value: 'W', confidence: 0.92, source: 'local' };
  }
  if (!indexOpen && middleOpen && ringOpen && pinkyOpen) {
    return { type: 'LETTER', value: 'F', confidence: 0.9, source: 'local' };
  }

  // 6. Only Pinky open -> Y, J, I
  if (!indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    if (thumbOpen) {
      return { type: 'LETTER', value: 'Y', confidence: 0.92, source: 'local' };
    }
    const pinkyTilted = Math.abs(lm[20].x - lm[17].x) > Math.abs(lm[20].y - lm[17].y) * 0.65;
    return { type: 'LETTER', value: pinkyTilted ? 'J' : 'I', confidence: 0.88, source: 'local' };
  }

  // 7. Index + Middle open -> H, P, R, K, V, U
  if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) {
      const pointingDown = lm[12].y > lm[8].y + 0.04 || lm[8].y > lm[0].y;
      return { type: 'LETTER', value: pointingDown ? 'P' : 'H', confidence: 0.88, source: 'local' };
    }
    if ((lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < 0 || indexMiddle < 0.032) {
      return { type: 'LETTER', value: 'R', confidence: 0.87, source: 'local' };
    }
    if (thumbOpen && thumbMiddle < 0.09) {
      return { type: 'LETTER', value: 'K', confidence: 0.87, source: 'local' };
    }
    if (indexMiddle > 0.052) {
      return { type: 'LETTER', value: 'V', confidence: 0.92, source: 'local' };
    }
    return { type: 'LETTER', value: 'U', confidence: 0.88, source: 'local' };
  }

  // 8. Only Index open -> G, Q, L, D, Z
  if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) {
      const pointingDown = lm[8].y > lm[0].y;
      return { type: 'LETTER', value: pointingDown ? 'Q' : 'G', confidence: 0.87, source: 'local' };
    }
    if (thumbOpen) {
      return { type: 'LETTER', value: 'L', confidence: 0.92, source: 'local' };
    }
    if (thumbMiddle < 0.08) {
      return { type: 'LETTER', value: 'D', confidence: 0.88, source: 'local' };
    }
    return { type: 'LETTER', value: 'Z', confidence: 0.85, source: 'local' };
  }

  return { type: 'UNKNOWN', value: '', confidence: 0.2, source: 'local' };
}

// Stabilizer + Single-Shot Gate: Prevents "AAAAA..." spam so each sign types only ONCE
function stabilizeAndGate(raw: VisionResult): VisionResult {
  const now = Date.now();

  if (raw.type === 'UNKNOWN' || !raw.value) {
    return raw;
  }

  // New sign detected
  if (raw.value !== lockedSign) {
    lockedSign = raw.value;
    signFirstSeenAt = now;
    // Let the new sign register for a brief 350ms window, then pause for 2.2 seconds
    lockCooldownUntil = now + 350;
    return raw;
  }

  // If still within the initial 350ms registration window, return the sign
  if (now <= lockCooldownUntil) {
    return raw;
  }

  // After 350ms, block duplicate repeats for 2.2 seconds so holding a sign doesn't spam "AAAA"
  if (now - signFirstSeenAt < 2500) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  // If user intentionally keeps holding for >2.5 seconds, allow 1 more letter (for double letters like "LL")
  signFirstSeenAt = now;
  lockCooldownUntil = now + 300;
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
      framesWithoutHand++;
      // Reset the repeat lock as soon as you lower your hand out of the camera
      if (framesWithoutHand > 3) {
        lockedSign = '';
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    framesWithoutHand = 0;

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
