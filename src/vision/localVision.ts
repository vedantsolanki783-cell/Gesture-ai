import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// 8% faster response time (HOLD_TIME_MS set to 920ms)
const HOLD_TIME_MS = 920;
const REPEAT_COOLDOWN_MS = 1800;

let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;
let lastActionTime = 0;

let candidateSign = '';
let candidateStartTime = 0;
let hasEmittedCurrentHold = false;
let lastEmitTime = 0;
let mismatchFrames = 0;

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

function getFingerStates(lm: Landmark[]) {
  const wrist = lm[0];
  const indexOpen = lm[8].y < lm[6].y && d(lm[8], wrist) > d(lm[6], wrist);
  const middleOpen = lm[12].y < lm[10].y && d(lm[12], wrist) > d(lm[10], wrist);
  const ringOpen = lm[16].y < lm[14].y && d(lm[16], wrist) > d(lm[14], wrist);
  const pinkyOpen = lm[20].y < lm[18].y && d(lm[20], wrist) > d(lm[18], wrist);
  const indexHalf = lm[8].y >= lm[6].y && lm[6].y < lm[5].y;
  return { indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf };
}

// Updated specifically for palms facing inward / matching your uploaded image posture
function isOnlyMiddleFinger(lm: Landmark[]): boolean {
  if (!lm || lm.length < 21) return false;
  const { indexOpen, middleOpen, ringOpen, pinkyOpen } = getFingerStates(lm);
  // Middle finger tip extended upward, other 4 fingers curled closed
  const middleExtended = lm[12].y < lm[10].y && lm[12].y < lm[0].y;
  return middleExtended && !indexOpen && !ringOpen && !pinkyOpen;
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

function classifyRaw(lm: Landmark[]): VisionResult {
  if (!lm || lm.length < 21) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  const { thumbOpen, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf } = (() => {
    const wrist = lm[0];
    const thumbOpen = d(lm[4], lm[17]) > d(lm[3], lm[17]) * 1.08;
    const thumbUp = lm[4].y < lm[3].y && lm[3].y < lm[2].y;
    const indexOpen = lm[8].y < lm[6].y && d(lm[8], wrist) > d(lm[6], wrist);
    const middleOpen = lm[12].y < lm[10].y && d(lm[12], wrist) > d(lm[10], wrist);
    const ringOpen = lm[16].y < lm[14].y && d(lm[16], wrist) > d(lm[14], wrist);
    const pinkyOpen = lm[20].y < lm[18].y && d(lm[20], wrist) > d(lm[18], wrist);
    const indexHalf = lm[8].y >= lm[6].y && lm[6].y < lm[5].y;
    return { thumbOpen, thumbUp, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf };
  })();

  const thumbIndex = d(lm[4], lm[8]);
  const thumbMiddle = d(lm[4], lm[12]);
  const indexMiddle = d(lm[8], lm[12]);
  const horizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y) * 1.12;

  const allFingersClosed = !indexOpen && !middleOpen && !ringOpen && !pinkyOpen;
  const thumbPointingDown = lm[4].y > lm[3].y && lm[4].y > lm[5].y + 0.03 && lm[4].y > lm[0].y + 0.04;

  if (thumbPointingDown && allFingersClosed) {
    return { type: 'GESTURE', value: 'CLEAR', confidence: 0.95, source: 'local' };
  }

  if (indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    return { type: 'GESTURE', value: 'SEND', confidence: 0.95, source: 'local' };
  }

  if (allFingersClosed) {
    if (indexHalf) {
      if (thumbOpen && thumbIndex > 0.08) return { type: 'LETTER', value: 'C', confidence: 0.86, source: 'local' };
      return { type: 'LETTER', value: 'X', confidence: 0.86, source: 'local' };
    }
    if (thumbIndex < 0.07 && thumbMiddle < 0.09) return { type: 'LETTER', value: 'O', confidence: 0.88, source: 'local' };
    if (thumbOpen && thumbIndex > 0.13) return { type: 'LETTER', value: 'C', confidence: 0.85, source: 'local' };
    if (thumbOpen) return { type: 'LETTER', value: 'A', confidence: 0.92, source: 'local' };
    if (lm[4].y > lm[8].y && lm[4].y > lm[12].y) return { type: 'LETTER', value: 'E', confidence: 0.86, source: 'local' };
    if (d(lm[4], lm[6]) < 0.065) return { type: 'LETTER', value: 'T', confidence: 0.85, source: 'local' };
    if (d(lm[4], lm[10]) < 0.065) return { type: 'LETTER', value: 'N', confidence: 0.85, source: 'local' };
    if (d(lm[4], lm[14]) < 0.075) return { type: 'LETTER', value: 'M', confidence: 0.85, source: 'local' };
    return { type: 'LETTER', value: 'S', confidence: 0.86, source: 'local' };
  }

  if (indexOpen && middleOpen && ringOpen && pinkyOpen) return { type: 'LETTER', value: 'B', confidence: 0.92, source: 'local' };
  if (indexOpen && middleOpen && ringOpen && !pinkyOpen) return { type: 'LETTER', value: 'W', confidence: 0.92, source: 'local' };
  if (!indexOpen && middleOpen && ringOpen && pinkyOpen) return { type: 'LETTER', value: 'F', confidence: 0.9, source: 'local' };

  if (!indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    return { type: 'LETTER', value: 'I', confidence: 0.88, source: 'local' };
  }

  if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) {
      const pointingDown = lm[12].y > lm[8].y + 0.04 || lm[8].y > lm[0].y;
      return { type: 'LETTER', value: pointingDown ? 'P' : 'H', confidence: 0.88, source: 'local' };
    }
    if ((lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < 0 || indexMiddle < 0.032) {
      return { type: 'LETTER', value: 'R', confidence: 0.87, source: 'local' };
    }
    if (indexMiddle > 0.052) return { type: 'LETTER', value: 'V', confidence: 0.92, source: 'local' };
    return { type: 'LETTER', value: 'U', confidence: 0.88, source: 'local' };
  }

  if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) {
      const pointingDown = lm[8].y > lm[0].y;
      return { type: 'LETTER', value: pointingDown ? 'Q' : 'G', confidence: 0.87, source: 'local' };
    }
    return { type: 'LETTER', value: 'L', confidence: 0.92, source: 'local' };
  }

  return { type: 'UNKNOWN', value: '', confidence: 0.2, source: 'local' };
}

function stabilizeAndGate(raw: VisionResult): VisionResult {
  const now = Date.now();
  const empty: VisionResult = { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

  if (raw.type === 'UNKNOWN' || !raw.value) {
    mismatchFrames++;
    if (mismatchFrames > 3) {
      candidateSign = '';
      hasEmittedCurrentHold = false;
    }
    return empty;
  }

  if (raw.value !== candidateSign) {
    mismatchFrames++;
    if (mismatchFrames > 2 || !candidateSign) {
      candidateSign = raw.value;
      candidateStartTime = now;
      hasEmittedCurrentHold = false;
      mismatchFrames = 0;
    }
    return empty;
  }

  mismatchFrames = 0;

  if (!hasEmittedCurrentHold) {
    if (now - candidateStartTime < HOLD_TIME_MS) {
      return empty;
    }
    hasEmittedCurrentHold = true;
    lastEmitTime = now;
    if (raw.value === 'CLEAR') triggerInputAction('CLEAR');
    if (raw.value === 'SEND') triggerInputAction('SEND');
    return raw;
  }

  if (now - lastEmitTime >= REPEAT_COOLDOWN_MS) {
    lastEmitTime = now;
    if (raw.value === 'CLEAR') triggerInputAction('CLEAR');
    if (raw.value === 'SEND') triggerInputAction('SEND');
    return raw;
  }

  return empty;
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
      if (mismatchFrames > 3) {
        candidateSign = '';
        hasEmittedCurrentHold = false;
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // CHECK TWO-HAND MIDDLE FINGER GESTURE (Matching your uploaded image)
    if (hands.length >= 2 && isOnlyMiddleFinger(hands[0]) && isOnlyMiddleFinger(hands[1])) {
      if (now - lastToggleTime > 1200) {
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
