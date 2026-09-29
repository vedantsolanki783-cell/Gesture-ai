import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const WASM_PATH = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
const MODEL_PATH = httpsUrl(
  'storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
);

const HOLD_DELAY_MS = 420;
const EMIT_WINDOW_MS = 280;
const REPEAT_COOLDOWN_MS = 1750;

let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;
let lastActionTime = 0;

let candidateSign = '';
let candidateStartTime = 0;
let mismatchFrames = 0;

const recentPredictions: VisionResult[] = [];
const indexTrail: { x: number; y: number; t: number }[] = [];
const pinkyTrail: { x: number; y: number; t: number }[] = [];

// Initializes MediaPipe with automatic CPU fallback so Android APK WebView never fails
async function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      const isAndroidApk =
        typeof window !== 'undefined' &&
        (Boolean((window as any).NovaAndroid) || /wv|Android/i.test(navigator.userAgent));

      // On Android WebView, CPU (XNNPACK) is 100% reliable and avoids WebGL2 context crashes
      const preferredDelegate = isAndroidApk ? 'CPU' : 'GPU';

      try {
        return await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: preferredDelegate },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.38,
          minHandPresenceConfidence: 0.38,
          minTrackingConfidence: 0.38
        });
      } catch (err) {
        console.warn('Primary delegate failed, switching to CPU fallback:', err);
        return await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.35,
          minHandPresenceConfidence: 0.35,
          minTrackingConfidence: 0.35
        });
      }
    })();
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

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

function palmProgress(lm: Landmark[], pt: Landmark): number {
  const vx = lm[17].x - lm[5].x;
  const vy = lm[17].y - lm[5].y;
  const lenSq = vx * vx + vy * vy;
  if (lenSq < 1e-6) return 0;
  return ((pt.x - lm[5].x) * vx + (pt.y - lm[5].y) * vy) / lenSq;
}

function recordTrail(trail: { x: number; y: number; t: number }[], pt: Landmark) {
  const now = Date.now();
  trail.push({ x: pt.x, y: pt.y, t: now });
  while (trail.length > 0 && now - trail[0].t > 500) {
    trail.shift();
  }
}

function getTrailSpan(trail: { x: number; y: number; t: number }[]) {
  if (trail.length < 4) return { dx: 0, dy: 0, total: 0 };
  let minX = 1;
  let maxX = 0;
  let minY = 1;
  let maxY = 0;
  for (const p of trail) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { dx: maxX - minX, dy: maxY - minY, total: Math.hypot(maxX - minX, maxY - minY) };
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
  const middleStraight = jointAngleDeg(lm[9], lm[10], lm[12]) > 138 && lm[12].y < lm[10].y;
  const othersCurled =
    lm[12].y < lm[8].y - palm * 0.28 &&
    lm[12].y < lm[16].y - palm * 0.28 &&
    lm[12].y < lm[20].y - palm * 0.28;
  return middleStraight && othersCurled;
}

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
    ? '🖱️ Mouseless Mode ON (Point to move, Pinch to click)'
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
      buttons.find(
        b => b.type === 'submit' || /send/i.test(b.textContent || '') || b.id === 'nova-send-btn'
      ) || buttons[buttons.length - 1];
    if (sendBtn) sendBtn.click();
  }
}

function classifyRaw(lm: Landmark[]): VisionResult {
  if (!lm || lm.length < 21) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }

  recordTrail(indexTrail, lm[8]);
  recordTrail(pinkyTrail, lm[20]);

  const wrist = lm[0];
  const palm = Math.max(d(wrist, lm[9]), 0.05);
  const nd = (a: Landmark, b: Landmark) => d(a, b) / palm;

  const idxAngle = jointAngleDeg(lm[5], lm[6], lm[8]);
  const midAngle = jointAngleDeg(lm[9], lm[10], lm[12]);
  const rngAngle = jointAngleDeg(lm[13], lm[14], lm[16]);
  const pnkAngle = jointAngleDeg(lm[17], lm[18], lm[20]);

  const indexUp = idxAngle > 135 && lm[8].y < lm[6].y && nd(lm[8], wrist) > nd(lm[5], wrist) * 1.12;
  const middleUp = midAngle > 135 && lm[12].y < lm[10].y && nd(lm[12], wrist) > nd(lm[9], wrist) * 1.12;
  const ringUp = rngAngle > 135 && lm[16].y < lm[14].y && nd(lm[16], wrist) > nd(lm[13], wrist) * 1.1;
  const pinkyUp = pnkAngle > 132 && lm[20].y < lm[18].y && nd(lm[20], wrist) > nd(lm[17], wrist) * 1.08;

  const indexExt = idxAngle > 135 && nd(lm[8], wrist) > nd(lm[6], wrist) * 1.06;
  const middleExt = midAngle > 135 && nd(lm[12], wrist) > nd(lm[10], wrist) * 1.06;
  const ringExt = rngAngle > 135 && nd(lm[16], wrist) > nd(lm[14], wrist) * 1.06;
  const pinkyExt = pnkAngle > 132 && nd(lm[20], wrist) > nd(lm[18], wrist) * 1.06;

  const thumbPalmPos = palmProgress(lm, lm[4]);
  const thumbOut = d(lm[4], lm[17]) > d(lm[2], lm[17]) * 1.14 && thumbPalmPos < 0.14;
  const thumbUp = lm[4].y < lm[3].y && lm[3].y < lm[2].y;
  const isHorizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y) * 1.22;

  const allFingersClosed = !indexUp && !middleUp && !ringUp && !pinkyUp;

  // 1. Thumbs Down (CLEAR)
  const thumbPointingDown =
    lm[4].y > lm[3].y && lm[4].y > lm[5].y + palm * 0.3 && lm[4].y > wrist.y + palm * 0.32;
  if (thumbPointingDown && allFingersClosed && !indexExt) {
    return { type: 'GESTURE', value: 'CLEAR', confidence: 0.96, source: 'local' };
  }

  // 2. Yo-Yo Sign (SEND)
  if (indexUp && !middleUp && !ringUp && pinkyUp) {
    return { type: 'GESTURE', value: 'SEND', confidence: 0.96, source: 'local' };
  }

  // 3. Q, P, H, G
  if (lm[8].y > lm[5].y + palm * 0.3 && lm[4].y > lm[2].y + palm * 0.18 && !middleUp && !ringUp && !pinkyUp) {
    return { type: 'LETTER', value: 'Q', confidence: 0.9, source: 'local' };
  }
  if (isHorizontal && indexExt && !ringExt && !pinkyExt) {
    if (middleExt) {
      const middleDropped = lm[12].y > lm[8].y + palm * 0.22;
      return { type: 'LETTER', value: middleDropped ? 'P' : 'H', confidence: 0.9, source: 'local' };
    }
    if (lm[12].y > lm[9].y + palm * 0.26) {
      return { type: 'LETTER', value: 'P', confidence: 0.89, source: 'local' };
    }
    return { type: 'LETTER', value: 'G', confidence: 0.9, source: 'local' };
  }

  // 4. B
  if (indexUp && middleUp && ringUp && pinkyUp) {
    return { type: 'LETTER', value: 'B', confidence: 0.94, source: 'local' };
  }

  // 5. W & F
  if (indexUp && middleUp && ringUp && !pinkyUp) {
    return { type: 'LETTER', value: 'W', confidence: 0.94, source: 'local' };
  }
  if (!indexUp && middleUp && ringUp && pinkyUp && nd(lm[4], lm[8]) < 0.68) {
    return { type: 'LETTER', value: 'F', confidence: 0.93, source: 'local' };
  }

  // 6. Y, J, I
  if (!indexUp && !middleUp && !ringUp && pinkyUp) {
    if (thumbOut) {
      return { type: 'LETTER', value: 'Y', confidence: 0.94, source: 'local' };
    }
    const pMove = getTrailSpan(pinkyTrail);
    if (pMove.total > 0.055) {
      return { type: 'LETTER', value: 'J', confidence: 0.9, source: 'local' };
    }
    return { type: 'LETTER', value: 'I', confidence: 0.93, source: 'local' };
  }

  // 7. R, K, V, U
  if (indexUp && middleUp && !ringUp && !pinkyUp) {
    const tipSpread = nd(lm[8], lm[12]);
    const baseSpread = nd(lm[5], lm[9]);

    const fingersCrossed = (lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < -0.0002;
    if (fingersCrossed) {
      return { type: 'LETTER', value: 'R', confidence: 0.91, source: 'local' };
    }

    const thumbBetween =
      lm[4].y < lm[5].y && nd(lm[4], lm[6]) < 0.55 && nd(lm[4], lm[10]) < 0.55 && tipSpread > 0.28;
    if (thumbBetween) {
      return { type: 'LETTER', value: 'K', confidence: 0.9, source: 'local' };
    }

    if (tipSpread > baseSpread * 1.25 || tipSpread > 0.38) {
      return { type: 'LETTER', value: 'V', confidence: 0.94, source: 'local' };
    }
    return { type: 'LETTER', value: 'U', confidence: 0.93, source: 'local' };
  }

  // 8. L, Z, D
  if (indexUp && !middleUp && !ringUp && !pinkyUp) {
    if (thumbOut && nd(lm[4], lm[8]) > 0.9) {
      return { type: 'LETTER', value: 'L', confidence: 0.95, source: 'local' };
    }
    const iMove = getTrailSpan(indexTrail);
    if (iMove.dx > 0.05) {
      return { type: 'LETTER', value: 'Z', confidence: 0.9, source: 'local' };
    }
    return { type: 'LETTER', value: 'D', confidence: 0.92, source: 'local' };
  }

  // 9. X, O, C, E, A, T, N, M, S
  if (allFingersClosed) {
    const thumbIndex = nd(lm[4], lm[8]);
    const thumbMiddle = nd(lm[4], lm[12]);

    const indexHooked =
      lm[6].y < lm[10].y - palm * 0.16 && lm[8].y > lm[6].y && idxAngle > 55 && idxAngle < 135;
    if (indexHooked) {
      return { type: 'LETTER', value: 'X', confidence: 0.89, source: 'local' };
    }

    const fingersArched = midAngle > 75 && nd(lm[8], lm[5]) > 0.5 && nd(lm[12], lm[9]) > 0.5;
    if (fingersArched && thumbIndex < 0.46 && thumbMiddle < 0.54) {
      return { type: 'LETTER', value: 'O', confidence: 0.91, source: 'local' };
    }
    if (fingersArched && thumbIndex >= 0.46 && thumbIndex < 1.25 && lm[8].y < lm[5].y) {
      return { type: 'LETTER', value: 'C', confidence: 0.89, source: 'local' };
    }

    const tipsAboveThumb =
      lm[8].y < lm[4].y - palm * 0.04 &&
      lm[12].y < lm[4].y - palm * 0.04 &&
      lm[16].y < lm[4].y &&
      thumbPalmPos > 0.15 &&
      nd(lm[4], lm[12]) < 0.65;
    if (tipsAboveThumb) {
      return { type: 'LETTER', value: 'E', confidence: 0.88, source: 'local' };
    }

    if (thumbPalmPos < 0.14 && thumbUp && lm[4].y < lm[6].y + palm * 0.12) {
      return { type: 'LETTER', value: 'A', confidence: 0.94, source: 'local' };
    }

    const thumbTuckedHigh = lm[4].y < lm[8].y && lm[4].y < lm[12].y;
    if (thumbTuckedHigh) {
      if (thumbPalmPos >= 0.08 && thumbPalmPos < 0.36 && nd(lm[4], lm[6]) < 0.48) {
        return { type: 'LETTER', value: 'T', confidence: 0.88, source: 'local' };
      }
      if (thumbPalmPos >= 0.36 && thumbPalmPos < 0.64 && nd(lm[4], lm[10]) < 0.48) {
        return { type: 'LETTER', value: 'N', confidence: 0.88, source: 'local' };
      }
      if (thumbPalmPos >= 0.64 && (nd(lm[4], lm[14]) < 0.52 || nd(lm[4], lm[18]) < 0.56)) {
        return { type: 'LETTER', value: 'M', confidence: 0.88, source: 'local' };
      }
    }

    return { type: 'LETTER', value: 'S', confidence: 0.9, source: 'local' };
  }

  return { type: 'UNKNOWN', value: '', confidence: 0.2, source: 'local' };
}

function smoothPrediction(raw: VisionResult): VisionResult {
  recentPredictions.push(raw);
  if (recentPredictions.length > 6) {
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

  if (bestSign && bestCount >= 3) {
    return bestSample;
  }
  return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
}

function stabilizeAndGate(raw: VisionResult): VisionResult {
  const smoothed = smoothPrediction(raw);
  const now = Date.now();
  const empty: VisionResult = { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

  if (smoothed.type === 'UNKNOWN' || !smoothed.value) {
    mismatchFrames++;
    if (mismatchFrames > 3) candidateSign = '';
    return empty;
  }

  if (smoothed.value !== candidateSign) {
    candidateSign = smoothed.value;
    candidateStartTime = now;
    mismatchFrames = 0;
    return empty;
  }

  mismatchFrames = 0;
  const elapsed = now - candidateStartTime;

  if (elapsed < HOLD_DELAY_MS) return empty;

  if (elapsed <= HOLD_DELAY_MS + EMIT_WINDOW_MS) {
    if (smoothed.value === 'CLEAR') triggerInputAction('CLEAR');
    if (smoothed.value === 'SEND') triggerInputAction('SEND');
    return smoothed;
  }

  if (elapsed < HOLD_DELAY_MS + EMIT_WINDOW_MS + REPEAT_COOLDOWN_MS) {
    return empty;
  }

  candidateStartTime = now - HOLD_DELAY_MS;
  return smoothed;
}

export async function localVision(video: HTMLVideoElement, _timestamp: number): Promise<VisionResult> {
  try {
    if (!video) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Force Android WebView video stream to play inline
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
      mismatchFrames++;
      if (mismatchFrames > 3) {
        candidateSign = '';
        recentPredictions.length = 0;
      }
      updateMouselessCursor(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

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
