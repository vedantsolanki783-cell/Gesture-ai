import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

// Loaded from Google's/jsDelivr's public CDNs so no manual file download or
// build-time asset placement is needed. The browser caches both after first
// load, so the app keeps working offline on repeat visits (same device).
const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarkerPromise: Promise<HandLandmarker> | null = null;

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      });
    })();
  }
  return landmarkerPromise;
}

const d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);
const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

function extended(lm: Landmark[], tip: number, pip: number, mcp: number) {
  const wrist = lm[0];
  return d(lm[tip], wrist) > d(lm[pip], wrist) * 1.08 && d(lm[tip], lm[mcp]) > d(lm[pip], lm[mcp]);
}

function classify(lm: Landmark[]): VisionResult {
  if (lm.length < 21) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

  const index = extended(lm, 8, 6, 5);
  const middle = extended(lm, 12, 10, 9);
  const ring = extended(lm, 16, 14, 13);
  const pinky = extended(lm, 20, 18, 17);

  const thumbOut = d(lm[4], lm[5]) > d(lm[3], lm[5]) * 1.05;
  const thumbDown = lm[4].y > lm[0].y + 0.08 && !index && !middle && !ring && !pinky;
  const four = [index, middle, ring, pinky];
  const count = four.filter(Boolean).length;
  const spread = avg([d(lm[8], lm[12]), d(lm[12], lm[16]), d(lm[16], lm[20])]);

  if (thumbDown) return { type: 'GESTURE', value: 'CLEAR', confidence: 0.9, source: 'local' };
  if (count === 4 && !thumbOut) return { type: 'LETTER', value: 'B', confidence: 0.82, source: 'local' };
  if (count === 4 && thumbOut) {
    const wide = spread > 0.08;
    return { type: 'GESTURE', value: 'THEME_SWITCH', confidence: wide ? 0.93 : 0.82, source: 'local' };
  }
  if (index && middle && !ring && !pinky) return { type: 'LETTER', value: 'V', confidence: 0.9, source: 'local' };
  if (index && !middle && !ring && !pinky && thumbOut) return { type: 'LETTER', value: 'L', confidence: 0.88, source: 'local' };
  if (thumbOut && pinky && !index && !middle && !ring) return { type: 'LETTER', value: 'Y', confidence: 0.87, source: 'local' };
  if (!index && !middle && !ring && !pinky && !thumbOut) return { type: 'LETTER', value: 'A', confidence: 0.72, source: 'local' };
  if (!index && !middle && !ring && !pinky && thumbOut && d(lm[4], lm[8]) > 0.16) return { type: 'LETTER', value: 'C', confidence: 0.68, source: 'local' };
  return { type: 'UNKNOWN', value: '', confidence: 0.2, source: 'local' };
}

export async function localVision(video: HTMLVideoElement, timestamp: number): Promise<VisionResult> {
  const landmarker = await getLandmarker();
  const result = landmarker.detectForVideo(video, timestamp);
  const hand = result.landmarks?.[0] as Landmark[] | undefined;
  return hand ? classify(hand) : { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
}

export async function isLocalVisionModelAvailable() {
  try {
    const response = await fetch(MODEL_PATH, { method: 'HEAD' });
    return response.ok;
  } catch { return false; }
}

export function applyCustomGesture(result: VisionResult, gestures: CustomGesture[]): VisionResult {
  if (result.type !== 'GESTURE') return result;
  const match = gestures.find(g => g.name.toUpperCase() === result.value.toUpperCase());
  return match ? { ...result, value: match.name } : result;
                                  }
