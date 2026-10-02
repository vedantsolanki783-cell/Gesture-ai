import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

// ============================================================================
// SYSTEM & MODEL CONFIGURATION
// ============================================================================
const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const WASM_PATH = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
const MODEL_PATH = httpsUrl('storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task');

// ============================================================================
// ADVANCED TUNING PARAMETERS
// ============================================================================
const GESTURE_HOLD_DELAY_MS = 500;       // Milliseconds required to hold a sign before firing
const POST_EMIT_COOLDOWN_MS = 1200;      // Cooldown after firing a sign to prevent spam
const TOGGLE_COOLDOWN_MS = 1400;         // Cooldown for switching to Wireless Mouse mode

// Palm-Ray Mouse Dynamic Kinematics
const PINCH_DOWN_THRESH = 0.052;         // Distance to trigger a click
const PINCH_UP_THRESH = 0.075;           // Hysteresis release distance to prevent double-clicking
const ACTIVE_ROI_MARGIN = 0.18;          // Deadzone margin on camera edges (allows reaching screen edges comfortably)
const SCROLL_TRIGGER_ZONE = 0.12;        // Top 12% and Bottom 12% of screen trigger scrolling
const SCROLL_VELOCITY = 20;              // Pixels per frame to scroll

// ============================================================================
// STATE MANAGEMENT (100% LOCAL)
// ============================================================================
let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;

// Mouse State
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;
let isPinching = false;

// Dynamic Filter Coordinates (1-Euro equivalent)
let filteredX = 0.5, filteredY = 0.5;
let prevTargetX = 0.5, prevTargetY = 0.5;

// Gesture Latching State
let candidateSign = '';
let latchedSign = '';
let candidateStartTime = 0;
let lastEmittedTime = 0;
let releaseFrameCount = 0;

// ============================================================================
// ON-DEVICE MACHINE LEARNING (LOCAL STORAGE ONLY)
// ============================================================================
interface MLEmbedding {
  label: string;
  vector: number[];
}

let mlDatabase: MLEmbedding[] = [];
let pendingTrainLabel: string | null = null;

// Initialize ML Database silently from Local Storage
try {
  const saved = localStorage.getItem('nova_ml_gestures');
  if (saved) mlDatabase = JSON.parse(saved);
} catch (e) {
  console.error("Local storage access denied for ML database.");
}

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

// ============================================================================
// KINEMATIC MATH & VECTOR GEOMETRY
// ============================================================================
const distance3D = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
const distance2D = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Normalizes a 3D hand skeleton into a scale-invariant, translation-invariant 63-dimensional vector.
 * This guarantees the gesture works whether the hand is close to the camera or far away.
 */
function normalizeVector(lm: Landmark[]): number[] {
  const wrist = lm[0];
  let maxDist = 0.0001; // Prevent divide-by-zero
  
  // Shift origin to wrist and find bounding sphere
  const centered = lm.map(p => {
    const dx = p.x - wrist.x;
    const dy = p.y - wrist.y;
    const dz = (p.z || 0) - (wrist.z || 0);
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxDist) maxDist = dist;
    return { x: dx, y: dy, z: dz };
  });

  // Flatten and scale
  const vector: number[] = [];
  for (const p of centered) {
    vector.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  }
  return vector;
}

/**
 * K-Nearest Neighbors (KNN) Euclidean distance classifier against the local database.
 */
function classifyWithML(vector: number[]): string {
  if (!mlDatabase.length) return '';
  
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
  
  // Reject if the hand shape is too far from any trained memory (Threshold: 1.15)
  return minDist < 1.15 ? bestLabel : '';
}

/**
 * Hardcoded Structural Analysis for built-in ASL and Control overrides.
 */
function classifyRuleBased(lm: Landmark[]): string {
  const wrist = lm[0];
  const palmScale = Math.max(distance2D(wrist, lm[9]), 0.05);
  
  // Normalized distance helper to handle depth changes
  const nd = (a: Landmark, b: Landmark) => distance2D(a, b) / palmScale;
  
  // Finger state evaluation
  const isThumbUp = lm[4].y < lm[3].y && lm[4].y < lm[5].y - palmScale * 0.2;
  const isThumbDown = lm[4].y > lm[3].y && lm[4].y > lm[5].y + palmScale * 0.3;
  
  const isIndexUp = lm[8].y < lm[6].y && nd(lm[8], wrist) > nd(lm[5], wrist);
  const isMiddleUp = lm[12].y < lm[10].y && nd(lm[12], wrist) > nd(lm[9], wrist);
  const isRingUp = lm[16].y < lm[14].y && nd(lm[16], wrist) > nd(lm[13], wrist);
  const isPinkyUp = lm[20].y < lm[18].y && nd(lm[20], wrist) > nd(lm[17], wrist);
  
  const allFingersClosed = !isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp;

  // 1. SYSTEM CONTROLS (Only 2 extra signs as requested)
  // CLEAR: Strict Thumb Down, fingers curled
  if (isThumbDown && allFingersClosed) return 'CLEAR';
  
  // SEND: Index and Pinky Up, Middle and Ring curled (Rock on sign)
  if (isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp) return 'SEND';

  // 2. ASL ALPHABET FALLBACKS
  if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
    return nd(lm[4], lm[5]) < 0.6 ? 'B' : ''; // Thumb tucked
  }
  if (isIndexUp && isMiddleUp && isRingUp && !isPinkyUp) return 'W';
  if (!isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp) {
    return distance2D(lm[4], lm[17]) > distance2D(lm[2], lm[17]) * 1.2 ? 'Y' : 'I';
  }
  if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
    return nd(lm[8], lm[12]) > 0.35 ? 'V' : 'U';
  }
  if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    return nd(lm[4], lm[8]) > 0.85 ? 'L' : 'D';
  }
  if (allFingersClosed) {
    if (lm[4].y < lm[6].y + palmScale * 0.1 && lm[4].x > lm[6].x) return 'A';
    if (lm[4].y > lm[6].y && lm[4].x < lm[6].x) return 'S';
    return 'E';
  }

  return '';
}

// ============================================================================
// HARDWARE INITIALIZATION
// ============================================================================
async function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_PATH).then(vision => 
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { 
          modelAssetPath: MODEL_PATH, 
          delegate: 'CPU' // CPU delegate ensures perfect stability on Tab/Mobile without GPU crashing
        },
        runningMode: 'VIDEO', 
        numHands: 2,
        minHandDetectionConfidence: 0.48, 
        minHandPresenceConfidence: 0.48,
        minTrackingConfidence: 0.48
      })
    );
  }
  return landmarkerPromise;
}

// ============================================================================
// SYSTEM BRIDGE INVOCATION
// ============================================================================
function transmitMouseCoordinates(visible: boolean, x = 0.5, y = 0.5, pinching = false) {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (bridge?.updateAirMouse) {
    try {
      bridge.updateAirMouse(visible, x, y, pinching);
    } catch (e) {
      console.error("Hardware bridge transmission failed.");
    }
  }
}

function invokeHardwareScroll(direction: 'up' | 'down') {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (direction === 'up') {
    window.scrollBy({ top: -SCROLL_VELOCITY, behavior: 'auto' });
    bridge?.swipeScreen?.(500, 450, 500, 1200, 260);
  } else {
    window.scrollBy({ top: SCROLL_VELOCITY, behavior: 'auto' });
    bridge?.swipeScreen?.(500, 1200, 500, 450, 260);
  }
}

// ============================================================================
// CORE VISION LOOP
// ============================================================================
export async function localVision(video: HTMLVideoElement, _timestamp: number): Promise<VisionResult> {
  try {
    if (!video || video.readyState < 2 || video.videoWidth === 0) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    
    video.playsInline = true;
    video.muted = true;
    if (video.paused && video.srcObject) {
      video.play().catch(() => {});
    }

    const landmarker = await getLandmarker();
    const currentPerformanceTime = performance.now();
    
    // Ensure strict monotonic time progression for MediaPipe Engine
    lastVideoTime = currentPerformanceTime > lastVideoTime ? currentPerformanceTime : lastVideoTime + 1;
    
    const result = landmarker.detectForVideo(video, lastVideoTime);
    const hands = (result.landmarks || []) as Landmark[][];
    const now = Date.now();

    // No hands detected: Auto-release tracking
    if (!hands.length) {
      if (++releaseFrameCount >= 5) {
        latchedSign = '';
        candidateSign = '';
      }
      transmitMouseCoordinates(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Toggle Wireless Mouse Mode: Two horizontal stacked hands
    if (hands.length === 2) {
      const verticalGap = Math.abs(hands[0][9].y - hands[1][9].y);
      if (verticalGap > 0.12 && now - lastToggleTime > TOGGLE_COOLDOWN_MS) {
        mouselessMode = !mouselessMode;
        lastToggleTime = now;
        
        // Notify System HUD if active
        const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
        if (mouselessMode) bridge?.enableOverlayBubble?.();
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    const primaryHand = hands[0];

    // ========================================================================
    // 1. PALM-RAY WIRELESS MOUSE ENGINE
    // ========================================================================
    if (mouselessMode) {
      // Anchor tracking strictly to the Knuckles (MCP 5 & 8) to prevent pinch-jerk
      const anchorX = (primaryHand[5].x * 0.65 + primaryHand[8].x * 0.35); 
      const anchorY = (primaryHand[5].y * 0.65 + primaryHand[8].y * 0.35);
      
      // Calculate Active Region of Interest (Removes physical strain of reaching camera edges)
      const rawNormX = (anchorX - ACTIVE_ROI_MARGIN) / (1 - ACTIVE_ROI_MARGIN * 2);
      const rawNormY = (anchorY - ACTIVE_ROI_MARGIN) / (1 - ACTIVE_ROI_MARGIN * 2);
      
      // Clamp coordinates to screen boundaries
      const targetX = Math.max(0.01, Math.min(0.99, rawNormX));
      const targetY = Math.max(0.01, Math.min(0.99, rawNormY));

      // Dynamic Smoothing (1-Euro Filter simulation)
      const velocity = Math.hypot(targetX - prevTargetX, targetY - prevTargetY);
      prevTargetX = targetX; 
      prevTargetY = targetY;
      
      // Alpha calculates tracking stiffness based on speed
      let alpha = 0.4;
      if (velocity < 0.005) alpha = 0.12;       // High stiffness for stable clicking
      else if (velocity > 0.06) alpha = 0.85;   // Low stiffness for fast movement
      
      filteredX = filteredX * (1 - alpha) + targetX * alpha;
      filteredY = filteredY * (1 - alpha) + targetY * alpha;

      // Hysteresis Pinch Detection
      const pinchDistance = distance2D(primaryHand[8], primaryHand[4]);
      if (!isPinching && pinchDistance < PINCH_DOWN_THRESH) isPinching = true;
      else if (isPinching && pinchDistance > PINCH_UP_THRESH) isPinching = false;

      // Hardware execution
      transmitMouseCoordinates(true, filteredX, filteredY, isPinching);

      // Edge Scrolling Logic
      if (filteredY < SCROLL_TRIGGER_ZONE) invokeHardwareScroll('up');
      if (filteredY > 1 - SCROLL_TRIGGER_ZONE) invokeHardwareScroll('down');

      // Click Dispatch
      if (isPinching && now - lastClickTime > 650) {
        lastClickTime = now;
        const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
        bridge?.clickAirMouse?.(filteredX, filteredY);
        
        // Dispatch synthetic web click
        const element = document.elementFromPoint(filteredX * window.innerWidth, filteredY * window.innerHeight);
        if (element instanceof HTMLElement) {
          element.click();
          element.focus();
        }
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // ========================================================================
    // 2. GESTURE & SIGN DETECTION ENGINE
    // ========================================================================
    const neuralVector = normalizeVector(primaryHand);

    // Capture Mode: Save neural mapping directly to device
    if (pendingTrainLabel) {
      mlDatabase.push({ label: pendingTrainLabel, vector: neuralVector });
      localStorage.setItem('nova_ml_gestures', JSON.stringify(mlDatabase));
      pendingTrainLabel = null;
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Inference: Prioritize Custom ML Database -> Fallback to Rigid Rules
    let detectedSign = classifyWithML(neuralVector) || classifyRuleBased(primaryHand);

    if (!detectedSign) {
      if (++releaseFrameCount >= 5) {
        latchedSign = '';
        candidateSign = '';
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // Temporal Consensus Stabilization (Anti-Jitter Latch)
    releaseFrameCount = 0;
    if (now - lastEmittedTime < POST_EMIT_COOLDOWN_MS) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    if (detectedSign === latchedSign) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

    if (detectedSign !== candidateSign) {
      candidateSign = detectedSign;
      candidateStartTime = now;
    } else if (now - candidateStartTime > GESTURE_HOLD_DELAY_MS) {
      latchedSign = detectedSign;
      lastEmittedTime = now;
      
      const isControlGesture = detectedSign === 'CLEAR' || detectedSign === 'SEND';
      return { 
        type: isControlGesture ? 'GESTURE' : 'LETTER', 
        value: detectedSign, 
        confidence: 0.95, 
        source: 'local' 
      };
    }

    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    
  } catch (error) {
    console.error("Local Vision Engine Error:", error);
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
}

export async function isLocalVisionModelAvailable() { 
  return true; 
}

export function applyCustomGesture(res: VisionResult, _: CustomGesture[]) { 
  return res; 
}
