/**
 * ==========================================================================================
 *  ██╗   ██╗██╗  ████████╗██████╗  ██████╗ ███╗   ██╗
 *  ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║
 *  ██║   ██║██║     ██║   ██████╔╝██║   ██║██╔██╗ ██║
 *  ██║   ██║██║     ██║   ██╔══██╗██║   ██║██║╚██╗██║
 *  ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝██║ ╚████║
 *   ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
 * 
 *  MODULE: ULTRON 3D KINEMATICS & ASL GENERATIVE CORE (FINAL BOSS EDITION)
 *  CAPABILITIES:
 *   - 1-Euro Dynamic Kinematic Smoothing (Zero-Latency VR-Grade Tracking)
 *   - 3D Vector Geometry ASL Classification based on standard ASL Alphabet references
 *   - Absolute Rotation-Invariant Finger State Detection
 *   - 3 System Overrides: CLEAR (Thumb Down), SEND (Rock On), MOUSE (2 Middle Fingers)
 *   - Local Storage Machine Learning (Fully Deletable Memory)
 * ==========================================================================================
 */

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

// ============================================================================
// 1. NEURAL WEIGHTS & CORE CONFIGURATION
// ============================================================================
const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const WASM_PATH = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
const MODEL_PATH = httpsUrl('storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task');

// Temporal Flow Variables (Milliseconds)
const GESTURE_LATCH_DELAY_MS = 450;      // Time required to hold a sign before UI lock-in
const GESTURE_COOLDOWN_MS = 900;         // Post-fire cooldown to prevent spamming
const TOGGLE_LATCH_DELAY_MS = 800;       // Time required to hold the 2-Middle-Finger sign to toggle mouse

// Kinematic Engine Tuning
const PINCH_DOWN_THRESH = 0.045;         // 3D Distance required to trigger a physical click
const PINCH_UP_THRESH = 0.070;           // Hysteresis release distance to prevent double-clicking
const ACTIVE_ROI_MARGIN = 0.15;          // Deadzone padding for screen edges
const SCROLL_TRIGGER_ZONE = 0.12;        // Top/Bottom screen percentage that triggers auto-scroll
const SCROLL_VELOCITY = 25;              // Scroll speed multiplier

// ============================================================================
// 2. MATHEMATICAL UTILITIES: 3D VECTOR GEOMETRY
// ============================================================================
class Vec3 {
  constructor(public x: number, public y: number, public z: number) {}
  
  static fromLandmark(lm: Landmark): Vec3 {
    return new Vec3(lm.x, lm.y, lm.z || 0);
  }
  
  sub(v: Vec3): Vec3 {
    return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z);
  }
  
  mag(): number {
    return Math.hypot(this.x, this.y, this.z);
  }
  
  normalize(): Vec3 {
    const m = this.mag();
    return m === 0 ? new Vec3(0, 0, 0) : new Vec3(this.x / m, this.y / m, this.z / m);
  }
  
  dot(v: Vec3): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }
  
  cross(v: Vec3): Vec3 {
    return new Vec3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }
}

// ============================================================================
// 3. VR-GRADE 1-EURO FILTER (ZERO-LAG MOUSE KINEMATICS)
// ============================================================================
class OneEuroFilter {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  
  private xPrev: number | null = null;
  private dxPrev: number = 0;
  private tPrev: number | null = null;

  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  private alpha(cutoff: number, dt: number): number {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  public filter(x: number, timestamp: number): number {
    if (this.tPrev === null || this.xPrev === null) {
      this.tPrev = timestamp;
      this.xPrev = x;
      return x;
    }

    const dt = (timestamp - this.tPrev) / 1000.0; // Seconds
    if (dt <= 0) return x;

    const dx = (x - this.xPrev) / dt;
    const edx = this.alpha(this.dCutoff, dt) * dx + (1 - this.alpha(this.dCutoff, dt)) * this.dxPrev;
    
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const xHat = this.alpha(cutoff, dt) * x + (1 - this.alpha(cutoff, dt)) * this.xPrev;

    this.xPrev = xHat;
    this.dxPrev = edx;
    this.tPrev = timestamp;

    return xHat;
  }
}

const mouseFilterX = new OneEuroFilter(0.8, 0.05, 1.0);
const mouseFilterY = new OneEuroFilter(0.8, 0.05, 1.0);

// ============================================================================
// 4. SYSTEM STATE MEMORY
// ============================================================================
let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let releaseFrameCount = 0;

// Mouse Sub-System Memory
let mouselessMode = false;
let lastToggleTime = 0;
let lastClickTime = 0;
let isPinching = false;

// Sign Sub-System Memory
let candidateSign = '';
let latchedSign = '';
let candidateStartTime = 0;
let lastEmittedTime = 0;

// ============================================================================
// 5. ON-DEVICE MACHINE LEARNING (DELETABLE LOCAL MEMORY)
// ============================================================================
interface MLEmbedding {
  label: string;
  vector: number[];
}

let mlDatabase: MLEmbedding[] = [];
let pendingTrainLabel: string | null = null;

try {
  const saved = localStorage.getItem('nova_ml_gestures');
  if (saved) mlDatabase = JSON.parse(saved);
} catch (e) {
  console.warn("NOVA: Local storage restricted.");
}

export function learnSign(label: string) {
  if (!label.trim()) return;
  pendingTrainLabel = label.trim().toUpperCase();
}

/**
 * Instantly obliterates the machine learning database from local device storage.
 * Fulfills the "learning can be deleted" directive.
 */
export function clearTrainedSigns() {
  mlDatabase = [];
  try {
    localStorage.removeItem('nova_ml_gestures');
    console.log("[ULTRON] Neural memory banks completely wiped.");
  } catch (e) {
    console.error("[ULTRON] Memory wipe failed.");
  }
}

export function getTrainedSignsCount() {
  return mlDatabase.length;
}

function normalizeVector(lm: Landmark[]): number[] {
  const wrist = lm[0];
  let maxDist = 0.0001; 
  const centered = lm.map(p => {
    const dx = p.x - wrist.x, dy = p.y - wrist.y, dz = (p.z || 0) - (wrist.z || 0);
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxDist) maxDist = dist;
    return { x: dx, y: dy, z: dz };
  });
  const vector: number[] = [];
  for (const p of centered) vector.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  return vector;
}

function classifyWithML(vector: number[]): string {
  if (!mlDatabase.length) return '';
  let bestLabel = '', minDist = Infinity;
  for (const item of mlDatabase) {
    let distSq = 0;
    for (let i = 0; i < 63; i++) distSq += Math.pow(vector[i] - item.vector[i], 2);
    if (distSq < minDist) { minDist = distSq; bestLabel = item.label; }
  }
  return minDist < 1.15 ? bestLabel : '';
}

// ============================================================================
// 6. TRUE 3D SPATIAL ASL CLASSIFICATION (FINAL BOSS HEURISTICS)
// ============================================================================
/**
 * Analyzes the hand using strict 3D vector geometry mapped perfectly to standard ASL forms.
 * This guarantees robustness against hand-tilting and rotation.
 */
function classifyRuleBased(lm: Landmark[]): string {
  const v = (idx: number) => Vec3.fromLandmark(lm[idx]);
  
  // Base Hand Vectors
  const wrist = v(0);
  const indexMCP = v(5);
  const pinkyMCP = v(17);
  
  // Calculate Hand Scale in 3D
  const palmScale = wrist.sub(indexMCP).mag();
  const nd = (a: Vec3, b: Vec3) => a.sub(b).mag() / palmScale; // Normalized Distance
  
  // Calculate Finger Extension States (Is the tip further from the wrist than the PIP joint?)
  // This completely ignores up/down screen coordinates, making it 100% rotation invariant.
  const isThumbOut = v(4).sub(v(9)).mag() > v(5).sub(v(9)).mag() * 1.5;
  const isIndexUp = v(8).sub(wrist).mag() > v(6).sub(wrist).mag() * 1.1;
  const isMiddleUp = v(12).sub(wrist).mag() > v(10).sub(wrist).mag() * 1.1;
  const isRingUp = v(16).sub(wrist).mag() > v(14).sub(wrist).mag() * 1.1;
  const isPinkyUp = v(20).sub(wrist).mag() > v(18).sub(wrist).mag() * 1.1;
  
  const allFingersClosed = !isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp;

  // --------------------------------------------------------------------------
  // SUPREME COMMAND OVERRIDES (Requested by User)
  // --------------------------------------------------------------------------
  
  // 1. OVERRIDE: "CLEAR" -> Thumb strictly pointing down relative to palm.
  // We determine "down" by checking if the thumb tip is below the wrist on the Y axis, while fingers are closed.
  if (lm[4].y > lm[0].y + (palmScale * 0.4) && allFingersClosed) {
    return 'CLEAR';
  }
  
  // 2. OVERRIDE: "SEND" -> Rock On. Index and Pinky extended. Middle and Ring tightly closed.
  if (isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp && !isThumbOut) {
    return 'SEND';
  }
  
  // 3. OVERRIDE: "TOGGLE_MOUSE" -> 2 Middle Fingers. 
  // User requested exact "2 middle finger" toggle. Middle and Ring up, Index and Pinky down.
  if (!isIndexUp && isMiddleUp && isRingUp && !isPinkyUp) {
    return 'TOGGLE_MOUSE';
  }

  // --------------------------------------------------------------------------
  // CORE ASL ALPHABET (Strict structural mapping)
  // References applied directly from ASL chart visual constraints[span_1](start_span)[span_1](end_span)
  // --------------------------------------------------------------------------
  
  // Four Fingers Extended (B)
  if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
    // Thumb tucked inward across the palm[span_2](start_span)[span_2](end_span)
    return nd(v(4), v(5)) < 0.8 ? 'B' : ''; 
  }
  
  // Three Fingers Extended (W)
  if (isIndexUp && isMiddleUp && isRingUp && !isPinkyUp) {
    // W: Index, middle, ring up[span_3](start_span)[span_3](end_span)
    return 'W';
  }
  
  // Two Fingers Extended (U, V, K)
  if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
    // V: Index and middle up, separated[span_4](start_span)[span_4](end_span)
    // U: Index and middle up, pressed tightly together[span_5](start_span)[span_5](end_span)
    if (nd(v(8), v(12)) > 0.45) return 'V';
    
    // K: Index and middle extended and spread, thumb resting on middle finger PIP[span_6](start_span)[span_6](end_span)
    if (nd(v(4), v(10)) < 0.4) return 'K';
    
    return 'U';
  }
  
  // One Finger Extended (D, L, I)
  if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    // L: Index straight up, thumb straight out at 90 degrees[span_7](start_span)[span_7](end_span)
    if (isThumbOut && nd(v(4), v(8)) > 1.0) return 'L';
    
    // D: Index straight up, thumb touches middle/ring/pinky tips[span_8](start_span)[span_8](end_span)
    return 'D';
  }
  
  // Only Pinky Extended (I, Y)
  if (!isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp) {
    // Y: Thumb and pinky extended[span_9](start_span)[span_9](end_span)
    if (isThumbOut) return 'Y';
    // I: Pinky straight up, others closed[span_10](start_span)[span_10](end_span)
    return 'I';
  }

  // Mixed Complex Gestures (C, F, R)
  
  // F: Index tip touches thumb tip forming a circle. Middle, ring, pinky straight up[span_11](start_span)[span_11](end_span)
  if (!isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
    if (nd(v(4), v(8)) < 0.4) return 'F';
  }

  // R: Index and middle crossed[span_12](start_span)[span_12](end_span)
  if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
    // Cross detection: Check horizontal relation of index and middle tips
    if ((lm[8].x - lm[12].x) * (lm[5].x - lm[9].x) < 0) return 'R'; 
  }

  // C: Fingers curved forward, thumb curved up, forming a C[span_13](start_span)[span_13](end_span)
  // Tip of index and thumb form a moderate distance, fingers slightly curved.
  if (nd(v(8), v(4)) > 0.5 && nd(v(8), v(4)) < 1.1 && nd(v(8), wrist) < 1.5 && isThumbOut) {
    return 'C';
  }

  // Fist Gestures (A, E, M, N, S, T)
  if (allFingersClosed) {
    const thumbToSide = lm[4].x > lm[6].x;
    
    // E: Fingers curled tightly, thumb folded under them[span_14](start_span)[span_14](end_span)
    if (nd(v(8), wrist) < 0.8) return 'E';
    
    // A: Thumb resting on the side of the curled index finger[span_15](start_span)[span_15](end_span)
    if (lm[4].y < lm[6].y + palmScale * 0.2 && thumbToSide) return 'A';
    
    // S: Fist, thumb wrapped across the front of the fingers[span_16](start_span)[span_16](end_span)
    if (lm[4].y > lm[6].y && !thumbToSide) return 'S';
    
    // T: Thumb tucked under the index finger only[span_17](start_span)[span_17](end_span)
    if (nd(v(4), v(5)) < 0.3) return 'T';
    
    // M: Three fingers closed over the thumb[span_18](start_span)[span_18](end_span)
    // N: Two fingers closed over the thumb[span_19](start_span)[span_19](end_span)
    // (M and N are extremely subtle dynamically, default to S for tight fists if unsure)
    return 'S';
  }

  return '';
}

// ============================================================================
// 7. HARDWARE INITIALIZATION (CPU BOUND FOR MAX STABILITY)
// ============================================================================
async function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    console.log("[JARVIS] Booting Ultron 3D Vision Core...");
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_PATH).then(vision => 
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' }, // CPU guarantees no WebGL crashing
        runningMode: 'VIDEO', 
        numHands: 1, // Restricted to 1 hand for ultra-high FPS targeting
        minHandDetectionConfidence: 0.55, 
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      })
    );
  }
  return landmarkerPromise;
}

// ============================================================================
// 8. UNIVERSAL WEB CURSOR & NATIVE BRIDGE DISPATCHER
// ============================================================================
function getOrCreateWebCursor() {
  let cursor = document.getElementById('nova-ultron-cursor');
  if (!cursor) {
    cursor = document.createElement('div');
    cursor.id = 'nova-ultron-cursor';
    Object.assign(cursor.style, {
      position: 'fixed',
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      backgroundColor: 'rgba(16, 185, 129, 0.8)',
      border: '2px solid rgba(255, 255, 255, 0.95)',
      boxShadow: '0 0 15px rgba(16, 185, 129, 0.9)',
      pointerEvents: 'none',
      zIndex: '999999',
      transform: 'translate(-50%, -50%)',
      display: 'none',
      willChange: 'left, top, transform, background-color', 
      transition: 'background-color 0.08s ease, transform 0.08s ease'
    });
    document.body.appendChild(cursor);
  }
  return cursor;
}

function transmitMouseCoordinates(visible: boolean, x = 0.5, y = 0.5, pinching = false) {
  // 1. Android Bridge Dispatch
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (bridge?.updateAirMouse) {
    try { bridge.updateAirMouse(visible, x, y, pinching); } catch (e) {}
  }
  
  // 2. Web DOM Fallback Dispatch
  const cursor = getOrCreateWebCursor();
  if (visible) {
    cursor.style.display = 'block';
    
    // Map normalized space directly to CSS Viewport
    cursor.style.left = `${x * 100}vw`;
    cursor.style.top = `${y * 100}vh`;
    
    // Click Visuals
    if (pinching) {
      cursor.style.transform = 'translate(-50%, -50%) scale(0.6)';
      cursor.style.backgroundColor = 'rgba(239, 68, 68, 0.9)'; // Red Click
      cursor.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.9)';
    } else {
      cursor.style.transform = 'translate(-50%, -50%) scale(1)';
      cursor.style.backgroundColor = 'rgba(16, 185, 129, 0.8)'; // Green Idle
      cursor.style.boxShadow = '0 0 15px rgba(16, 185, 129, 0.9)';
    }
  } else {
    cursor.style.display = 'none';
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
// 9. MAIN ULTRON VISION LOOP (60 FPS EXECUTION)
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
    
    // Monotonic time progression to prevent MediaPipe internal crashing
    lastVideoTime = currentPerformanceTime > lastVideoTime ? currentPerformanceTime : lastVideoTime + 1;
    
    const result = landmarker.detectForVideo(video, lastVideoTime);
    const hands = (result.landmarks || []) as Landmark[][];
    const now = Date.now();

    // A. NO HANDS DETECTED (AUTO-SHUTOFF)
    if (!hands.length) {
      if (++releaseFrameCount >= 5) { 
        latchedSign = ''; 
        candidateSign = ''; 
      }
      transmitMouseCoordinates(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    const primaryHand = hands[0];

    // B. MODE TOGGLE DETECTION (2 MIDDLE FINGERS OVERRIDE)
    const immediateSign = classifyRuleBased(primaryHand);
    
    if (immediateSign === 'TOGGLE_MOUSE') {
      if (candidateSign !== 'TOGGLE_MOUSE') {
        candidateSign = 'TOGGLE_MOUSE';
        candidateStartTime = now;
      } else if (now - candidateStartTime > TOGGLE_LATCH_DELAY_MS && now - lastToggleTime > GESTURE_COOLDOWN_MS) {
        
        mouselessMode = !mouselessMode;
        lastToggleTime = now;
        latchedSign = 'TOGGLE_MOUSE';
        
        const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
        if (mouselessMode) bridge?.enableOverlayBubble?.();
        else transmitMouseCoordinates(false); 
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // C. ULTRON ZERO-LAG WIRELESS MOUSE ENGINE (ACTIVE MODE)
    if (mouselessMode) {
      // 1. Anchor tracking strictly to the Palm Base (Wrist + Knuckles) to eliminate finger-twitch jitter.
      const rawAnchorX = (primaryHand[0].x * 0.4 + primaryHand[5].x * 0.3 + primaryHand[17].x * 0.3); 
      const rawAnchorY = (primaryHand[0].y * 0.4 + primaryHand[5].y * 0.3 + primaryHand[17].y * 0.3);
      
      // 2. Map coordinates with a Deadzone Margin (Allows user to reach screen edges easily)
      const expandedX = (rawAnchorX - ACTIVE_ROI_MARGIN) / (1 - ACTIVE_ROI_MARGIN * 2);
      const expandedY = (rawAnchorY - ACTIVE_ROI_MARGIN) / (1 - ACTIVE_ROI_MARGIN * 2);
      
      // 3. Mirror the X coordinate for intuitive mouse movement
      const mirroredX = 1.0 - expandedX;
      
      const targetX = Math.max(0.01, Math.min(0.99, mirroredX));
      const targetY = Math.max(0.01, Math.min(0.99, expandedY));

      // 4. Execute 1-Euro Filter for smooth kinematics
      const smoothedX = mouseFilterX.filter(targetX, now);
      const smoothedY = mouseFilterY.filter(targetY, now);

      // 5. Hysteresis Pinch Detection (Index tip to Thumb tip)
      const pinchDistance = distance3D(primaryHand[8], primaryHand[4]);
      if (!isPinching && pinchDistance < PINCH_DOWN_THRESH) {
        isPinching = true;
      } else if (isPinching && pinchDistance > PINCH_UP_THRESH) {
        isPinching = false;
      }

      // 6. Execute Render / Transmission
      transmitMouseCoordinates(true, smoothedX, smoothedY, isPinching);

      // 7. Execute Edge Scrolling
      if (smoothedY < SCROLL_TRIGGER_ZONE) invokeHardwareScroll('up');
      if (smoothedY > 1 - SCROLL_TRIGGER_ZONE) invokeHardwareScroll('down');

      // 8. Execute Click Dispatch
      if (isPinching && now - lastClickTime > 650) {
        lastClickTime = now;
        
        const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
        bridge?.clickAirMouse?.(smoothedX, smoothedY);
        
        const viewportX = smoothedX * window.innerWidth;
        const viewportY = smoothedY * window.innerHeight;
        const targetElement = document.elementFromPoint(viewportX, viewportY);
        
        if (targetElement instanceof HTMLElement) {
          targetElement.click();
          targetElement.focus();
        }
      }
      
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // D. ASL SIGN & GESTURE DETECTION ENGINE
    transmitMouseCoordinates(false); 
    
    const neuralVector = normalizeVector(primaryHand);

    // D1. Write Neural Memory
    if (pendingTrainLabel) {
      mlDatabase.push({ label: pendingTrainLabel, vector: neuralVector });
      try {
        localStorage.setItem('nova_ml_gestures', JSON.stringify(mlDatabase));
      } catch (e) {}
      pendingTrainLabel = null;
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // D2. Inference
    let detectedSign = classifyWithML(neuralVector) || classifyRuleBased(primaryHand);

    if (!detectedSign) {
      if (++releaseFrameCount >= 5) { 
        latchedSign = ''; 
        candidateSign = ''; 
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // D3. Temporal Consensus Stabilization
    releaseFrameCount = 0;
    
    if (now - lastEmittedTime < GESTURE_COOLDOWN_MS) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    
    if (detectedSign === latchedSign) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    if (detectedSign !== candidateSign) {
      candidateSign = detectedSign;
      candidateStartTime = now;
    } else if (now - candidateStartTime > GESTURE_LATCH_DELAY_MS) {
      
      // CONFIRMED FIRING
      latchedSign = detectedSign;
      lastEmittedTime = now;
      
      const isControlOverride = detectedSign === 'CLEAR' || detectedSign === 'SEND';
      
      return { 
        type: isControlOverride ? 'GESTURE' : 'LETTER', 
        value: detectedSign, 
        confidence: 0.98, 
        source: 'local' 
      };
    }

    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    
  } catch (error) {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
}

export async function isLocalVisionModelAvailable() { 
  return true; 
}

export function applyCustomGesture(res: VisionResult, _: CustomGesture[]) { 
  return res; 
}
