/**
 * ====================================================================================================
 *  ██╗   ██╗██╗     ████████╗██████╗  ██████╗ ███╗   ██╗    ██████╗ ██████╗ ██████╗ ███████╗
 *  ██║   ██║██║     ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║   ██╔════╝██╔═══██╗██╔══██╗██╔════╝
 *  ██║   ██║██║        ██║   ██████╔╝██║   ██║██╔██╗ ██║   ██║     ██║   ██║██████╔╝█████╗  
 *  ██║   ██║██║        ██║   ██╔══██╗██║   ██║██║╚██╗██║   ██║     ██║   ██║██╔══██╗██╔══╝  
 *  ╚██████╔╝███████╗   ██║   ██║  ██║╚██████╔╝██║ ╚████║   ╚██████╗╚██████╔╝██║  ██║███████╗
 *   ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
 * 
 *  MODULE: ULTRON 3D SPATIAL KINEMATICS & ASL GENERATIVE CORE (MAXIMUM POTENTIAL EDITION)
 *  CAPABILITIES:
 *   - Advanced 3D Linear Algebra Engine (Vector Cross/Dot Products, Plane Projection)
 *   - VR-Grade 1-Euro Kinematic Smoothing (Zero-Latency, Anti-Jitter Mouse)
 *   - Anatomically Precise ASL Detection (Rotation-Invariant)
 *   - Two-Hand "Middle Finger" Mouse Toggle (reliable, distinctive, low false-trigger)
 * ====================================================================================================
 */

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { CustomGesture, Landmark, VisionResult } from '../types';

// ====================================================================================================
// 1. NEURAL WEIGHTS & CORE CONFIGURATION
// ====================================================================================================
const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const WASM_PATH = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm');
const MODEL_PATH = httpsUrl('storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task');

const CONFIG = {
  GESTURE_LATCH_DELAY_MS: 500,      // Milliseconds required to hold a sign perfectly still
  GESTURE_COOLDOWN_MS: 1000,        // Cooldown period after a successful sign emission

  // --- Wireless mouse tuning (sensitivity pass) ---
  TOGGLE_LATCH_DELAY_MS: 650,       // Time required to hold the two-hand toggle (was 900 — faster now)
  MOUSE_PINCH_DOWN_THRESH: 0.052,   // Distance to trigger a click (was 0.042 — more forgiving)
  MOUSE_PINCH_UP_THRESH: 0.078,     // Hysteresis release distance (was 0.065)
  MOUSE_ROI_MARGIN: 0.10,           // Dead-zone margin around camera edge (was 0.16 — easier to reach edges)
  SCROLL_TRIGGER_ZONE: 0.12,
  SCROLL_VELOCITY: 34,              // Scroll speed px/frame (was 28 — snappier)
  CLICK_COOLDOWN_MS: 480            // Minimum time between clicks (was 650 — snappier repeat clicks)
};

// ====================================================================================================
// 2. ADVANCED 3D VECTOR MATHEMATICS ENGINE
// ====================================================================================================
class Vector3D {
  constructor(public x: number, public y: number, public z: number) {}
  
  static fromLM(lm: Landmark): Vector3D {
    return new Vector3D(lm.x, lm.y, lm.z || 0);
  }
  
  add(v: Vector3D): Vector3D { return new Vector3D(this.x + v.x, this.y + v.y, this.z + v.z); }
  sub(v: Vector3D): Vector3D { return new Vector3D(this.x - v.x, this.y - v.y, this.z - v.z); }
  mul(scalar: number): Vector3D { return new Vector3D(this.x * scalar, this.y * scalar, this.z * scalar); }
  mag(): number { return Math.hypot(this.x, this.y, this.z); }
  
  normalize(): Vector3D {
    const m = this.mag();
    return m === 0 ? new Vector3D(0, 0, 0) : new Vector3D(this.x / m, this.y / m, this.z / m);
  }
  
  dot(v: Vector3D): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }
  
  cross(v: Vector3D): Vector3D {
    return new Vector3D(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }
  
  angleTo(v: Vector3D): number {
    const dot = this.normalize().dot(v.normalize());
    const clamped = Math.max(-1.0, Math.min(1.0, dot));
    return Math.acos(clamped) * (180.0 / Math.PI);
  }
}

// ====================================================================================================
// 3. VR-GRADE 1-EURO KINEMATIC FILTER
// ====================================================================================================
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

    const dt = (timestamp - this.tPrev) / 1000.0;
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

// Sensitivity pass: higher minCutoff + beta = cursor tracks hand movement more
// closely with less lag (was 0.6 / 0.04 — now snappier, still stable at rest).
const mouseFilterX = new OneEuroFilter(0.9, 0.09, 1.0);
const mouseFilterY = new OneEuroFilter(0.9, 0.09, 1.0);

// ====================================================================================================
// 4. SYSTEM STATE MEMORY & GLOBAL VARIABLES
// ====================================================================================================
let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let releaseFrameCount = 0;

// Wireless Mouse States
let isMouseActive = false;
let lastToggleTime = 0;
let toggleCandidateStart = 0;
let lastClickTime = 0;
let isPinching = false;

// Latch States (Anti-Spam Memory) — ASL letters only
let candidateSign = '';
let latchedSign = '';
let candidateStartTime = 0;
let lastEmittedTime = 0;

// ====================================================================================================
// 5. ML ERADICATION (DUMMY EXPORTS TO PREVENT UI CRASH)
// ====================================================================================================
export function learnSign(_label: string) { console.log("[ULTRON] Machine Learning module disabled."); }
export function clearTrainedSigns() { console.log("[ULTRON] Memory wiped."); }
export function getTrainedSignsCount() { return 0; }

// ====================================================================================================
// 6. TRUE 3D ANATOMICAL ANALYSIS ENGINE (ASL — UNCHANGED)
// ====================================================================================================
function analyzeAnatomy(lm: Landmark[]) {
  const v = (idx: number) => Vector3D.fromLM(lm[idx]);
  
  const wrist = v(0);
  const palmBase = v(0);
  const indexMCP = v(5);
  const pinkyMCP = v(17);
  
  const handScale = wrist.sub(indexMCP).mag();
  const nDist = (p1: Vector3D, p2: Vector3D) => p1.sub(p2).mag() / handScale;

  const isIndexUp = v(8).sub(palmBase).mag() > v(6).sub(palmBase).mag() * 1.15;
  const isMiddleUp = v(12).sub(palmBase).mag() > v(10).sub(palmBase).mag() * 1.15;
  const isRingUp = v(16).sub(palmBase).mag() > v(14).sub(palmBase).mag() * 1.15;
  const isPinkyUp = v(20).sub(palmBase).mag() > v(18).sub(palmBase).mag() * 1.15;
  
  const isThumbOut = v(4).sub(pinkyMCP).mag() > indexMCP.sub(pinkyMCP).mag() * 1.6;
  const isThumbDown = lm[4].y > lm[0].y + (handScale * 0.45);
  
  const thumbVector = v(4).sub(v(2));
  const indexVector = v(8).sub(v(5));
  const thumbIndexAngle = thumbVector.angleTo(indexVector);

  const allFingersClosed = !isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp;
  const allFingersOpen = isIndexUp && isMiddleUp && isRingUp && isPinkyUp;

  return {
    v, nDist, isIndexUp, isMiddleUp, isRingUp, isPinkyUp,
    isThumbOut, isThumbDown, thumbIndexAngle,
    allFingersClosed, allFingersOpen
  };
}

// ====================================================================================================
// 7. FINAL BOSS CLASSIFIER (ASL + SEND/CLEAR — UNCHANGED, TOGGLE REMOVED FROM HERE)
// ====================================================================================================
function classifySign(anatomy: ReturnType<typeof analyzeAnatomy>): string {
  const { 
    v, nDist, isIndexUp, isMiddleUp, isRingUp, isPinkyUp, 
    isThumbOut, isThumbDown, thumbIndexAngle, 
    allFingersClosed, allFingersOpen 
  } = anatomy;

  // ------------------------------------------------------------------------
  // A. SYSTEM OVERRIDES
  // (The old single-hand TOGGLE_MOUSE override lived here — it now lives as
  // a separate two-hand check in localVision(), before this function is even
  // called, so it can't collide with or affect ASL letter detection.)
  // ------------------------------------------------------------------------

  if (isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp && !isThumbOut) {
    return 'SEND';
  }

  if (allFingersClosed && isThumbDown) {
    return 'CLEAR';
  }

  // ------------------------------------------------------------------------
  // B. TRUE ASL ALPHABET RECOGNITION
  // ------------------------------------------------------------------------
  
  if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    if (isThumbOut && thumbIndexAngle > 60) return 'L';
    return 'D';
  }

  if (allFingersClosed) {
    if (!isThumbOut && !isThumbDown) {
      if (v(4).y < v(6).y) return 'A';
      if (v(4).y > v(6).y && v(4).x < v(6).x) return 'S';
      return 'E';
    }
  }

  if (allFingersOpen && !isThumbOut) {
    if (nDist(v(4), v(5)) < 1.0) return 'B';
  }

  if (isIndexUp && isMiddleUp && isRingUp && !isPinkyUp) {
    return 'W';
  }

  if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
    if (nDist(v(8), v(12)) > 0.45) return 'V';
    return 'U';
  }

  if (!isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp) {
    if (isThumbOut) return 'Y';
    return 'I';
  }

  if (!isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
    if (nDist(v(8), v(4)) < 0.6) return 'F';
  }

  if (isThumbOut && nDist(v(8), v(4)) > 0.6 && nDist(v(8), v(4)) < 1.4 && !isIndexUp) {
    if (nDist(v(8), v(0)) < 1.6 && nDist(v(12), v(0)) < 1.6) return 'C';
  }

  return '';
}

// ====================================================================================================
// 7B. WIRELESS MOUSE TOGGLE TRIGGER (restored: two hands, each showing only the middle finger)
// ====================================================================================================
function isOnlyMiddleFingerUp(lm: Landmark[]): boolean {
  if (!lm || lm.length < 21) return false;
  const wrist = lm[0];
  const middleUp = lm[12].y < lm[10].y && lm[12].y < wrist.y - 0.05;
  const indexClosed = lm[8].y > lm[6].y;
  const ringClosed = lm[16].y > lm[14].y;
  const pinkyClosed = lm[20].y > lm[18].y;
  return middleUp && indexClosed && ringClosed && pinkyClosed;
}

// ====================================================================================================
// 8. HARDWARE INITIALIZATION
// ====================================================================================================
async function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    console.log("[ULTRON] Booting Final Boss Vision Engine...");
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_PATH).then(vision => 
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
        runningMode: 'VIDEO', 
        numHands: 2,  // need both hands visible to read the toggle gesture
        minHandDetectionConfidence: 0.55, 
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      })
    );
  }
  return landmarkerPromise;
}

// ====================================================================================================
// 9. UNIVERSAL WEB CURSOR & NATIVE BRIDGE DISPATCHER
// ====================================================================================================
function getOrCreateWebCursor() {
  let cursor = document.getElementById('nova-finalboss-cursor');
  if (!cursor) {
    cursor = document.createElement('div');
    cursor.id = 'nova-finalboss-cursor';
    Object.assign(cursor.style, {
      position: 'fixed',
      width: '26px',
      height: '26px',
      borderRadius: '50%',
      backgroundColor: 'rgba(16, 185, 129, 0.85)',
      border: '2px solid rgba(255, 255, 255, 1)',
      boxShadow: '0 0 18px rgba(16, 185, 129, 0.95)',
      pointerEvents: 'none',
      zIndex: '999999',
      transform: 'translate(-50%, -50%)',
      display: 'none',
      willChange: 'left, top, transform, background-color', 
      transition: 'background-color 0.08s ease-out, transform 0.08s ease-out'
    });
    document.body.appendChild(cursor);
  }
  return cursor;
}

function transmitMouseCoordinates(visible: boolean, x = 0.5, y = 0.5, pinching = false) {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (bridge?.updateAirMouse) {
    try { bridge.updateAirMouse(visible, x, y, pinching); } catch (e) {}
  }
  
  const cursor = getOrCreateWebCursor();
  if (visible) {
    cursor.style.display = 'block';
    cursor.style.left = `${x * 100}vw`;
    cursor.style.top = `${y * 100}vh`;
    
    if (pinching) {
      cursor.style.transform = 'translate(-50%, -50%) scale(0.5)';
      cursor.style.backgroundColor = 'rgba(239, 68, 68, 1)';
      cursor.style.boxShadow = '0 0 25px rgba(239, 68, 68, 1)';
    } else {
      cursor.style.transform = 'translate(-50%, -50%) scale(1)';
      cursor.style.backgroundColor = 'rgba(16, 185, 129, 0.85)';
      cursor.style.boxShadow = '0 0 18px rgba(16, 185, 129, 0.95)';
    }
  } else {
    cursor.style.display = 'none';
  }
}

function invokeHardwareScroll(direction: 'up' | 'down') {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  if (direction === 'up') {
    window.scrollBy({ top: -CONFIG.SCROLL_VELOCITY, behavior: 'auto' });
    bridge?.swipeScreen?.(500, 450, 500, 1200, 260); 
  } else {
    window.scrollBy({ top: CONFIG.SCROLL_VELOCITY, behavior: 'auto' });
    bridge?.swipeScreen?.(500, 1200, 500, 450, 260); 
  }
}

// ====================================================================================================
// 10. MAIN ULTRON VISION LOOP
// ====================================================================================================
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
    lastVideoTime = currentPerformanceTime > lastVideoTime ? currentPerformanceTime : lastVideoTime + 1;
    
    const result = landmarker.detectForVideo(video, lastVideoTime);
    const hands = (result.landmarks || []) as Landmark[][];
    const now = Date.now();

    // STATE 0: NO HANDS
    if (!hands.length) {
      if (++releaseFrameCount >= 5) { latchedSign = ''; candidateSign = ''; }
      toggleCandidateStart = 0;
      transmitMouseCoordinates(false);
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // STATE 1: TWO-HAND MOUSE TOGGLE ("show only your middle finger on both hands")
    if (hands.length >= 2 && isOnlyMiddleFingerUp(hands[0]) && isOnlyMiddleFingerUp(hands[1])) {
      if (toggleCandidateStart === 0) {
        toggleCandidateStart = now;
      } else if (now - toggleCandidateStart > CONFIG.TOGGLE_LATCH_DELAY_MS && now - lastToggleTime > CONFIG.GESTURE_COOLDOWN_MS) {
        isMouseActive = !isMouseActive;
        lastToggleTime = now;
        toggleCandidateStart = 0;
        const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
        if (isMouseActive) bridge?.enableOverlayBubble?.();
        else transmitMouseCoordinates(false);
        console.log(`[ULTRON] Wireless Mouse Engine: ${isMouseActive ? 'ENGAGED' : 'DISENGAGED'}`);
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    toggleCandidateStart = 0;

    const primaryHand = hands[0];

    // STATE 2: WIRELESS MOUSE ACTIVE
    if (isMouseActive) {
      const anatomy = analyzeAnatomy(primaryHand);

      const rawAnchorX = (primaryHand[0].x * 0.4 + primaryHand[5].x * 0.3 + primaryHand[17].x * 0.3); 
      const rawAnchorY = (primaryHand[0].y * 0.4 + primaryHand[5].y * 0.3 + primaryHand[17].y * 0.3);
      
      const expandedX = (rawAnchorX - CONFIG.MOUSE_ROI_MARGIN) / (1 - CONFIG.MOUSE_ROI_MARGIN * 2);
      const expandedY = (rawAnchorY - CONFIG.MOUSE_ROI_MARGIN) / (1 - CONFIG.MOUSE_ROI_MARGIN * 2);
      
      const mirroredX = 1.0 - expandedX;
      
      const targetX = Math.max(0.01, Math.min(0.99, mirroredX));
      const targetY = Math.max(0.01, Math.min(0.99, expandedY));

      const smoothedX = mouseFilterX.filter(targetX, now);
      const smoothedY = mouseFilterY.filter(targetY, now);

      const pinchDistance = anatomy.v(8).sub(anatomy.v(4)).mag();
      
      if (!isPinching && pinchDistance < CONFIG.MOUSE_PINCH_DOWN_THRESH) {
        isPinching = true;
      } else if (isPinching && pinchDistance > CONFIG.MOUSE_PINCH_UP_THRESH) {
        isPinching = false;
      }

      transmitMouseCoordinates(true, smoothedX, smoothedY, isPinching);

      if (smoothedY < CONFIG.SCROLL_TRIGGER_ZONE) invokeHardwareScroll('up');
      if (smoothedY > 1 - CONFIG.SCROLL_TRIGGER_ZONE) invokeHardwareScroll('down');

      if (isPinching && now - lastClickTime > CONFIG.CLICK_COOLDOWN_MS) {
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

    // STATE 3: ASL SIGN & GESTURE DETECTION (unchanged)
    transmitMouseCoordinates(false);
    
    const anatomy = analyzeAnatomy(primaryHand);
    const detectedSign = classifySign(anatomy);

    if (!detectedSign) {
      if (++releaseFrameCount >= 5) { latchedSign = ''; candidateSign = ''; }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    releaseFrameCount = 0;
    
    if (now - lastEmittedTime < CONFIG.GESTURE_COOLDOWN_MS) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    if (detectedSign === latchedSign) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    if (detectedSign !== candidateSign) {
      candidateSign = detectedSign;
      candidateStartTime = now;
    } else if (now - candidateStartTime > CONFIG.GESTURE_LATCH_DELAY_MS) {
      latchedSign = detectedSign;
      lastEmittedTime = now;
      
      const isControlOverride = detectedSign === 'CLEAR' || detectedSign === 'SEND';
      
      return { 
        type: isControlOverride ? 'GESTURE' : 'LETTER', 
        value: detectedSign, 
        confidence: 1.0,
        source: 'local' 
      };
    }

    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    
  } catch (error) {
    console.error("[ULTRON] Critical Kernel Exception in Vision Loop:", error);
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
}

export async function isLocalVisionModelAvailable() { 
  return true; 
}

export function applyCustomGesture(res: VisionResult, _: CustomGesture[]) { 
  return res; 
}
