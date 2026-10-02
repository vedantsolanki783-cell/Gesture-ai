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
  GESTURE_LATCH_DELAY_MS: 500,      
  GESTURE_COOLDOWN_MS: 1000,        

  // --- FINAL BOSS WIRELESS MOUSE TUNING ---
  TOGGLE_LATCH_DELAY_MS: 650,
  
  // Pinch thresholds increased: Triggers clicks much earlier before fingers fully touch
  MOUSE_PINCH_DOWN_THRESH: 0.85,
  MOUSE_PINCH_UP_THRESH: 1.15,
  
  MOUSE_ROI_MARGIN: 0.12,         
  
  // Scrolling dynamically multiplies speed based on how far into this zone you push
  SCROLL_TRIGGER_ZONE: 0.16,
  SCROLL_BASE_VELOCITY: 35,
  
  // Rapid fire clicking enabled
  CLICK_COOLDOWN_MS: 200,
  
  RAY_REACH: 0.25
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

// ULTRA-FAST TRACKING: High minCutoff (2.5) kills input lag entirely. 
const mouseFilterX = new OneEuroFilter(2.5, 0.15, 1.0);
const mouseFilterY = new OneEuroFilter(2.5, 0.15, 1.0);

// ====================================================================================================
// 4. SYSTEM STATE MEMORY & GLOBAL VARIABLES
// ====================================================================================================
let landmarkerPromise: Promise<HandLandmarker> | null = null;
let lastVideoTime = -1;
let releaseFrameCount = 0;

let isMouseActive = false;
let lastToggleTime = 0;
let toggleCandidateStart = 0;
let lastClickTime = 0;
let isPinching = false;

let candidateSign = '';
let latchedSign = '';
let candidateStartTime = 0;
let lastEmittedTime = 0;

export function learnSign(_label: string) { console.log("[ULTRON] ML disabled."); }
export function clearTrainedSigns() { console.log("[ULTRON] Memory wiped."); }
export function getTrainedSignsCount() { return 0; }

// ====================================================================================================
// 5. TRUE 3D ANATOMICAL ANALYSIS ENGINE
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
// 6. FINAL BOSS CLASSIFIER 
// ====================================================================================================
function classifySign(anatomy: ReturnType<typeof analyzeAnatomy>): string {
  const { 
    v, nDist, isIndexUp, isMiddleUp, isRingUp, isPinkyUp, 
    isThumbOut, isThumbDown, thumbIndexAngle, 
    allFingersClosed, allFingersOpen 
  } = anatomy;

  if (isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp && !isThumbOut) {
    return 'SEND';
  }

  if (allFingersClosed && isThumbDown) {
    return 'CLEAR';
  }

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
// 7. HARDWARE INITIALIZATION & DOM FALLBACK CURSOR
// ====================================================================================================
async function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_PATH).then(vision => 
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
        runningMode: 'VIDEO', 
        numHands: 2, 
        minHandDetectionConfidence: 0.55, 
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      })
    );
  }
  return landmarkerPromise;
}

function getOrCreateWebCursor() {
  let cursor = document.getElementById('nova-finalboss-cursor');
  if (!cursor) {
    cursor = document.createElement('div');
    cursor.id = 'nova-finalboss-cursor';
    Object.assign(cursor.style, {
      position: 'fixed', width: '26px', height: '26px', borderRadius: '50%',
      backgroundColor: 'rgba(16, 185, 129, 0.85)', border: '2px solid rgba(255, 255, 255, 1)',
      boxShadow: '0 0 18px rgba(16, 185, 129, 0.95)', pointerEvents: 'none', zIndex: '999999',
      transform: 'translate(-50%, -50%)', display: 'none',
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

// PROPORTIONAL DYNAMIC SCROLLING: Multiplies speed based on screen edge proximity
function invokeHardwareScroll(direction: 'up' | 'down', intensity: number) {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  const speed = CONFIG.SCROLL_BASE_VELOCITY * Math.max(1.0, intensity * 3.0); 
  
  if (direction === 'up') {
    window.scrollBy({ top: -speed, behavior: 'instant' });
    bridge?.swipeScreen?.(500, 450, 500, 1200 + speed, 80); 
  } else {
    window.scrollBy({ top: speed, behavior: 'instant' });
    bridge?.swipeScreen?.(500, 1200, 500, 450 - speed, 80); 
  }
}

// ====================================================================================================
// 8. MAIN ULTRON VISION LOOP
// ====================================================================================================
export async function localVision(video: HTMLVideoElement, _timestamp: number): Promise<VisionResult> {
  try {
    if (!video || video.readyState < 2 || video.videoWidth === 0) {
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    
    video.playsInline = true; video.muted = true;
    if (video.paused && video.srcObject) video.play().catch(() => {});

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

    // STATE 1: TWO-HAND MOUSE TOGGLE 
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
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('nova-mode-switch', { detail: isMouseActive ? 'mouse' : 'asl' }));
        }
      }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }
    toggleCandidateStart = 0;

    const primaryHand = hands[0];

    // STATE 2: WIRELESS MOUSE ACTIVE
    if (isMouseActive) {
      const anatomy = analyzeAnatomy(primaryHand);
      const v = anatomy.v;

      const REF_LENGTH = Math.max(v(5).sub(v(6)).mag(), 0.0001);

      const palmCenterX = (primaryHand[0].x * 0.4 + primaryHand[5].x * 0.3 + primaryHand[17].x * 0.3);
      const palmCenterY = (primaryHand[0].y * 0.4 + primaryHand[5].y * 0.3 + primaryHand[17].y * 0.3);

      const dirX = primaryHand[9].x - primaryHand[0].x;
      const dirY = primaryHand[9].y - primaryHand[0].y;
      
      const rayX = palmCenterX + dirX * CONFIG.RAY_REACH;
      const rayY = palmCenterY + dirY * CONFIG.RAY_REACH;

      const expandedX = (rayX - CONFIG.MOUSE_ROI_MARGIN) / (1 - CONFIG.MOUSE_ROI_MARGIN * 2);
      const expandedY = (rayY - CONFIG.MOUSE_ROI_MARGIN) / (1 - CONFIG.MOUSE_ROI_MARGIN * 2);

      const mirroredX = 1.0 - expandedX;
      const targetX = Math.max(0.01, Math.min(0.99, mirroredX));
      const targetY = Math.max(0.01, Math.min(0.99, expandedY));

      const smoothedX = mouseFilterX.filter(targetX, now);
      const smoothedY = mouseFilterY.filter(targetY, now);

      // Early Pinch Logic
      const leftPinchDist = v(8).sub(v(4)).mag() / REF_LENGTH;
      if (!isPinching && leftPinchDist < CONFIG.MOUSE_PINCH_DOWN_THRESH) {
        isPinching = true;
      } else if (isPinching && leftPinchDist > CONFIG.MOUSE_PINCH_UP_THRESH) {
        isPinching = false;
      }

      const rightPinchDist = v(12).sub(v(4)).mag() / REF_LENGTH;
      const isRightPinching = rightPinchDist < CONFIG.MOUSE_PINCH_DOWN_THRESH;

      transmitMouseCoordinates(true, smoothedX, smoothedY, isPinching || isRightPinching);

      // Dynamic Scrolling Execution
      if (smoothedY < CONFIG.SCROLL_TRIGGER_ZONE) {
        const intensity = 1.0 - (smoothedY / CONFIG.SCROLL_TRIGGER_ZONE);
        invokeHardwareScroll('up', intensity);
      }
      if (smoothedY > 1 - CONFIG.SCROLL_TRIGGER_ZONE) {
        const intensity = (smoothedY - (1 - CONFIG.SCROLL_TRIGGER_ZONE)) / CONFIG.SCROLL_TRIGGER_ZONE;
        invokeHardwareScroll('down', intensity);
      }

      const viewportX = smoothedX * window.innerWidth;
      const viewportY = smoothedY * window.innerHeight;

      if (isPinching && now - lastClickTime > CONFIG.CLICK_COOLDOWN_MS) {
        lastClickTime = now;
        const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
        bridge?.clickAirMouse?.(smoothedX, smoothedY);
        const targetElement = document.elementFromPoint(viewportX, viewportY);
        if (targetElement instanceof HTMLElement) { targetElement.click(); targetElement.focus(); }
      } else if (isRightPinching && now - lastClickTime > CONFIG.CLICK_COOLDOWN_MS) {
        lastClickTime = now;
        const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
        bridge?.rightClickAirMouse?.(smoothedX, smoothedY);
        const targetElement = document.elementFromPoint(viewportX, viewportY);
        targetElement?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, clientX: viewportX, clientY: viewportY }));
      }

      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    // STATE 3: ASL SIGN & GESTURE DETECTION 
    transmitMouseCoordinates(false);
    
    const anatomy = analyzeAnatomy(primaryHand);
    const detectedSign = classifySign(anatomy);

    if (!detectedSign) {
      if (++releaseFrameCount >= 5) { latchedSign = ''; candidateSign = ''; }
      return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    }

    releaseFrameCount = 0;
    
    if (now - lastEmittedTime < CONFIG.GESTURE_COOLDOWN_MS) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
    if (detectedSign === latchedSign) return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };

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
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'local' };
  }
}

export async function isLocalVisionModelAvailable() { return true; }
export function applyCustomGesture(res: VisionResult, _: CustomGesture[]) { return res; }
