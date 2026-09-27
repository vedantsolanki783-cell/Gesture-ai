import { useCallback, useEffect, useRef, useState } from 'react';
import type { AppSettings, CustomGesture, VisionResult } from '../types';
import { analyzeWithGemini } from '../services/aiRouter';
import { applyCustomGesture, isLocalVisionModelAvailable, localVision } from '../vision/localVision';

function parseCloud(text: string): VisionResult {
  try {
    const clean = text.replace(/```json|```/g, '').trim();
    const data = JSON.parse(clean);
    return { type: data.type || 'UNKNOWN', value: data.value || '', confidence: Number(data.confidence ?? 0.7), source: 'gemini' };
  } catch {
    return { type: 'UNKNOWN', value: '', confidence: 0, source: 'gemini' };
  }
}

export function useVision(settings: AppSettings, customGestures: CustomGesture[], onDetected: (result: VisionResult) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processingRef = useRef(false);
  const lastValueRef = useRef('');
  const lastTimeRef = useRef(0);
  const [status, setStatus] = useState<'off' | 'starting' | 'local' | 'cloud' | 'model-missing' | 'error'>('off');
  const [lastDetection, setLastDetection] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus('off');
  }, []);

  useEffect(() => {
    if (!settings.visionEnabled) { stop(); return; }
    let cancelled = false;
    const start = async () => {
      setStatus('starting');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 }, audio: false });
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
        const model = await isLocalVisionModelAvailable();
        setStatus(model ? 'local' : 'model-missing');
      } catch (error) {
        console.error(error);
        setStatus('error');
      }
    };
    start();
    return () => { cancelled = true; stop(); };
  }, [settings.visionEnabled, stop]);

  useEffect(() => {
    if (!settings.visionEnabled) return;
    let timer = 0;
    const tick = async () => {
      if (processingRef.current || !videoRef.current || videoRef.current.readyState < 2) return;
      processingRef.current = true;
      try {
        let result: VisionResult;
        if (status === 'local') {
          result = await localVision(videoRef.current, performance.now());
        } else if (status === 'model-missing' && settings.aiProvider !== 'ollama') {
          const canvas = canvasRef.current || document.createElement('canvas');
          canvas.width = 320; canvas.height = 240;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          ctx.drawImage(videoRef.current, 0, 0, 320, 240);
          const raw = await analyzeWithGemini(canvas.toDataURL('image/jpeg', 0.65), settings, customGestures);
          result = parseCloud(raw);
          setStatus('cloud');
        } else return;
        result = applyCustomGesture(result, customGestures);
        const now = Date.now();
        if (result.type !== 'UNKNOWN' && result.type !== 'ERROR' && result.confidence >= settings.confidenceThreshold && (result.value !== lastValueRef.current || now - lastTimeRef.current > 1800)) {
          lastValueRef.current = result.value;
          lastTimeRef.current = now;
          setLastDetection(result.value);
          onDetected(result);
          window.setTimeout(() => setLastDetection(null), 900);
        }
      } catch (error) {
        console.error('Vision loop:', error);
      } finally {
        processingRef.current = false;
      }
    };
    timer = window.setInterval(tick, status === 'local' ? 350 : 2200);
    return () => window.clearInterval(timer);
  }, [settings, customGestures, onDetected, status]);

  return { videoRef, canvasRef, lastDetection, status, stop };
}
