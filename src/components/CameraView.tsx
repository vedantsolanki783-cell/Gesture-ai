import React from 'react';
import { Camera, CameraOff, Cpu, Cloud, AlertTriangle, WifiOff } from 'lucide-react';
import type { AppSettings, VisionResult } from '../types';

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  enabled: boolean;
  status: string;
  lastDetection: string | null;
  onToggle: () => void;
  onDetected?: (result: VisionResult) => void;
  settings: AppSettings;
}

export function CameraView({ videoRef, enabled, status, lastDetection, onToggle }: Props) {
  const local = status === 'local';
  return <div className="camera-card">
    {!enabled ? <div className="camera-empty"><CameraOff size={46}/><b>Vision is offline</b><button onClick={onToggle}>Enable camera</button></div> : <>
      <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
      <div className="camera-top"><span className="status-pill">{local ? <Cpu size={13}/> : status === 'cloud' ? <Cloud size={13}/> : <AlertTriangle size={13}/>} {local ? 'LOCAL VISION' : status === 'cloud' ? 'CLOUD VISION' : status === 'model-missing' ? 'MODEL MISSING' : status.toUpperCase()}</span><button onClick={onToggle}><CameraOff size={18}/></button></div>
      {lastDetection && <div className="detection"><span>GESTURE</span><strong>{lastDetection.replaceAll('_',' ')}</strong></div>}
      <div className="camera-bottom">{local ? 'Offline-ready • hand model active' : status === 'model-missing' ? 'Add public/models/hand_landmarker.task for offline vision' : status === 'cloud' ? 'Using optional Gemini vision fallback' : 'Starting vision…'}</div>
    </>}
  </div>;
}
