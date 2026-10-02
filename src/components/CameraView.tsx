import React, { useEffect, useState } from 'react';
import { CameraOff, Cpu, Cloud, AlertTriangle, MousePointer2 } from 'lucide-react';
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
  const [mouseless, setMouseless] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number; pinching: boolean } | null>(null);

  useEffect(() => {
    const onToggleEvt = (e: Event) => setMouseless(!!(e as CustomEvent).detail);
    const onCursorEvt = (e: Event) => setCursor((e as CustomEvent).detail);
    window.addEventListener('nova-mouseless-toggle', onToggleEvt);
    window.addEventListener('nova-mouseless-cursor', onCursorEvt);
    return () => {
      window.removeEventListener('nova-mouseless-toggle', onToggleEvt);
      window.removeEventListener('nova-mouseless-cursor', onCursorEvt);
    };
  }, []);

  return <div className="camera-card">
    {mouseless && <div className="mouseless-badge"><MousePointer2 size={14}/> MOUSELESS MODE ON</div>}
    {mouseless && cursor && <div className="mouseless-cursor" style={{ left: cursor.x - 12, top: cursor.y - 12, background: cursor.pinching ? 'rgba(46,204,113,0.6)' : 'rgba(46,204,113,0.15)' }} />}
    {!enabled ? <div className="camera-empty"><CameraOff size={40}/><b>Vision is offline</b><button onClick={onToggle}>Enable camera</button></div> : <>
      <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
      <div className="camera-top"><span className="status-pill">{local ? <Cpu size={13}/> : status === 'cloud' ? <Cloud size={13}/> : <AlertTriangle size={13}/>} {local ? 'LOCAL VISION' : status === 'cloud' ? 'CLOUD VISION' : status.toUpperCase()}</span><button onClick={onToggle}><CameraOff size={16}/></button></div>
      {lastDetection && <div className="detection"><span>GESTURE</span><strong>{lastDetection.replaceAll('_',' ')}</strong></div>}
      <div className="camera-bottom">{local ? 'Offline-ready • hand model active' : status === 'cloud' ? 'Using optional cloud vision fallback' : 'Starting vision…'}</div>
    </>}
  </div>;
}
