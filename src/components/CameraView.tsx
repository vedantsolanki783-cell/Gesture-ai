import React, { useEffect, useRef, useState } from 'react';
import { CameraOff, Cpu, Cloud, AlertTriangle, MousePointer2, Hand } from 'lucide-react';
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
  const [toast, setToast] = useState<'mouse' | 'asl' | null>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const onModeSwitch = (e: Event) => {
      const mode = (e as CustomEvent).detail as 'mouse' | 'asl';
      setToast(mode);
      if (hideTimer.current) clearTimeout(hideTimer.current);
      hideTimer.current = setTimeout(() => setToast(null), 3000);
    };
    window.addEventListener('nova-mode-switch', onModeSwitch);
    return () => {
      window.removeEventListener('nova-mode-switch', onModeSwitch);
      if (hideTimer.current) clearTimeout(hideTimer.current);
    };
  }, []);

  return <div className="camera-card">
    {toast && <div className="mode-toast">
      {toast === 'mouse' ? <MousePointer2 size={14}/> : <Hand size={14}/>}
      Switched to {toast === 'mouse' ? 'Mouse' : 'ASL'} Mode
    </div>}
    {!enabled ? <div className="camera-empty"><CameraOff size={40}/><b>Vision is offline</b><button onClick={onToggle}>Enable camera</button></div> : <>
      <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
      <div className="camera-top"><span className="status-pill">{local ? <Cpu size={13}/> : status === 'cloud' ? <Cloud size={13}/> : <AlertTriangle size={13}/>} {local ? 'LOCAL VISION' : status === 'cloud' ? 'CLOUD VISION' : status.toUpperCase()}</span><button onClick={onToggle}><CameraOff size={16}/></button></div>
      {lastDetection && <div className="detection"><span>GESTURE</span><strong>{lastDetection.replaceAll('_',' ')}</strong></div>}
      <div className="camera-bottom">{local ? 'Offline-ready • hand model active' : status === 'cloud' ? 'Using optional cloud vision fallback' : 'Starting vision…'}</div>
    </>}
  </div>;
}
