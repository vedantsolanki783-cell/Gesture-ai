import React, { useEffect, useRef, useState } from 'react';

interface Props {
  size?: number;
  state?: 'idle' | 'listening' | 'thinking';
  className?: string;
}

export function NovaFace({ size = 140, state = 'idle', className = '' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [blink, setBlink] = useState(false);
  const [pupilOffset, setPupilOffset] = useState({ x: 0, y: 0 });

  // Autonomous Random Blinking
  useEffect(() => {
    let timeout: number;
    const triggerBlink = () => {
      setBlink(true);
      setTimeout(() => setBlink(false), 150); // Blink shut for 150ms
      timeout = window.setTimeout(triggerBlink, Math.random() * 4000 + 2000); // Wait 2-6s
    };
    timeout = window.setTimeout(triggerBlink, 2000);
    return () => clearTimeout(timeout);
  }, []);

  // 3D Look-At Logic (Click anywhere on screen)
  useEffect(() => {
    let resetTimeout: number;

    const handleGlobalClick = (e: MouseEvent) => {
      if (!containerRef.current) return;
      
      const rect = containerRef.current.getBoundingClientRect();
      const faceCenterX = rect.left + rect.width / 2;
      const faceCenterY = rect.top + rect.height / 2;

      const deltaX = e.clientX - faceCenterX;
      const deltaY = e.clientY - faceCenterY;

      // Calculate tilt limits (max 25 degrees)
      const maxTilt = 25;
      const screenW = window.innerWidth / 2;
      const screenH = window.innerHeight / 2;

      let tiltY = (deltaX / screenW) * maxTilt;
      let tiltX = -(deltaY / screenH) * maxTilt;

      // Clamp tilts so the head doesn't snap backwards
      tiltY = Math.max(-maxTilt, Math.min(maxTilt, tiltY));
      tiltX = Math.max(-maxTilt, Math.min(maxTilt, tiltX));

      // Pupil offset mapping (shifts the eyes slightly toward the click)
      const pX = (tiltY / maxTilt) * 4; 
      const pY = -(tiltX / maxTilt) * 4;

      // Apply CSS variables to rotate the sphere in 3D
      containerRef.current.style.setProperty('--nf-tiltx', `${tiltX}deg`);
      containerRef.current.style.setProperty('--nf-tilty', `${tiltY}deg`);
      
      setPupilOffset({ x: pX, y: pY });
      clearTimeout(resetTimeout);

      // Reset back to center after 2.5 seconds
      resetTimeout = window.setTimeout(() => {
        if (containerRef.current) {
          containerRef.current.style.setProperty('--nf-tiltx', `0deg`);
          containerRef.current.style.setProperty('--nf-tilty', `0deg`);
        }
        setPupilOffset({ x: 0, y: 0 });
      }, 2500);
    };

    window.addEventListener('click', handleGlobalClick);
    return () => {
      window.removeEventListener('click', handleGlobalClick);
      clearTimeout(resetTimeout);
    };
  }, []);

  return (
    <div 
      ref={containerRef} 
      className={`nova-face nova-face-${state} ${className}`} 
      style={{ width: size, height: size }}
    >
      <div className="nova-face-sphere">
        <div className="nova-face-shine" />
        <div className="nova-face-rim" />
        {state === 'listening' && <div className="nova-face-pulse" />}
        <div className="nova-face-eyes">
          {/* LEFT EYE */}
          <div className="nova-eye" style={{ transform: blink ? 'scaleY(0.1)' : 'scaleY(1)' }}>
            <div 
              className="nova-eye-pupil" 
              style={{ transform: `translate(${pupilOffset.x}px, ${pupilOffset.y}px)` }} 
            />
          </div>
          {/* RIGHT EYE */}
          <div className="nova-eye" style={{ transform: blink ? 'scaleY(0.1)' : 'scaleY(1)' }}>
            <div 
              className="nova-eye-pupil" 
              style={{ transform: `translate(${pupilOffset.x}px, ${pupilOffset.y}px)` }} 
            />
          </div>
        </div>
      </div>
    </div>
  );
}
