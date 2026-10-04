import React, { useEffect, useRef, useState } from 'react';

interface Props {
  size?: number;
  state?: 'idle' | 'listening' | 'thinking';
  className?: string;
}

export function NovaFace({ size = 140, state = 'idle', className = '' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [blink, setBlink] = useState(false);

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

      // Calculate tilt limits (max 35 degrees for deep 3D effect)
      const maxTilt = 35;
      const screenW = window.innerWidth / 2;
      const screenH = window.innerHeight / 2;

      let tiltY = (deltaX / screenW) * maxTilt;
      let tiltX = -(deltaY / screenH) * maxTilt;

      // Clamp tilts
      tiltY = Math.max(-maxTilt, Math.min(maxTilt, tiltY));
      tiltX = Math.max(-maxTilt, Math.min(maxTilt, tiltX));

      // Target the inner 3D head
      const head = containerRef.current.querySelector('.nova-exact-head') as HTMLElement;
      if (head) {
        head.style.setProperty('--nf-tiltx', `${tiltX}deg`);
        head.style.setProperty('--nf-tilty', `${tiltY}deg`);
      }
      
      clearTimeout(resetTimeout);

      // Reset back to center after 2.5 seconds
      resetTimeout = window.setTimeout(() => {
        if (head) {
          head.style.setProperty('--nf-tiltx', `0deg`);
          head.style.setProperty('--nf-tilty', `0deg`);
        }
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
      className={`nova-exact-wrapper nova-face-${state} ${className}`} 
      style={{ width: size, height: size }}
    >
      {/* Sweeping Energy Aura */}
      <div className="nova-aura-ring ring-1" />
      <div className="nova-aura-ring ring-2" />
      <div className="nova-aura-ring ring-3" />
      
      {/* 3D Glass Head */}
      <div className="nova-exact-head">
        <div className="nova-exact-highlight" />
        <div className="nova-exact-eyes">
          <div className="nova-exact-eye" style={{ transform: blink ? 'scaleY(0)' : 'scaleY(1)' }} />
          <div className="nova-exact-eye" style={{ transform: blink ? 'scaleY(0)' : 'scaleY(1)' }} />
        </div>
      </div>
    </div>
  );
}
