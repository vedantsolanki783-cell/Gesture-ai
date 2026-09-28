import React, { useState, useEffect, useRef } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface Attachment {
  name: string;
  content: string;
}

interface ChatItem {
  role: 'user' | 'assistant';
  text: string;
  files?: string[];
  imageUrl?: string;
}

function dist(a: Point3D, b: Point3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function getFingerStates(lm: Point3D[]) {
  const thumbOpen = dist(lm[4], lm[17]) > dist(lm[3], lm[17]) * 1.15;
  const thumbUp = lm[4].y < lm[3].y && lm[3].y < lm[2].y;
  const indexOpen = lm[8].y < lm[6].y;
  const middleOpen = lm[12].y < lm[10].y;
  const ringOpen = lm[16].y < lm[14].y;
  const pinkyOpen = lm[20].y < lm[18].y;
  const indexHalf = lm[8].y > lm[6].y && lm[6].y < lm[5].y;
  return { thumbOpen, thumbUp, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf };
}

function isOnlyMiddleFinger(lm: Point3D[]): boolean {
  if (!lm || lm.length < 21) return false;
  const { indexOpen, middleOpen, ringOpen, pinkyOpen } = getFingerStates(lm);
  return middleOpen && !indexOpen && !ringOpen && !pinkyOpen;
}

// Complete 26-Letter ASL Classifier (A-Z) — Theme gesture removed
function classifySign26(lm: Point3D[]): string {
  if (!lm || lm.length < 21) return '';
  const { thumbOpen, thumbUp, indexOpen, middleOpen, ringOpen, pinkyOpen, indexHalf } =
    getFingerStates(lm);

  const thumbIndex = dist(lm[4], lm[8]);
  const thumbMiddle = dist(lm[4], lm[12]);
  const indexMiddle = dist(lm[8], lm[12]);
  const horizontal = Math.abs(lm[8].x - lm[5].x) > Math.abs(lm[8].y - lm[5].y);

  if (!indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    if (thumbIndex < 0.06 && thumbMiddle < 0.08) return 'O';
    if (thumbOpen && thumbUp) return 'A';
    if (lm[4].y > lm[8].y && lm[4].y > lm[12].y) return 'E';
    if (lm[4].x > Math.min(lm[6].x, lm[10].x) && lm[4].x < Math.max(lm[6].x, lm[10].x)) return 'T';
    if (lm[4].x > Math.min(lm[10].x, lm[14].x) && lm[4].x < Math.max(lm[10].x, lm[14].x)) return 'N';
    if (lm[4].x > Math.min(lm[14].x, lm[18].x) && lm[4].x < Math.max(lm[14].x, lm[18].x)) return 'M';
    return 'S';
  }

  if (indexOpen && middleOpen && ringOpen && pinkyOpen) {
    return thumbOpen ? 'SPACE' : 'B';
  }

  if (indexHalf && !middleOpen && thumbIndex > 0.08 && thumbIndex < 0.18) return 'C';

  if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) return lm[8].y > lm[0].y ? 'Q' : 'G';
    if (thumbOpen) return 'L';
    if (thumbMiddle < 0.07) return 'D';
    return 'Z';
  }

  if (indexHalf && !middleOpen && !ringOpen && !pinkyOpen) return 'X';
  if (!indexOpen && middleOpen && ringOpen && pinkyOpen && thumbIndex < 0.08) return 'F';

  if (!indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    if (thumbOpen) return 'Y';
    return horizontal ? 'J' : 'I';
  }

  if (indexOpen && !middleOpen && !ringOpen && pinkyOpen) return 'SEND';

  if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
    if (horizontal) return lm[8].y > lm[0].y ? 'P' : 'H';
    if (lm[8].x > lm[12].x && lm[5].x < lm[9].x) return 'R';
    if (thumbOpen && thumbMiddle < 0.08) return 'K';
    if (indexMiddle > 0.06) return 'V';
    return 'U';
  }

  if (indexOpen && middleOpen && ringOpen && !pinkyOpen) return 'W';

  return '';
}

export function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [currentSign, setCurrentSign] = useState<string>('None');
  const [controlMode, setControlMode] = useState<boolean>(false);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number }>({ x: 200, y: 200 });
  const [statusText, setStatusText] = useState<string>('Initializing camera & 26-sign engine...');

  const [apiKey, setApiKey] = useState<string>(() => localStorage.getItem('nova_api_key') || '');
  const [selectedModel, setSelectedModel] = useState<string>('deepseek/deepseek-r1:free');
  const [showSettings, setShowSettings] = useState<boolean>(false);

  const [input, setInput] = useState<string>('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<ChatItem[]>([
    {
      role: 'assistant',
      text: 'NOVA AGI Ready! Use A–Z hand signs, show both middle fingers to toggle Mouseless Control Mode, attach files with 📎, or type "draw a futuristic city".',
    },
  ]);

  const controlModeRef = useRef(controlMode);
  controlModeRef.current = controlMode;

  useEffect(() => {
    let landmarker: HandLandmarker | null = null;
    let animationFrameId: number;
    let lastToggle = 0;
    let lastClick = 0;
    let lastLetter = '';
    let letterHoldStart = 0;

    async function setupCameraAndVision() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: 640, height: 480 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
        });

        setStatusText('Ready: Sign A–Z or hold 2 middle fingers for Mouseless Mode');

        const detectLoop = () => {
          if (videoRef.current && landmarker && videoRef.current.readyState >= 2) {
            const results = landmarker.detectForVideo(videoRef.current, performance.now());
            const hands = results.landmarks as Point3D[][];
            const now = Date.now();

            if (hands && hands.length > 0) {
              // 1. Check 2-hand middle finger gesture
              if (hands.length >= 2 && isOnlyMiddleFinger(hands[0]) && isOnlyMiddleFinger(hands[1])) {
                if (now - lastToggle > 1500) {
                  lastToggle = now;
                  const nextMode = !controlModeRef.current;
                  setControlMode(nextMode);
                  setStatusText(
                    nextMode
                      ? 'Mouseless Mode ON: Point index to move/scroll, pinch to click'
                      : 'Sign Language Mode ON (A–Z)'
                  );
                }
              } else if (controlModeRef.current) {
                // 2. Mouseless Control Mode
                const indexTip = hands[0][8];
                const thumbTip = hands[0][4];
                const cx = (1 - indexTip.x) * window.innerWidth;
                const cy = indexTip.y * window.innerHeight;
                setCursorPos({ x: cx, y: cy });

                if (indexTip.y < 0.2) window.scrollBy({ top: -15, behavior: 'auto' });
                if (indexTip.y > 0.8) window.scrollBy({ top: 15, behavior: 'auto' });

                if (dist(indexTip, thumbTip) < 0.05 && now - lastClick > 800) {
                  lastClick = now;
                  const el = document.elementFromPoint(cx, cy) as HTMLElement | null;
                  if (el) el.click();
                }
              } else {
                // 3. 26-Letter Sign Mode
                const sign = classifySign26(hands[0]);
                setCurrentSign(sign || 'None');

                if (sign && sign === lastLetter && now - letterHoldStart > 950) {
                  letterHoldStart = now;
                  if (sign === 'SPACE') {
                    setInput((prev) => prev + ' ');
                  } else if (sign === 'SEND') {
                    document.getElementById('nova-send-btn')?.click();
                  } else {
                    setInput((prev) => prev + sign);
                  }
                } else if (sign !== lastLetter) {
                  lastLetter = sign;
                  letterHoldStart = now;
                }
              }
            } else {
              setCurrentSign('None');
            }
          }
          animationFrameId = requestAnimationFrame(detectLoop);
        };

        detectLoop();
      } catch (err: any) {
        setStatusText(`Camera error: ${err.message || 'Please allow camera access'}`);
      }
    }

    setupCameraAndVision();
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  const handleSaveKey = (val: string) => {
    setApiKey(val);
    localStorage.setItem('nova_api_key', val);
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setAttachments((prev) => [...prev, { name: file.name, content: text.slice(0, 15000) }]);
  };

  const launchExternalApp = (app: string) => {
    if (app === 'whatsapp') window.location.href = 'whatsapp://send?text=Hello';
    if (app === 'youtube') window.open('https://m.youtube.com', '_blank');
    if (app === 'instagram') window.location.href = 'instagram://app';
    if (app === 'google') window.open('https://www.google.com', '_blank');
  };

  const handleSend = async () => {
    if (!input.trim() && attachments.length === 0) return;

    const userText = input.trim();
    const currentFiles = [...attachments];
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: userText || '(Attached File)', files: currentFiles.map((f) => f.name) },
    ]);
    setInput('');
    setAttachments([]);

    // Quick App Launch Commands
    const upper = userText.toUpperCase();
    if (upper === 'WHATSAPP' || upper.includes('OPEN WHATSAPP')) {
      launchExternalApp('whatsapp');
      return;
    }
    if (upper === 'YOUTUBE' || upper.includes('OPEN YOUTUBE')) {
      launchExternalApp('youtube');
      return;
    }

    // Free Image Generation Check
    if (/^(draw|generate image|create image|paint)\b/i.test(userText)) {
      const promptClean = userText.replace(/^(draw|generate image of|generate image|create image|paint)/i, '').trim();
      const imgUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(promptClean || userText)}?width=768&height=768&nologo=true`;
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Generated image for: "${promptClean || userText}"`, imageUrl: imgUrl },
      ]);
      return;
    }

    setLoading(true);
    try {
      const fileContext =
        currentFiles.length > 0
          ? '\n\nAttached Files:\n' + currentFiles.map((f) => `[${f.name}]:\n${f.content}`).join('\n\n')
          : '';

      if (!navigator.onLine || !apiKey) {
        // Offline / No-Key Fallback
        if ('speechSynthesis' in window && userText) {
          window.speechSynthesis.speak(new SpeechSynthesisUtterance(userText));
        }
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            text: !apiKey
              ? `Spoke "${userText}" aloud! To unlock full DeepSeek-R1 & Qwen cloud thinking, tap ⚙️ Settings at the top right and paste your free OpenRouter key.`
              : `[Offline Mode]: Spoke "${userText}" aloud. File characters loaded: ${fileContext.length}.`,
          },
        ]);
        setLoading(false);
        return;
      }

      const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey.trim()}`,
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: [
            {
              role: 'system',
              content:
                'You are NOVA AGI, an advanced assistant supporting sign language users, deep reasoning, code analysis, and file inspection.',
            },
            { role: 'user', content: userText + fileContext },
          ],
        }),
      });

      const data = await res.json();
      const reply =
        data.choices?.[0]?.message?.content ||
        data.error?.message ||
        'Could not get response. Check your API key in Settings.';

      setMessages((prev) => [...prev, { role: 'assistant', text: reply }]);
    } catch (err: any) {
      setMessages((prev) => [...prev, { role: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#f8fafc', fontFamily: 'sans-serif', paddingBottom: '30px' }}>
      {/* Floating Virtual Cursor when Mouseless Control Mode is ON */}
      {controlMode && (
        <div
          style={{
            position: 'fixed',
            left: cursorPos.x - 12,
            top: cursorPos.y - 12,
            width: 24,
            height: 24,
            borderRadius: '50%',
            background: 'rgba(239, 68, 68, 0.85)',
            border: '3px solid #ffffff',
            pointerEvents: 'none',
            zIndex: 9999,
            boxShadow: '0 0 12px #ef4444',
          }}
        />
      )}

      {/* Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', background: '#1e293b', borderBottom: '1px solid #334155' }}>
        <div>
          <strong style={{ fontSize: '18px', color: '#38bdf8' }}>NOVA AGI</strong>
          <span style={{ marginLeft: '10px', fontSize: '12px', padding: '3px 8px', borderRadius: '12px', background: controlMode ? '#dc2626' : '#0284c7' }}>
            {controlMode ? '🖱️ Mouseless Mode' : `✋ Sign: ${currentSign}`}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setControlMode((prev) => !prev)}
            style={{ background: '#334155', color: '#fff', border: 'none', padding: '6px 10px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' }}
          >
            {controlMode ? 'Switch to Sign Mode' : 'Switch to Mouse Mode'}
          </button>
          <button
            onClick={() => setShowSettings((prev) => !prev)}
            style={{ background: '#2563eb', color: '#fff', border: 'none', padding: '6px 10px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' }}
          >
            ⚙️ Settings
          </button>
        </div>
      </div>

      {/* Settings Drawer */}
      {showSettings && (
        <div style={{ padding: '14px', background: '#1e293b', borderBottom: '1px solid #475569' }}>
          <div style={{ marginBottom: '8px', fontSize: '13px' }}>
            <strong>OpenRouter Free API Key:</strong> (Saved on your device only)
          </div>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => handleSaveKey(e.target.value)}
            placeholder="Paste sk-or-v1-... key here"
            style={{ width: '100%', padding: '8px', borderRadius: '6px', background: '#0f172a', color: '#fff', border: '1px solid #475569', marginBottom: '10px' }}
          />
          <div style={{ fontSize: '13px', marginBottom: '4px' }}>
            <strong>AI Brain Model:</strong>
          </div>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ width: '100%', padding: '8px', borderRadius: '6px', background: '#0f172a', color: '#fff', border: '1px solid #475569' }}
          >
            <option value="deepseek/deepseek-r1:free">DeepSeek-R1 (Deep Reasoning - Free)</option>
            <option value="qwen/qwen-2.5-72b-instruct:free">Qwen 2.5 72B (Coding & General - Free)</option>
            <option value="meta-llama/llama-3.3-70b-instruct:free">Llama 3.3 70B (Fast - Free)</option>
            <option value="google/gemini-2.0-flash-exp:free">Google Gemini 2.0 Flash (Free)</option>
          </select>
        </div>
      )}

      {/* Camera Feed + Quick App Launcher Bar */}
      <div style={{ padding: '12px', maxWidth: '680px', margin: '0 auto' }}>
        <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', background: '#000', maxHeight: '220px', display: 'flex', justifyContent: 'center' }}>
          <video ref={videoRef} playsInline muted style={{ height: '220px', transform: 'scaleX(-1)', objectFit: 'cover' }} />
          <div style={{ position: 'absolute', bottom: 8, left: 8, right: 8, background: 'rgba(15, 23, 42, 0.8)', padding: '6px 10px', borderRadius: '8px', fontSize: '12px' }}>
            {statusText}
          </div>
        </div>

        {/* Hands-Free App Launcher */}
        <div style={{ display: 'flex', gap: '8px', marginTop: '10px', flexWrap: 'wrap' }}>
          <button onClick={() => launchExternalApp('whatsapp')} style={{ flex: 1, background: '#16a34a', color: '#fff', border: 'none', padding: '8px', borderRadius: '8px', fontSize: '12px', fontWeight: 'bold' }}>
            WhatsApp
          </button>
          <button onClick={() => launchExternalApp('youtube')} style={{ flex: 1, background: '#dc2626', color: '#fff', border: 'none', padding: '8px', borderRadius: '8px', fontSize: '12px', fontWeight: 'bold' }}>
            YouTube
          </button>
          <button onClick={() => launchExternalApp('instagram')} style={{ flex: 1, background: '#db2777', color: '#fff', border: 'none', padding: '8px', borderRadius: '8px', fontSize: '12px', fontWeight: 'bold' }}>
            Instagram
          </button>
          <button onClick={() => launchExternalApp('google')} style={{ flex: 1, background: '#2563eb', color: '#fff', border: 'none', padding: '8px', borderRadius: '8px', fontSize: '12px', fontWeight: 'bold' }}>
            Google
          </button>
        </div>

        {/* Chat Messages */}
        <div style={{ marginTop: '14px', background: '#1e293b', borderRadius: '12px', padding: '12px', minHeight: '240px', maxHeight: '380px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                background: m.role === 'user' ? '#2563eb' : '#0f172a',
                padding: '10px 14px',
                borderRadius: '10px',
                maxWidth: '85%',
                fontSize: '14px',
                lineHeight: 1.4,
              }}
            >
              {m.files && m.files.map((fname, idx) => (
                <div key={idx} style={{ fontSize: '11px', color: '#93c5fd', marginBottom: '4px' }}>
                  📎 {fname}
                </div>
              ))}
              <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
              {m.imageUrl && (
                <img src={m.imageUrl} alt="Generated" style={{ width: '100%', borderRadius: '8px', marginTop: '8px' }} />
              )}
            </div>
          ))}
          {loading && <div style={{ fontSize: '13px', color: '#94a3b8' }}>🧠 Thinking deeply...</div>}
        </div>

        {/* Attached file badges */}
        {attachments.length > 0 && (
          <div style={{ display: 'flex', gap: '6px', marginTop: '8px', flexWrap: 'wrap' }}>
            {attachments.map((a, i) => (
              <span key={i} style={{ fontSize: '12px', background: '#334155', padding: '4px 8px', borderRadius: '6px' }}>
                📎 {a.name}
              </span>
            ))}
          </div>
        )}

        {/* Chat Input Bar with 📎 File Upload */}
        <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{ background: '#334155', color: '#fff', border: 'none', borderRadius: '8px', padding: '0 14px', fontSize: '18px', cursor: 'pointer' }}
            title="Attach File"
          >
            📎
          </button>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Sign A-Z, ask anything, or type 'draw a dragon'..."
            style={{ flex: 1, padding: '10px 12px', borderRadius: '8px', background: '#1e293b', color: '#fff', border: '1px solid #475569' }}
          />
          <button
            id="nova-send-btn"
            onClick={handleSend}
            style={{ background: '#2563eb', color: '#fff', border: 'none', borderRadius: '8px', padding: '0 16px', fontWeight: 'bold', cursor: 'pointer' }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
