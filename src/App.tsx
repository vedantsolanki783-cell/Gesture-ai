import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Bot, Copy, Mic, MicOff, Moon, Paperclip, Send, Settings, Sparkles, Sun, Trash2, User, WifiOff, X } from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

interface LocalAttachment {
  name: string;
  content: string;
  isImage?: boolean;
  dataUrl?: string;
  mimeType?: string;
  localVisualReport?: string;
}

const defaults: AppSettings = {
  theme: 'dark',
  voiceEnabled: true,
  visionEnabled: false,
  confidenceThreshold: 0.72,
  aiProvider: 'cloudFree',
  ollamaUrl: 'http://localhost:11434',
  ollamaModel: 'qwen3:4b',
  geminiModel: 'gemini-2.5-flash',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction: 'You are NOVA, a multimodal AI assistant capable of analyzing images, reading files, writing code, and assisting sign language users clearly and directly.',
  customGestures: []
};

const LETTER_COOLDOWN_MS = 1000;
const SAME_LETTER_COOLDOWN_MS = 1800;

// Blocks any model response that claims it cannot view images
function isVisionRefusal(text: string): boolean {
  if (!text) return true;
  return /can't view images|cannot view images|cannot analyze or interpret image|unable to view images|unable to see images|can't see images|cannot see the image|i am a text-based|describe what's in the picture|describe what’s in the picture|do not have inherent image/i.test(
    text
  );
}

// Local MediaPipe Image Landmarker to inspect hands/gestures inside uploaded photos
let imageLandmarkerPromise: Promise<HandLandmarker> | null = null;
async function getImageLandmarker(): Promise<HandLandmarker | null> {
  try {
    if (!imageLandmarkerPromise) {
      imageLandmarkerPromise = FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      ).then(vision =>
        HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'IMAGE',
          numHands: 2,
          minHandDetectionConfidence: 0.35
        })
      );
    }
    return await imageLandmarkerPromise;
  } catch {
    imageLandmarkerPromise = null;
    return null;
  }
}

// Analyzes dominant colors, lighting, and hand landmarks directly from the image pixels
async function inspectImageLocally(img: HTMLImageElement, canvas: HTMLCanvasElement, fileName: string): Promise<string> {
  const details: string[] = [
    `Image File: "${fileName}" (${img.naturalWidth || canvas.width}x${img.naturalHeight || canvas.height}px)`
  ];

  // 1. Pixel lighting & color analysis
  try {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const { width, height } = canvas;
      const data = ctx.getImageData(0, 0, width, height).data;
      let rSum = 0, gSum = 0, bSum = 0, brightSum = 0, count = 0;
      for (let i = 0; i < data.length; i += 16) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        rSum += r; gSum += g; bSum += b;
        brightSum += (r + g + b) / 3;
        count++;
      }
      if (count > 0) {
        const avgR = Math.round(rSum / count);
        const avgG = Math.round(gSum / count);
        const avgB = Math.round(bSum / count);
        const avgBright = Math.round(brightSum / count);
        const tone = avgBright > 200 ? 'Bright / Light background' : avgBright < 65 ? 'Dark / Low-key background' : 'Balanced lighting';
        details.push(`Visual Lighting: ${tone} (Average RGB: ${avgR}, ${avgG}, ${avgB})`);
      }
    }
  } catch {}

  // 2. Local MediaPipe Hand & Finger detection on the static image
  try {
    const landmarker = await getImageLandmarker();
    if (landmarker) {
      const res = landmarker.detect(img);
      const hands = res.landmarks || [];
      if (hands.length > 0) {
        const handDescriptions = hands.map((lm, idx) => {
          const d = (a: any, b: any) => Math.hypot(a.x - b.x, a.y - b.y);
          const wrist = lm[0];
          const indexOpen = lm[8].y < lm[6].y && d(lm[8], wrist) > d(lm[6], wrist);
          const middleOpen = lm[12].y < lm[10].y && d(lm[12], wrist) > d(lm[10], wrist);
          const ringOpen = lm[16].y < lm[14].y && d(lm[16], wrist) > d(lm[14], wrist);
          const pinkyOpen = lm[20].y < lm[18].y && d(lm[20], wrist) > d(lm[18], wrist);
          const thumbOpen = d(lm[4], lm[17]) > d(lm[3], lm[17]) * 1.1;

          const openList: string[] = [];
          if (thumbOpen) openList.push('thumb');
          if (indexOpen) openList.push('index');
          if (middleOpen) openList.push('middle');
          if (ringOpen) openList.push('ring');
          if (pinkyOpen) openList.push('pinky');

          let gestureHint = 'custom posture / fist';
          if (middleOpen && !indexOpen && !ringOpen && !pinkyOpen) gestureHint = 'Middle finger extended upward';
          else if (indexOpen && pinkyOpen && !middleOpen && !ringOpen) gestureHint = 'Yo-Yo / Rock-on sign (SEND gesture)';
          else if (indexOpen && middleOpen && ringOpen && pinkyOpen) gestureHint = 'Open palm / ASL Letter B';
          else if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) gestureHint = 'Two fingers up (ASL V / U / R)';
          else if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) gestureHint = thumbOpen ? 'ASL Letter L' : 'ASL Letter D / pointing index';
          else if (!indexOpen && !middleOpen && !ringOpen && pinkyOpen) gestureHint = thumbOpen ? 'ASL Letter Y' : 'ASL Letter I / J';
          else if (!indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureHint = lm[4].y > lm[0].y + 0.05 ? 'Thumbs Down (CLEAR gesture)' : 'Closed fist / ASL A, S, E, O, or T';
          }

          return `Hand #${idx + 1}: Extended fingers = [${openList.length ? openList.join(', ') : 'none (closed)'}], Detected posture = ${gestureHint}`;
        });
        details.push(`Hands Detected in Image (${hands.length}):\n` + handDescriptions.join('\n'));
      }
    }
  } catch {}

  // 3. Built-in Browser OCR TextDetector (if supported on device)
  try {
    const AnyWin = window as any;
    if ('TextDetector' in AnyWin) {
      const detector = new AnyWin.TextDetector();
      const texts = await detector.detect(img);
      if (texts && texts.length > 0) {
        const extracted = texts.map((t: any) => t.rawValue).filter(Boolean).join(' | ');
        if (extracted) details.push(`Visible Text (OCR): ${extracted.slice(0, 2000)}`);
      }
    }
  } catch {}

  return details.join('\n');
}

// Converts an image to a compact 512px Base64 JPEG + runs local pixel & hand inspection
function processImageFile(file: File): Promise<LocalAttachment> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onerror = () =>
      resolve({ name: file.name, isImage: true, content: `[Image: ${file.name}]` });
    reader.onload = () => {
      const img = new Image();
      img.onerror = () =>
        resolve({ name: file.name, isImage: true, content: `[Image: ${file.name}]` });
      img.onload = async () => {
        const maxDim = 512;
        let { width, height } = img;
        if (width > maxDim || height > maxDim) {
          const scale = maxDim / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.drawImage(img, 0, 0, width, height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.74);
        const localVisualReport = await inspectImageLocally(img, canvas, file.name);

        resolve({
          name: file.name,
          isImage: true,
          dataUrl,
          mimeType: 'image/jpeg',
          localVisualReport,
          content: localVisualReport
        });
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  });
}

async function readPickedFile(file: File): Promise<LocalAttachment> {
  if (file.type.startsWith('image/')) {
    return processImageFile(file);
  }
  try {
    const rawText = await file.text();
    const clipped = rawText.slice(0, 25000);
    return {
      name: file.name,
      content: `--- FILE CONTENT (${file.name}) ---\n${clipped}\n--- END OF FILE ---`
    };
  } catch {
    return {
      name: file.name,
      content: `[Attached File: ${file.name} (${Math.round(file.size / 1024)}KB)]`
    };
  }
}

// Multi-Model Vision Pipeline with Anti-Refusal Verification
async function analyzeImagesWithVisionAI(
  promptText: string,
  images: LocalAttachment[],
  settings: AppSettings
): Promise<{ text: string; provider: string } | null> {
  const validImages = images.filter(img => !!img.dataUrl);
  if (validImages.length === 0 || !navigator.onLine) return null;

  const localHints = validImages
    .map(img => img.localVisualReport)
    .filter(Boolean)
    .join('\n\n');

  const userQuestion =
    promptText.trim() ||
    'Describe what is shown in this image clearly, including any people, hand gestures/signs, objects, text, or UI elements.';

  const anySettings = settings as any;

  // 1. Try Google Gemini Vision API if a Gemini key is available
  const geminiKey =
    anySettings.geminiApiKey ||
    anySettings.geminiKey ||
    (import.meta as any).env?.VITE_GEMINI_API_KEY;

  if (geminiKey) {
    try {
      const parts: any[] = [{ text: userQuestion }];
      for (const img of validImages) {
        parts.push({
          inlineData: {
            mimeType: img.mimeType || 'image/jpeg',
            data: img.dataUrl!.split(',')[1]
          }
        });
      }
      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${settings.geminiModel || 'gemini-2.5-flash'}:generateContent?key=${geminiKey}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ contents: [{ parts }] })
        }
      );
      if (res.ok) {
        const data = await res.json();
        const reply = data?.candidates?.[0]?.content?.parts?.map((p: any) => p.text).join('\n');
        if (reply && !isVisionRefusal(reply)) {
          return { text: reply, provider: 'Cloud Vision · Gemini' };
        }
      }
    } catch {}
  }

  // 2. Try OpenRouter Free Vision Models if an OpenRouter key is saved
  const openRouterKey =
    anySettings.openrouterApiKey ||
    anySettings.openRouterApiKey ||
    anySettings.openrouterKey ||
    anySettings.apiKey ||
    localStorage.getItem('nova_api_key') ||
    (import.meta as any).env?.VITE_OPENROUTER_API_KEY;

  if (openRouterKey) {
    const visionModels = [
      'google/gemma-3-27b-it:free',
      'qwen/qwen2.5-vl-32b-instruct:free',
      'meta-llama/llama-3.2-11b-vision-instruct:free'
    ];
    for (const modelName of visionModels) {
      try {
        const contentParts: any[] = [{ type: 'text', text: userQuestion }];
        for (const img of validImages) {
          contentParts.push({ type: 'image_url', image_url: { url: img.dataUrl } });
        }
        const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${openRouterKey.trim()}`
          },
          body: JSON.stringify({
            model: modelName,
            messages: [{ role: 'user', content: contentParts }]
          })
        });
        if (res.ok) {
          const data = await res.json();
          const reply = data?.choices?.[0]?.message?.content;
          if (reply && !isVisionRefusal(reply)) {
            return { text: reply, provider: `Cloud Vision · ${modelName.split('/')[1]}` };
          }
        }
      } catch {}
    }
  }

  // 3. Keyless Multimodal Vision Endpoints (tries multiple vision models & rejects any refusal)
  const freeVisionModels = ['openai-large', 'gemini', 'openai'];
  for (const modelName of freeVisionModels) {
    try {
      const contentParts: any[] = [
        {
          type: 'text',
          text: `${userQuestion}\n\n(Local sensor telemetry for context: ${localHints})`
        }
      ];
      for (const img of validImages) {
        contentParts.push({
          type: 'image_url',
          image_url: { url: img.dataUrl }
        });
      }

      const res = await fetch('https://text.pollinations.ai/openai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          messages: [
            {
              role: 'system',
              content: 'You are NOVA Vision AI. Analyze the provided image directly and answer the user clearly.'
            },
            { role: 'user', content: contentParts }
          ]
        })
      });

      if (res.ok) {
        const data = await res.json();
        const reply = data?.choices?.[0]?.message?.content;
        if (reply && !isVisionRefusal(reply)) {
          return { text: reply, provider: 'Cloud Vision · Multimodal AI' };
        }
      }
    } catch {}
  }

  return null;
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('nova-unified-settings') || '{}');
      return {
        ...defaults,
        ...saved,
        systemInstruction: defaults.systemInstruction
      };
    } catch {
      return defaults;
    }
  });

  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'model',
      text: 'Hello. I am NOVA Gesture AI.\n\nI can combine local sign recognition, custom gestures, image & file analysis, and local/cloud AI models.',
      timestamp: Date.now()
    }
  ]);
  const [messageImages, setMessageImages] = useState<Record<string, string[]>>({});
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<LocalAttachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [typing, setTyping] = useState(false);
  const [provider, setProvider] = useState('Ready');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  const lastLetterRef = useRef<string>('');
  const lastLetterTimeRef = useRef<number>(0);
  const lastGestureTimeRef = useRef<number>(0);

  useEffect(() => {
    localStorage.setItem('nova-unified-settings', JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const speak = useCallback((text: string) => {
    if (!settings.voiceEnabled || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    speechSynthesis.speak(u);
  }, [settings.voiceEnabled]);

  const send = useCallback(async (value = input) => {
    const text = value.trim();
    if ((!text && attachments.length === 0) || typing) return;

    const currentAttachments = [...attachments];
    const imageAttachments = currentAttachments.filter(a => a.isImage && a.dataUrl);

    const fileNames = currentAttachments.map(a => `📎 ${a.name}`).join(', ');
    const displayLabel = text
      ? (fileNames ? `${text}\n(${fileNames})` : text)
      : `Attached: ${fileNames}`;

    const msgId = crypto.randomUUID();
    const userMsg: Message = {
      id: msgId,
      role: 'user',
      text: displayLabel,
      timestamp: Date.now()
    };

    if (imageAttachments.length > 0) {
      setMessageImages(prev => ({
        ...prev,
        [msgId]: imageAttachments.map(img => img.dataUrl!)
      }));
    }

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setAttachments([]);
    setTyping(true);

    try {
      // 1. Run Multimodal Cloud Vision if an image is attached
      if (imageAttachments.length > 0) {
        const visionResult = await analyzeImagesWithVisionAI(text, imageAttachments, settings);
        if (visionResult) {
          setProvider(visionResult.provider);
          setMessages(prev => [
            ...prev,
            { id: crypto.randomUUID(), role: 'model', text: visionResult.text, timestamp: Date.now() }
          ]);
          speak(visionResult.text);
          setTyping(false);
          return;
        }
      }

      // 2. Fallback: Use Local MediaPipe/OCR/Pixel telemetry or text file contents
      const telemetryBlock = currentAttachments.length > 0
        ? '\n\n[LOCAL VISION & FILE SCANNER RESULTS — Analyze this extracted data directly and NEVER say you cannot view images]:\n' +
          currentAttachments.map(a => a.localVisualReport || a.content).join('\n\n')
        : '';

      const combinedPrompt =
        (text || 'Summarize and explain the findings from the scanned file/image below.') + telemetryBlock;

      const result = await generateLocalOrCloud(combinedPrompt, messages, settings);

      // If the text model still attempts a refusal on an image, return the Local Vision Scanner report directly
      const finalReply =
        imageAttachments.length > 0 && isVisionRefusal(result.text)
          ? `**Local Vision Analysis:**\n\n${imageAttachments.map(a => a.localVisualReport).join('\n\n')}`
          : result.text;

      setProvider(result.provider);
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'model', text: finalReply, timestamp: Date.now() }
      ]);
      speak(finalReply);
    } catch (err: any) {
      const localFallback =
        imageAttachments.length > 0
          ? `**Local Vision Analysis (Offline):**\n\n${imageAttachments.map(a => a.localVisualReport).join('\n\n')}`
          : `Processed locally: "${text || fileNames}"`;
      setProvider('Local Vision Engine');
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'model', text: localFallback, timestamp: Date.now() }
      ]);
      speak(localFallback);
    } finally {
      setTyping(false);
    }
  }, [input, messages, settings, typing, speak, attachments]);

  const onFilePicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    try {
      const parsed = await Promise.all(Array.from(files).map(readPickedFile));
      setAttachments(prev => [...prev, ...parsed]);
    } catch (err) {
      console.error('File read error:', err);
    }
    e.target.value = '';
  };

  const toggleMic = () => {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setProvider('Browser speech recognition unavailable');
      return;
    }
    const r = new SpeechRecognition();
    r.lang = 'en-US';
    r.continuous = false;
    r.interimResults = false;
    r.onstart = () => setListening(true);
    r.onend = () => setListening(false);
    r.onerror = () => setListening(false);
    r.onresult = (e: any) => setInput((e.results?.[0]?.[0]?.transcript || '').trim());
    recognitionRef.current = r;
    r.start();
  };

  const execute = useCallback((result: VisionResult) => {
    const now = Date.now();

    if (result.type === 'LETTER' && result.value) {
      const isSame = result.value === lastLetterRef.current;
      const waitMs = isSame ? SAME_LETTER_COOLDOWN_MS : LETTER_COOLDOWN_MS;
      if (now - lastLetterTimeRef.current < waitMs) return;

      lastLetterRef.current = result.value;
      lastLetterTimeRef.current = now;
      setInput(prev => prev + result.value);
      return;
    }

    if (result.value === 'CLEAR') {
      if (now - lastGestureTimeRef.current < 900) return;
      lastGestureTimeRef.current = now;
      lastLetterRef.current = '';
      setInput('');
      return;
    }

    if (result.value === 'SEND') {
      if (now - lastGestureTimeRef.current < 1200) return;
      lastGestureTimeRef.current = now;
      lastLetterRef.current = '';
      send();
      return;
    }

    if (result.value === 'THEME_SWITCH') return;

    const custom = settings.customGestures.find(g => g.name.toUpperCase() === result.value.toUpperCase());
    if (!custom) return;
    if (custom.action === 'CLEAR') setInput('');
    else if (custom.action === 'COPY_LAST') navigator.clipboard?.writeText(messages.filter(m => m.role === 'model').at(-1)?.text || '');
    else if (custom.action === 'TOGGLE_MIC') toggleMic();
    else if (custom.action === 'SEND_MESSAGE') send();
  }, [messages, settings.customGestures, send]);

  const onDetected = useCallback((result: VisionResult) => execute(result), [execute]);
  const vision = useVision(settings, settings.customGestures, onDetected);
  const themeIcon = settings.theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>;

  return (
    <div className="app">
      <header>
        <div className="brand">
          <div className="logo"><Sparkles size={20}/></div>
          <div>
            <h1>NOVA GESTURE AI</h1>
            <span><i/> {settings.aiProvider === 'ollama' ? 'OFFLINE-FIRST' : 'HYBRID AI'} · {provider}</span>
          </div>
        </div>
        <div className="header-actions">
          <button title="Voice" onClick={() => setSettings(s => ({ ...s, voiceEnabled: !s.voiceEnabled }))}>
            {settings.voiceEnabled ? <Mic/> : <MicOff/>}
          </button>
          <button title="Theme" onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))}>
            {themeIcon}
          </button>
          <button title="Settings" onClick={() => setSettingsOpen(true)}>
            <Settings/>
          </button>
        </div>
      </header>
      <main>
        <aside>
          <CameraView
            videoRef={vision.videoRef}
            enabled={settings.visionEnabled}
            status={vision.status}
            lastDetection={vision.lastDetection}
            onToggle={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))}
            settings={settings}
          />
          <div className="capabilities">
            <div className="eyebrow">SYSTEM CAPABILITIES</div>
            <div className="cap"><span>Local sign engine</span><b>{vision.status === 'local' ? 'ACTIVE' : 'MODEL NEEDED'}</b></div>
            <div className="cap"><span>Local LLM</span><b>{settings.ollamaModel}</b></div>
            <div className="cap"><span>Cloud fallback</span><b>{settings.aiProvider === 'ollama' ? 'OFF' : 'OPTIONAL'}</b></div>
            <div className="cap"><span>Custom gestures</span><b>{settings.customGestures.length}</b></div>
            <p><WifiOff size={14}/> The goal is to keep the frequent camera loop offline. Cloud AI is never required for chat when Ollama is running.</p>
          </div>
        </aside>
        <section className="chat">
          <div className="chat-head">
            <div><b>Assistant</b><span>Sign language companion</span></div>
            <button onClick={() => setMessages([])}><Trash2 size={16}/> Clear</button>
          </div>
          <div className="messages">
            {messages.map(m => (
              <div key={m.id} className={`message ${m.role}`}>
                <div className="avatar">{m.role === 'user' ? <User size={15}/> : <Bot size={15}/>}</div>
                <div className="bubble">
                  {messageImages[m.id] && (
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                      {messageImages[m.id].map((src, idx) => (
                        <img
                          key={idx}
                          src={src}
                          alt="Attached"
                          style={{ maxWidth: 220, maxHeight: 180, borderRadius: 8, objectFit: 'cover' }}
                        />
                      ))}
                    </div>
                  )}
                  <div>{m.text}</div>
                  {m.role === 'model' && (
                    <button className="copy" onClick={() => navigator.clipboard?.writeText(m.text)}>
                      <Copy size={13}/>
                    </button>
                  )}
                </div>
              </div>
            ))}
            {typing && (
              <div className="message model">
                <div className="avatar"><Bot size={15}/></div>
                <div className="bubble dots">● ● ●</div>
              </div>
            )}
            <div ref={chatEnd}/>
          </div>
          {attachments.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', padding: '0 14px 8px', alignItems: 'center' }}>
              {attachments.map((a, i) => (
                <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(148,163,184,0.15)', borderRadius: 999, padding: '4px 10px', fontSize: 12 }}>
                  {a.dataUrl && (
                    <img src={a.dataUrl} alt={a.name} style={{ width: 20, height: 20, borderRadius: 4, objectFit: 'cover' }} />
                  )}
                  {a.name}
                  <button style={{ display: 'flex' }} onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))}>
                    <X size={12}/>
                  </button>
                </span>
              ))}
            </div>
          )}
          <input ref={fileInputRef} type="file" multiple hidden onChange={onFilePicked} />
          <div className="composer">
            <button className={listening ? 'active mic' : 'mic'} onClick={toggleMic}>
              {listening ? <MicOff/> : <Mic/>}
            </button>
            <button className="mic" title="Attach file or image" onClick={() => fileInputRef.current?.click()}>
              <Paperclip/>
            </button>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="Type, speak, attach an image/file, or use a sign…"
            />
            <button id="nova-send-btn" className="send" onClick={() => send()} disabled={(!input.trim() && attachments.length === 0) || typing}>
              <Send/>
            </button>
          </div>
        </section>
      </main>
      {settingsOpen && <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)}/>}
    </div>
  );
}

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}
