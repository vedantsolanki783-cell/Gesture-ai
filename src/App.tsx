import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Bot, Copy, Mic, MicOff, Moon, Paperclip, Send, Settings, Sparkles, Sun, Trash2, User, WifiOff, X } from 'lucide-react';
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
  systemInstruction: 'You are NOVA, a powerful multimodal AI assistant capable of analyzing images, reading files, writing code, and assisting sign language users clearly and directly.',
  customGestures: []
};

const LETTER_COOLDOWN_MS = 1000;
const SAME_LETTER_COOLDOWN_MS = 1800;

// Converts an uploaded image into a fast, compressed Base64 Data URL so Vision AI can see it
function imageToDataUrl(file: File): Promise<{ dataUrl: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Failed to read image'));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error('Failed to decode image'));
      img.onload = () => {
        const maxDim = 900;
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
        const compressed = canvas.toDataURL('image/jpeg', 0.82);
        resolve({ dataUrl: compressed, width, height });
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  });
}

async function readPickedFile(file: File): Promise<LocalAttachment> {
  if (file.type.startsWith('image/')) {
    try {
      const { dataUrl, width, height } = await imageToDataUrl(file);
      return {
        name: file.name,
        isImage: true,
        dataUrl,
        mimeType: 'image/jpeg',
        content: `[Image: ${file.name} (${width}x${height}px)]`
      };
    } catch {
      return {
        name: file.name,
        isImage: true,
        content: `[Image: ${file.name}]`
      };
    }
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
      content: `[Attached Binary File: ${file.name} (${Math.round(file.size / 1024)}KB)]`
    };
  }
}

// Sends actual Base64 image pixels to a multimodal Vision model so NOVA genuinely sees the image
async function analyzeImagesWithVisionAI(
  promptText: string,
  images: LocalAttachment[],
  settings: AppSettings
): Promise<{ text: string; provider: string } | null> {
  const validImages = images.filter(img => !!img.dataUrl);
  if (validImages.length === 0 || !navigator.onLine) return null;

  const userQuestion =
    promptText.trim() ||
    'Analyze this image in detail. Describe what is shown, read any visible text, and explain its key details.';

  // 1. Try Gemini Vision API directly if a Gemini key is configured
  const anySettings = settings as any;
  const geminiKey = anySettings.geminiApiKey || (import.meta as any).env?.VITE_GEMINI_API_KEY;
  if (geminiKey) {
    try {
      const parts: any[] = [{ text: userQuestion }];
      for (const img of validImages) {
        const base64Data = img.dataUrl!.split(',')[1];
        parts.push({
          inlineData: {
            mimeType: img.mimeType || 'image/jpeg',
            data: base64Data
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
        if (reply) return { text: reply, provider: 'Cloud Vision · Gemini' };
      }
    } catch (e) {
      console.warn('Gemini vision fallback:', e);
    }
  }

  // 2. Free Keyless Multimodal Vision API (works out-of-the-box with zero API key)
  try {
    const contentParts: any[] = [
      {
        type: 'text',
        text: `You are NOVA, a helpful multimodal AI assistant. Analyze the attached image(s) directly and answer the user's request clearly.\n\nUser request: ${userQuestion}`
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
        model: 'openai',
        messages: [{ role: 'user', content: contentParts }]
      })
    });

    if (res.ok) {
      const data = await res.json();
      const reply = data?.choices?.[0]?.message?.content;
      if (reply) return { text: reply, provider: 'Cloud Vision · Multimodal AI' };
    }
  } catch (e) {
    console.warn('Multimodal vision error:', e);
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
        // Upgrade old restrictive system instruction automatically
        systemInstruction:
          saved.systemInstruction && !saved.systemInstruction.includes('limited')
            ? saved.systemInstruction
            : defaults.systemInstruction
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
  // Stores Base64 image previews keyed by message ID so TypeScript types stay 100% compatible
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
    const textAttachments = currentAttachments.filter(a => !a.isImage);

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
      // 1. If the user attached image(s), run real Multimodal Vision analysis first
      if (imageAttachments.length > 0) {
        const extraTextContext = textAttachments.length > 0
          ? '\n\nAdditional attached text files:\n' + textAttachments.map(a => a.content).join('\n\n')
          : '';
        const visionResult = await analyzeImagesWithVisionAI(
          text + extraTextContext,
          imageAttachments,
          settings
        );

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

      // 2. For text/code/CSV files or standard chat messages
      const fileBlock = currentAttachments.length > 0
        ? '\n\n[INSTRUCTION: Read and analyze the attached file content below directly. Do not say you are limited to gesture data.]\n\n' +
          currentAttachments.map(a => a.content).join('\n\n')
        : '';

      const combinedPrompt =
        (text || 'Please analyze the attached file(s) and provide a clear summary.') + fileBlock;

      const result = await generateLocalOrCloud(combinedPrompt, messages, settings);
      setProvider(result.provider);
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'model', text: result.text, timestamp: Date.now() }
      ]);
      speak(result.text);
    } catch (err: any) {
      const fallbackText = !navigator.onLine
        ? `Offline Mode Active: Your message ("${text || fileNames}") was recorded locally.`
        : `Could not reach AI provider (${err?.message || 'check settings'}).`;
      setProvider('Offline / Local Fallback');
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'model', text: fallbackText, timestamp: Date.now() }
      ]);
      speak(text || fallbackText);
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
