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
  systemInstruction: 'You are NOVA, a concise and helpful assistant designed for people who use sign language. Be clear, respectful and fast.',
  customGestures: []
};

// Controls letter speed so signs never spam "AAAA..."
const LETTER_COOLDOWN_MS = 1000;
const SAME_LETTER_COOLDOWN_MS = 1800;

// Built-in file reader so App.tsx never fails on missing external imports
async function readPickedFile(file: File): Promise<LocalAttachment> {
  if (file.type.startsWith('image/')) {
    return {
      name: file.name,
      content: `[Attached Image: ${file.name}, type: ${file.type}, size: ${Math.round(file.size / 1024)}KB]`,
      isImage: true
    };
  }
  try {
    const rawText = await file.text();
    const clipped = rawText.slice(0, 20000);
    return {
      name: file.name,
      content: `--- FILE: ${file.name} ---\n${clipped}\n--- END OF ${file.name} ---`
    };
  } catch {
    return {
      name: file.name,
      content: `[Attached Binary File: ${file.name} (${Math.round(file.size / 1024)}KB)]`
    };
  }
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      return { ...defaults, ...JSON.parse(localStorage.getItem('nova-unified-settings') || '{}') };
    } catch {
      return defaults;
    }
  });
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'model',
      text: 'Hello. I am NOVA Gesture AI.\n\nI can combine local sign recognition, custom gestures, file analysis, and local/cloud AI models.',
      timestamp: Date.now()
    }
  ]);
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
    const fileNames = currentAttachments.map(a => `📎 ${a.name}`).join(', ');
    const displayLabel = text
      ? (fileNames ? `${text}\n(${fileNames})` : text)
      : `Attached: ${fileNames}`;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      text: displayLabel,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setAttachments([]);
    setTyping(true);

    try {
      // Combine user text + attached file contents into a single prompt string
      // so generateLocalOrCloud only needs its standard 3 arguments
      const fileBlock = currentAttachments.length > 0
        ? '\n\nAttached File Contents:\n' + currentAttachments.map(a => a.content).join('\n\n')
        : '';
      const combinedPrompt = (text || 'Please analyze the attached file(s) and summarize key insights.') + fileBlock;

      const result = await generateLocalOrCloud(combinedPrompt, messages, settings);
      setProvider(result.provider);
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'model', text: result.text, timestamp: Date.now() }
      ]);
      speak(result.text);
    } catch (err: any) {
      const fallbackText = !navigator.onLine
        ? `Offline Mode Active: Your message ("${text || fileNames}") was processed locally.`
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

    // 1. 26 ASL Letters (A-Z) with anti-repeat cooldown
    if (result.type === 'LETTER' && result.value) {
      const isSame = result.value === lastLetterRef.current;
      const waitMs = isSame ? SAME_LETTER_COOLDOWN_MS : LETTER_COOLDOWN_MS;
      if (now - lastLetterTimeRef.current < waitMs) return;

      lastLetterRef.current = result.value;
      lastLetterTimeRef.current = now;
      setInput(prev => prev + result.value);
      return;
    }

    // 2. Thumbs Down -> Clear all text
    if (result.value === 'CLEAR') {
      if (now - lastGestureTimeRef.current < 900) return;
      lastGestureTimeRef.current = now;
      lastLetterRef.current = '';
      setInput('');
      return;
    }

    // 3. Yo-Yo Sign (Index + Pinky) -> Send message
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
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', padding: '0 14px 8px' }}>
              {attachments.map((a, i) => (
                <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(148,163,184,0.15)', borderRadius: 999, padding: '4px 10px', fontSize: 12 }}>
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
            <button className="mic" title="Attach file" onClick={() => fileInputRef.current?.click()}>
              <Paperclip/>
            </button>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="Type, speak, or use a sign…"
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
