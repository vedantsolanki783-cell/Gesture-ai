import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Bot, Copy, Mic, MicOff, Moon, Paperclip, Send, Settings, Sparkles, Sun, Trash2, User, X } from 'lucide-react';
import { generateLocalOrCloud } from './services/aiRouter';
import { parseUploadedFile } from './services/fileReader';
import type { MessageAttachment } from './services/hybridAI';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import { learnSign } from './vision/localVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

const defaults: AppSettings = {
  theme: 'dark', voiceEnabled: true, visionEnabled: false, confidenceThreshold: 0.72,
  aiProvider: 'cloudFree', ollamaUrl: 'http://localhost:11434', ollamaModel: 'qwen2.5:1.5b', geminiModel: 'gemini-2.5-flash',
  cloudFreeBaseUrl: 'https://api.groq.com/openai/v1', cloudFreeApiKey: '', cloudFreeModel: 'llama-3.3-70b-versatile',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction: 'You are NOVA, a concise and helpful assistant designed for people who use sign language. Be clear, respectful and fast.', customGestures: []
};

function executeAndroidAgentCommand(rawText: string): string | null {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  const text = rawText.toLowerCase().trim();
  const isOverlay = /\b(enable|start|show|turn on)\s+(overlay|bubble|hud)\b/i.test(text);
  const isUltron = /\b(enable|open|turn on)\s+(accessibility|ultron)\b/i.test(text);
  const isHome = /^(go\s+home|home)$/i.test(text);
  const isBack = /^(go\s+back|back)$/i.test(text);
  const openAppMatch = text.match(/^(?:open|launch)\s+([a-z0-9\s._-]+)$/i);
  if (!bridge) return null;
  if (isOverlay) return bridge.enableOverlayBubble() === 'OPENED_OVERLAY_SETTINGS' ? 'Opening overlay permission settings…' : 'Floating bubble active.';
  if (isUltron) { bridge.openAccessibilitySettings(); return 'Opening Accessibility settings…'; }
  if (openAppMatch && bridge.openApp(openAppMatch[1].trim())) return `Launching ${openAppMatch[1].trim()}…`;
  if (!bridge.isUltronConnected()) { bridge.openAccessibilitySettings(); return 'The Accessibility Service is off — turn it on to use device commands.'; }
  if (isHome) { bridge.globalAction('HOME'); return 'Done.'; }
  if (isBack) { bridge.globalAction('BACK'); return 'Done.'; }
  return null;
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try { return { ...defaults, ...JSON.parse(localStorage.getItem('nova-unified-settings') || '{}') }; } catch { return defaults; }
  });
  const [messages, setMessages] = useState<Message[]>([{ id: 'welcome', role: 'model', text: 'Hello. I am NOVA Gesture AI.\n\nI can combine local sign recognition, custom gestures and an AI model. You can also teach me new hand signs below the camera.', timestamp: Date.now() }]);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<MessageAttachment[]>([]);
  const [learnLabel, setLearnLabel] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [typing, setTyping] = useState(false);
  const [provider, setProvider] = useState('Ready');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => { localStorage.setItem('nova-unified-settings', JSON.stringify(settings)); document.documentElement.dataset.theme = settings.theme; }, [settings]);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, typing]);

  const speak = useCallback((text: string) => {
    if (!settings.voiceEnabled || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; speechSynthesis.speak(u);
  }, [settings.voiceEnabled]);

  const send = useCallback(async (value = input) => {
    const text = value.trim(); if ((!text && attachments.length === 0) || typing) return;
    const label = text || (attachments.length ? `[${attachments.length} file(s) attached]` : '');
    const user: Message = { id: crypto.randomUUID(), role: 'user', text: label, timestamp: Date.now() };
    setMessages(prev => [...prev, user]); setInput(''); setTyping(true);

    const androidReply = executeAndroidAgentCommand(text);
    if (androidReply) {
      setAttachments([]); setProvider('Device command'); setTyping(false);
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: androidReply, timestamp: Date.now() }]);
      return;
    }

    const result = await generateLocalOrCloud(text, messages, settings, attachments);
    setAttachments([]);
    setProvider(result.provider); setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: result.text, timestamp: Date.now() }]); setTyping(false); speak(result.text);
  }, [input, messages, settings, typing, speak, attachments]);

  const onFilePicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    try {
      const parsed = await Promise.all(Array.from(files).map(parseUploadedFile));
      setAttachments(prev => [...prev, ...parsed]);
    } catch (err) { console.error('File read error:', err); }
    e.target.value = '';
  };

  const toggleMic = () => {
    if (listening) { recognitionRef.current?.stop(); setListening(false); return; }
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) { setProvider('Browser speech recognition unavailable'); return; }
    const r = new SpeechRecognition(); r.lang = 'en-US'; r.continuous = false; r.interimResults = false;
    r.onstart = () => setListening(true); r.onend = () => setListening(false); r.onerror = () => setListening(false);
    r.onresult = (e: any) => setInput((e.results?.[0]?.[0]?.transcript || '').trim()); recognitionRef.current = r; r.start();
  };

  const execute = useCallback((result: VisionResult) => {
    if (result.type === 'LETTER') { setInput(prev => prev + result.value); return; }
    if (result.value === 'CLEAR') { setInput(''); return; }
    if (result.value === 'SEND') { send(); return; }
    if (result.value === 'THEME_SWITCH') { setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' })); return; }
    const custom = settings.customGestures.find(g => g.name.toUpperCase() === result.value.toUpperCase());
    if (!custom) return;
    if (custom.action === 'CLEAR') setInput('');
    else if (custom.action === 'THEME_SWITCH') setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }));
    else if (custom.action === 'COPY_LAST') navigator.clipboard?.writeText(messages.filter(m => m.role === 'model').at(-1)?.text || '');
    else if (custom.action === 'TOGGLE_MIC') toggleMic();
    else if (custom.action === 'SEND_MESSAGE') setInput(prev => `${prev}${prev ? ' ' : ''}${custom.name}`);
  }, [messages, settings.customGestures, send]);

  const onDetected = useCallback((result: VisionResult) => execute(result), [execute]);
  const vision = useVision(settings, settings.customGestures, onDetected);
  const themeIcon = settings.theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>;

  return <div className="app">
    <header><div className="brand"><div className="logo"><Sparkles size={20}/></div><div><h1>NOVA GESTURE AI</h1><span><i/> {provider}</span></div></div><div className="header-actions"><button title="Voice" onClick={() => setSettings(s => ({ ...s, voiceEnabled: !s.voiceEnabled }))}>{settings.voiceEnabled ? <Mic/> : <MicOff/>}</button><button title="Theme" onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))}>{themeIcon}</button><button title="Settings" onClick={() => setSettingsOpen(true)}><Settings/></button></div></header>
    <main>
      <aside>
        <CameraView videoRef={vision.videoRef} enabled={settings.visionEnabled} status={vision.status} lastDetection={vision.lastDetection} onToggle={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} settings={settings}/>
        {settings.visionEnabled && <div className="learn-row">
          <input value={learnLabel} onChange={e => setLearnLabel(e.target.value.toUpperCase())} placeholder="Sign label, e.g. A" maxLength={16}/>
          <button className="secondary" onClick={() => { if (learnLabel.trim()) { learnSign(learnLabel.trim()); setLearnLabel(''); } }}>Teach this sign</button>
        </div>}
      </aside>
      <section className="chat"><div className="chat-head"><div><b>Assistant</b><span>Sign language companion</span></div><button onClick={() => setMessages([])}><Trash2 size={16}/> Clear</button></div><div className="messages">{messages.map(m => <div key={m.id} className={`message ${m.role}`}><div className="avatar">{m.role === 'user' ? <User size={15}/> : <Bot size={15}/>}</div><div className="bubble"><div>{m.text}</div>{m.role === 'model' && <button className="copy" onClick={() => navigator.clipboard?.writeText(m.text)}><Copy size={13}/></button>}</div></div>)}{typing && <div className="message model"><div className="avatar"><Bot size={15}/></div><div className="bubble dots">● ● ●</div></div>}<div ref={chatEnd}/></div>
        {attachments.length > 0 && <div className="attachments-row">{attachments.map((a, i) => <span key={i} className="attachment-chip">{a.name}<button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))}><X size={12}/></button></span>)}</div>}
        <input ref={fileInputRef} type="file" multiple hidden onChange={onFilePicked} />
        <div className="composer"><button className={listening ? 'active mic' : 'mic'} onClick={toggleMic}>{listening ? <MicOff/> : <Mic/>}</button><button className="mic" title="Attach file" onClick={() => fileInputRef.current?.click()}><Paperclip/></button><input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()} placeholder="Type, speak, or use a sign…"/><button className="send" onClick={() => send()} disabled={(!input.trim() && attachments.length === 0) || typing}><Send/></button></div></section>
    </main>
    {settingsOpen && <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)}/>}
  </div>;
}

declare global { interface Window { SpeechRecognition: any; webkitSpeechRecognition: any; } }
