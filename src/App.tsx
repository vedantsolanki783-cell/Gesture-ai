import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Copy, Mic, MicOff, Moon, Send, Settings, Sparkles, Sun, Trash2, User, WifiOff } from 'lucide-react';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

const defaults: AppSettings = {
  theme: 'dark', voiceEnabled: true, visionEnabled: false, confidenceThreshold: 0.72,
  aiProvider: 'cloudFree', ollamaUrl: 'http://localhost:11434', ollamaModel: 'qwen3:4b', geminiModel: 'gemini-2.5-flash',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction: 'You are NOVA, a concise and helpful assistant designed for people who use sign language. Be clear, respectful and fast.', customGestures: []
};

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try { return { ...defaults, ...JSON.parse(localStorage.getItem('nova-unified-settings') || '{}') }; } catch { return defaults; }
  });
  const [messages, setMessages] = useState<Message[]>([{ id: 'welcome', role: 'model', text: 'Hello. I am NOVA Gesture AI.\n\nI can combine local sign recognition, custom gestures and a local AI model. Cloud AI is optional.', timestamp: Date.now() }]);
  const [input, setInput] = useState('');
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
    const text = value.trim(); if (!text || typing) return;
    const user: Message = { id: crypto.randomUUID(), role: 'user', text, timestamp: Date.now() };
    setMessages(prev => [...prev, user]); setInput(''); setTyping(true);
    const result = await generateLocalOrCloud(text, messages, settings);
    setProvider(result.provider); setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: result.text, timestamp: Date.now() }]); setTyping(false); speak(result.text);
  }, [input, messages, settings, typing, speak]);

  const toggleMic = () => {
    if (listening) { recognitionRef.current?.stop(); setListening(false); return; }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { setProvider('Browser speech recognition unavailable'); return; }
    const r = new SpeechRecognition(); r.lang = 'en-US'; r.continuous = false; r.interimResults = false;
    r.onstart = () => setListening(true); r.onend = () => setListening(false); r.onerror = () => setListening(false);
    r.onresult = (e: any) => setInput((e.results?.[0]?.[0]?.transcript || '').trim()); recognitionRef.current = r; r.start();
  };

  const execute = useCallback((result: VisionResult) => {
    if (result.type === 'LETTER') { setInput(prev => prev + result.value); return; }
    if (result.value === 'CLEAR') { setInput(''); return; }
    if (result.value === 'THEME_SWITCH') { setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' })); return; }
    const custom = settings.customGestures.find(g => g.name.toUpperCase() === result.value.toUpperCase());
    if (!custom) return;
    if (custom.action === 'CLEAR') setInput('');
    else if (custom.action === 'THEME_SWITCH') setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }));
    else if (custom.action === 'COPY_LAST') navigator.clipboard?.writeText(messages.filter(m => m.role === 'model').at(-1)?.text || '');
    else if (custom.action === 'TOGGLE_MIC') toggleMic();
    else if (custom.action === 'SEND_MESSAGE') setInput(prev => `${prev}${prev ? ' ' : ''}${custom.name}`);
  }, [messages, settings.customGestures]);

  const onDetected = useCallback((result: VisionResult) => execute(result), [execute]);
  const vision = useVision(settings, settings.customGestures, onDetected);
  const online = provider.startsWith('Cloud');
  const themeIcon = settings.theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>;

  return <div className="app">
    <header><div className="brand"><div className="logo"><Sparkles size={20}/></div><div><h1>NOVA GESTURE AI</h1><span><i/> {settings.aiProvider === 'ollama' ? 'OFFLINE-FIRST' : 'HYBRID AI'} · {provider}</span></div></div><div className="header-actions"><button title="Voice" onClick={() => setSettings(s => ({ ...s, voiceEnabled: !s.voiceEnabled }))}>{settings.voiceEnabled ? <Mic/> : <MicOff/>}</button><button title="Theme" onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))}>{themeIcon}</button><button title="Settings" onClick={() => setSettingsOpen(true)}><Settings/></button></div></header>
    <main>
      <aside><CameraView videoRef={vision.videoRef} enabled={settings.visionEnabled} status={vision.status} lastDetection={vision.lastDetection} onToggle={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} settings={settings}/><div className="capabilities"><div className="eyebrow">SYSTEM CAPABILITIES</div><div className="cap"><span>Local sign engine</span><b>{vision.status === 'local' ? 'ACTIVE' : 'MODEL NEEDED'}</b></div><div className="cap"><span>Local LLM</span><b>{settings.ollamaModel}</b></div><div className="cap"><span>Cloud fallback</span><b>{settings.aiProvider === 'ollama' ? 'OFF' : 'OPTIONAL'}</b></div><div className="cap"><span>Custom gestures</span><b>{settings.customGestures.length}</b></div><p><WifiOff size={14}/> The goal is to keep the frequent camera loop offline. Cloud AI is never required for chat when Ollama is running.</p></div></aside>
      <section className="chat"><div className="chat-head"><div><b>Assistant</b><span>Sign language companion</span></div><button onClick={() => setMessages([])}><Trash2 size={16}/> Clear</button></div><div className="messages">{messages.map(m => <div key={m.id} className={`message ${m.role}`}><div className="avatar">{m.role === 'user' ? <User size={15}/> : <Bot size={15}/>}</div><div className="bubble"><div>{m.text}</div>{m.role === 'model' && <button className="copy" onClick={() => navigator.clipboard?.writeText(m.text)}><Copy size={13}/></button>}</div></div>)}{typing && <div className="message model"><div className="avatar"><Bot size={15}/></div><div className="bubble dots">● ● ●</div></div>}<div ref={chatEnd}/></div><div className="composer"><button className={listening ? 'active mic' : 'mic'} onClick={toggleMic}>{listening ? <MicOff/> : <Mic/>}</button><input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()} placeholder="Type, speak, or use a sign…"/><button className="send" onClick={() => send()} disabled={!input.trim() || typing}><Send/></button></div></section>
    </main>
    {settingsOpen && <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)}/>} 
  </div>;
}

declare global { interface Window { SpeechRecognition: any; webkitSpeechRecognition: any; } }
