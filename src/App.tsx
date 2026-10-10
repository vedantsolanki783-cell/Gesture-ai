import React, { useCallback, useEffect, useRef, useState } from 'react';
import { BrainCircuit, Bell, Bot, Code2, Copy, Eye, Hand, History, Home, Image as ImageIcon, Menu, Mic, MicOff, MessageSquare, Moon, Paperclip, Plus, Send, Settings, Smartphone, Sparkles, Sun, User, UserCog, Wand2, X, Video } from 'lucide-react';
import { generateLocalOrCloud } from './services/aiRouter';
import { parseUploadedFile } from './services/fileReader';
import type { MessageAttachment } from './services/hybridAI';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { NovaFace } from './components/NovaFace';
import { useVision } from './hooks/useVision';
import { learnSign } from './vision/localVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

const defaults: AppSettings = {
  theme: 'light', voiceEnabled: true, visionEnabled: false, confidenceThreshold: 0.72,
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

interface ChatSession { id: string; title: string; messages: Message[]; updatedAt: number; }
interface AppNotification { id: string; text: string; time: number; }

const WELCOME: Message = { id: 'welcome', role: 'model', text: 'Hello. I am NOVA Gesture AI.\n\nI can combine local sign recognition, custom gestures and an AI model. You can also teach me new hand signs below the camera.', timestamp: Date.now() };

type View = 'home' | 'vision' | 'chat' | 'devices' | 'aitools';
type AiMode = 'assistant' | 'thinking' | 'image' | 'video' | 'coder';

const AI_MODE_INFO: Record<AiMode, { label: string; blurb: string; systemInstruction?: string }> = {
  assistant: { label: 'Personal Assistant', blurb: 'General help, chat and device control — NOVA\'s default mode.' },
  thinking: {
    label: 'Deep Thinking',
    blurb: 'Slower, more thorough reasoning for hard or multi-step questions.',
    systemInstruction: 'Think step-by-step before answering: break the problem down, weigh alternatives or edge cases, then give one clear, well-reasoned final answer.'
  },
  image: { label: 'Image Generation', blurb: 'Describe a picture and NOVA will generate it (max 40s).' },
  video: { label: 'Video Generation', blurb: 'Generate short videos from a text prompt (max 3m).' },
  coder: {
    label: 'Coder X',
    blurb: 'A focused coding assistant — complete, copy-paste-ready code with brief explanations.',
    systemInstruction: 'You are Coder X, an expert programming assistant inside NOVA. The person you are helping is phone/tablet-only and pastes whole files into GitHub\'s mobile editor, so always give complete, copy-paste-ready code (not diffs or snippets) and explain changes briefly in plain language.'
  }
};

function buildImageUrl(prompt: string): string {
  return `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=1024&height=1024&nologo=true`;
}

// Added Download Helper Function
const downloadMedia = async (url: string, filename: string) => {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    const blobUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(blobUrl);
  } catch (err) {
    console.error('Download failed:', err);
    alert('Failed to download file.');
  }
};

function Ring({ percent, color, label, value }: { percent: number; color: string; label: string; value: string }) {
  const r = 26; const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, percent));
  return <div className="ring-stat">
    <svg width="64" height="64" viewBox="0 0 64 64">
      <circle cx="32" cy="32" r={r} fill="none" stroke="var(--ring-track)" strokeWidth="6"/>
      <circle cx="32" cy="32" r={r} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round"
        strokeDasharray={`${c}`} strokeDashoffset={`${c - (pct / 100) * c}`} transform="rotate(-90 32 32)"/>
      <text x="32" y="36" textAnchor="middle" fontSize="13" fontWeight="700" fill="var(--text)">{value}</text>
    </svg>
    <span>{label}</span>
  </div>;
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try { return { ...defaults, ...JSON.parse(localStorage.getItem('nova-unified-settings') || '{}') }; } catch { return defaults; }
  });
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [history, setHistory] = useState<ChatSession[]>(() => {
    try { return JSON.parse(localStorage.getItem('nova-chat-history') || '[]'); } catch { return []; }
  });
  
  const [historyOpen, setHistoryOpen] = useState(false);
  const [notifsOpen, setNotifsOpen] = useState(false);
  const [desktopCollapsed, setDesktopCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  
  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const [unreadNotifs, setUnreadNotifs] = useState(0);

  const addNotification = useCallback((text: string) => {
    setNotifications(prev => [{ id: crypto.randomUUID(), text, time: Date.now() }, ...prev]);
    setUnreadNotifs(prev => prev + 1);
  }, []);

  const currentSessionId = useRef<string>(crypto.randomUUID());

  const saveHistory = (list: ChatSession[]) => { setHistory(list); localStorage.setItem('nova-chat-history', JSON.stringify(list)); };

  const persistCurrentSession = useCallback((msgs: Message[]) => {
    const realMsgs = msgs.filter(m => m.id !== 'welcome');
    if (realMsgs.length === 0) return;
    const title = realMsgs.find(m => m.role === 'user')?.text.slice(0, 48) || 'Conversation';
    const entry: ChatSession = { id: currentSessionId.current, title, messages: msgs, updatedAt: Date.now() };
    const rest = history.filter(h => h.id !== entry.id);
    saveHistory([entry, ...rest].slice(0, 50));
  }, [history]);

  const startNewChat = () => { 
    persistCurrentSession(messages); 
    currentSessionId.current = crypto.randomUUID(); 
    setMessages([WELCOME]); 
    setNotifications([{ id: crypto.randomUUID(), text: 'New chat started. System logs reset.', time: Date.now() }]);
    setUnreadNotifs(1);
  };
  
  const loadSession = (id: string) => { const s = history.find(h => h.id === id); if (!s) return; persistCurrentSession(messages); currentSessionId.current = s.id; setMessages(s.messages); setHistoryOpen(false); };
  const deleteSession = (id: string) => saveHistory(history.filter(h => h.id !== id));
  
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

  const [view, setView] = useState<View>('home');
  const [aiMode, setAiMode] = useState<AiMode>('assistant');
  const [batteryLevel, setBatteryLevel] = useState<number | null>(null);

  useEffect(() => { localStorage.setItem('nova-unified-settings', JSON.stringify(settings)); document.documentElement.dataset.theme = settings.theme; }, [settings]);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, typing]);
  useEffect(() => { if (messages.some(m => m.id !== 'welcome')) persistCurrentSession(messages); }, [messages]);

  // Log provider changes
  useEffect(() => {
    addNotification(`AI Provider active: ${settings.aiProvider}`);
  }, [settings.aiProvider, addNotification]);

  useEffect(() => {
    const nav = navigator as any;
    if (!nav.getBattery) return;
    let battery: any;
    const update = () => setBatteryLevel(Math.round(battery.level * 100));
    nav.getBattery().then((b: any) => { battery = b; update(); b.addEventListener('levelchange', update); });
    return () => battery?.removeEventListener?.('levelchange', update);
  }, []);

  const speak = useCallback((text: string) => {
    if (!settings.voiceEnabled || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; speechSynthesis.speak(u);
  }, [settings.voiceEnabled]);

  const send = useCallback(async (value = input) => {
    const text = value.trim(); if ((!text && attachments.length === 0) || typing) return;
    const label = text || (attachments.length ? `[${attachments.length} file(s) attached]` : '');
    const user: Message = { id: crypto.randomUUID(), role: 'user', text: label, timestamp: Date.now() };
    setMessages(prev => [...prev, user]); setInput(''); setTyping(true);
    setView('chat');

    const androidReply = executeAndroidAgentCommand(text);
    if (androidReply) {
      setAttachments([]); setProvider('Device command'); setTyping(false);
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: androidReply, timestamp: Date.now() }]);
      return;
    }

    if (aiMode === 'image' && text) {
      setAttachments([]); setProvider('Image Generation (max 40s)');
      
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 40000);
        
        const url = buildImageUrl(text);
        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Generation failed');
        
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: `Here's your image for: "${text}"`, imageUrl: url, timestamp: Date.now() }]);
      } catch (error) {
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: 'Image generation stopped: Exceeded maximum time of 40 seconds or failed.', timestamp: Date.now() }]);
      }
      setTyping(false);
      return;
    }

    // CORS FIX: Removed the custom header that triggers browser blocks, changed model repo to the official one
    if (aiMode === 'video' && text) {
      setAttachments([]); setProvider('Video Generation (max 3m)');
      
      try {
        let hfToken = localStorage.getItem('nova_hf_token');
        if (!hfToken) {
          hfToken = window.prompt("Security check: Enter your Hugging Face Access Token (starts with hf_):");
          if (!hfToken) throw new Error('Token required for video generation.');
          localStorage.setItem('nova_hf_token', hfToken.trim());
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 180000); // 3-minute limit
        
        // Removed custom headers entirely so the browser allows it safely
        const response = await fetch(
          "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b",
          {
            headers: { 
              "Authorization": `Bearer ${hfToken}`, 
              "Content-Type": "application/json"
            },
            method: "POST",
            body: JSON.stringify({ inputs: text }),
            signal: controller.signal
          }
        );
        
        clearTimeout(timeoutId);
        
        // Exact error handling so it doesn't just say "Failed to fetch"
        if (!response.ok) {
          const errorText = await response.text();
          if (response.status === 401) {
            localStorage.removeItem('nova_hf_token');
            throw new Error('Invalid API Key. It has been removed. Please try again with a fresh key.');
          }
          if (response.status === 503) {
            throw new Error('Server is currently waking up. Please wait 30 seconds and try generating again!');
          }
          throw new Error(`Hugging Face Error (${response.status}): ${errorText}`);
        }
        
        const blob = await response.blob();
        
        // Double check if Hugging face returned an error disguised as a success
        if (blob.type.includes('application/json')) {
            const errorData = await blob.text();
            throw new Error(`API Error: ${errorData}`);
        }

        const localVideoUrl = URL.createObjectURL(blob);
        
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: `Here is your generated video for: "${text}"`, videoUrl: localVideoUrl, timestamp: Date.now() } as any]);
      } catch (error: any) {
        let finalErrorMsg = error.message || 'Unknown error occurred.';
        
        if (error.name === 'AbortError') {
            finalErrorMsg = "Generation timed out. The server took longer than 3 minutes to process the video.";
        }
        
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: `⚠️ Video Error: ${finalErrorMsg}`, timestamp: Date.now() }]);
      }
      setTyping(false);
      return;
    }

    const modeInfo = AI_MODE_INFO[aiMode];
    const effectiveSettings = modeInfo.systemInstruction
      ? { ...settings, systemInstruction: `${settings.systemInstruction}\n\n${modeInfo.systemInstruction}` }
      : settings;

    const result = await generateLocalOrCloud(text, messages, effectiveSettings, attachments);
    setAttachments([]);
    setProvider(result.provider); setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: result.text, timestamp: Date.now() }]); setTyping(false); speak(result.text);
  }, [input, messages, settings, typing, speak, attachments, aiMode]);

  const switchAiMode = (mode: AiMode) => {
    setAiMode(mode);
    setView('chat');
    addNotification(`Switched to ${AI_MODE_INFO[mode].label} mode.`);
  };

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
  const themeIcon = settings.theme === 'dark' ? <Sun size={18}/> : <Moon size={18}/>;
  const bridgeConnected = typeof window !== 'undefined' && !!(window as any).NovaAndroid;
  const lastModelMessage = messages.filter(m => m.role === 'model').at(-1);

  const navItems: { id: View; label: string; icon: React.ReactNode }[] = [
    { id: 'home', label: 'Home', icon: <Home size={17}/> },
    { id: 'vision', label: 'Vision', icon: <Eye size={17}/> },
    { id: 'chat', label: 'Chat', icon: <MessageSquare size={17}/> },
    { id: 'devices', label: 'Devices', icon: <Smartphone size={17}/> },
    { id: 'aitools', label: 'AI Tools', icon: <Wand2 size={17}/> },
  ];

  return <div className="shell">
    {/* Mobile Overlay */}
    <div className={`sidebar-overlay ${mobileOpen ? 'open' : ''}`} onClick={() => setMobileOpen(false)} />
    
    <nav className={`sidebar ${desktopCollapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}>
      <div className="sidebar-brand">
        <div className="logo"><Bot size={18}/></div>
        <div><b>NOVA AI</b><small>Think. Assist. Achieve.</small></div>
      </div>
      <div className="sidebar-nav">
        {navItems.map(item => (
          <button key={item.id} className={view === item.id ? 'nav-item active' : 'nav-item'} onClick={() => { setView(item.id); setMobileOpen(false); }}>
            {item.icon}<span>{item.label}</span>
          </button>
        ))}
        <button className="nav-item" onClick={() => { setSettingsOpen(true); setMobileOpen(false); }}>
          <Settings size={17}/><span>Settings</span>
        </button>
      </div>
      <div className="sidebar-status">
        <div className="status-head"><i className={vision.status === 'local' || vision.status === 'cloud' ? 'dot on' : 'dot'}/> <span>NOVA Status</span></div>
        <span>{vision.status === 'local' || vision.status === 'cloud' ? 'Online' : 'Idle'}</span>
        <svg className="sparkline" viewBox="0 0 120 32" preserveAspectRatio="none">
          <polyline points="0,24 12,18 24,22 36,10 48,16 60,6 72,14 84,9 96,18 108,8 120,14" fill="none" stroke="var(--brand-2)" strokeWidth="2"/>
        </svg>
      </div>
      <button className="sidebar-user" onClick={() => { setSettingsOpen(true); setMobileOpen(false); }}>
        <div className="avatar-circle"><User size={15}/></div>
        <span>NOVA User</span>
      </button>
    </nav>

    <div className="content">
      <div className="content-top">
        <button className="menu-btn" onClick={() => {
          if (window.innerWidth <= 720) setMobileOpen(true);
          else setDesktopCollapsed(!desktopCollapsed);
        }}>
          <Menu size={20}/>
        </button>
        <div className="topbar-greeting">
          <b>{view === 'home' ? 'Good day, there! 👋' : navItems.find(n => n.id === view)?.label}</b>
          <span>How can I assist you today?</span>
        </div>
        
        <div className="content-top-actions">
          <button title="Notifications" onClick={() => { setNotifsOpen(true); setUnreadNotifs(0); }} className="bell-btn">
            <Bell size={17}/>
            {unreadNotifs > 0 && <span className="badge">{unreadNotifs}</span>}
          </button>
          <button title="Theme" onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))}>{themeIcon}</button>
          <button title="Settings" className="avatar-circle small" onClick={() => setSettingsOpen(true)}><User size={15}/></button>
        </div>
      </div>

      <div className="content-body">
        {view === 'home' && <div className="home-view">
          <div className="hero-card">
            <div className="hero-text">
              <h1>NOVA AI</h1>
              <b>Your Intelligent Assistant</b>
              <p>Voice. Vision. Automation. All in one.</p>
              <button className="primary hero-cta" onClick={toggleMic}>{listening ? 'Listening…' : 'Start Interaction'} <Mic size={15}/></button>
            </div>
            <NovaFace size={140} state={listening ? 'listening' : typing ? 'thinking' : 'idle'} className="hero-orb"/>
          </div>

          <div className="home-grid-3">
            <div className="panel">
              <div className="section-title">Quick access</div>
              <div className="quick-grid">
                <button className="quick-card" onClick={() => setView('vision')}><Eye size={20}/><b>Vision Mode</b></button>
                <button className="quick-card" onClick={toggleMic}><Mic size={20}/><b>Voice Chat</b></button>
                <button className="quick-card" onClick={() => setView('devices')}><Smartphone size={20}/><b>Device Control</b></button>
                <button className="quick-card" onClick={() => setView('aitools')}><Wand2 size={20}/><b>AI Tools</b></button>
              </div>
            </div>

            <div className="panel">
              <div className="section-title">System overview</div>
              <div className="ring-row">
                <Ring percent={settings.confidenceThreshold * 100} color="#60a5fa" label="Vision confidence" value={`${Math.round(settings.confidenceThreshold * 100)}%`}/>
                <Ring percent={settings.voiceEnabled ? 100 : 0} color="#34d399" label="Voice replies" value={settings.voiceEnabled ? 'On' : 'Off'}/>
                <Ring percent={batteryLevel ?? 0} color="#f472b6" label="Battery" value={batteryLevel !== null ? `${batteryLevel}%` : '—'}/>
              </div>
              <div className="bar-row">
                <span>Saved chats</span><b>{history.length} / 50</b>
              </div>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.min(100, (history.length / 50) * 100)}%` }}/></div>
            </div>

            <div className="panel ai-response-panel">
              <div className="section-title">AI response</div>
              <div className="ai-response-head"><NovaFace size={34} state={typing ? 'thinking' : 'idle'} className="mini-orb"/><div><b>Hello! 👋</b><span>{provider}</span></div></div>
              <p className="panel-text">{lastModelMessage ? lastModelMessage.text.slice(0, 140) : "I'm NOVA, your AI assistant. Ask me anything, or show me a sign."}</p>
            </div>
          </div>
        </div>}

        {view === 'vision' && <div className="vision-view">
          <CameraView videoRef={vision.videoRef} enabled={settings.visionEnabled} status={vision.status} lastDetection={vision.lastDetection} onToggle={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} settings={settings}/>
          {settings.visionEnabled && <div className="learn-row">
            <input value={learnLabel} onChange={e => setLearnLabel(e.target.value.toUpperCase())} placeholder="Sign label, e.g. A" maxLength={16}/>
            <button className="secondary" onClick={() => { if (learnLabel.trim()) { learnSign(learnLabel.trim()); setLearnLabel(''); } }}>Teach this sign</button>
          </div>}
        </div>}

        {view === 'chat' && <div className="chat-view">
          <div className="chat-head">
            <div><b>Assistant</b><span>{AI_MODE_INFO[aiMode].label}{aiMode !== 'assistant' ? ' mode' : ' • Sign language companion'}</span></div>
            <div className="chat-head-actions"><button onClick={() => setHistoryOpen(true)}><History size={14}/> History</button><button onClick={startNewChat}><Plus size={14}/> New chat</button></div>
          </div>
          <div className="messages">
            {messages.map(m => <div key={m.id} className={`message ${m.role}`}>
              <div className="avatar">{m.role === 'user' ? <User size={15}/> : <Bot size={15}/>}</div>
              <div className="bubble">
                <div>{m.text}</div>
                {/* UPDATED: Download button rendering directly beneath images */}
                {m.imageUrl && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
                    <img src={m.imageUrl} alt="Generated" className="chat-image"/>
                    <button className="secondary" style={{ padding: '6px', fontSize: '12px', alignSelf: 'flex-start' }} onClick={() => downloadMedia(m.imageUrl!, `NOVA_Image_${Date.now()}.png`)}>
                      💾 Download Image
                    </button>
                  </div>
                )}
                {/* ADDED: Video Player rendering directly beneath the image logic */}
                {(m as any).videoUrl && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
                    <video src={(m as any).videoUrl} controls autoPlay loop className="chat-image" style={{ maxWidth: '100%', borderRadius: '8px', backgroundColor: '#000' }}/>
                    <button className="secondary" style={{ padding: '6px', fontSize: '12px', alignSelf: 'flex-start' }} onClick={() => downloadMedia((m as any).videoUrl, `NOVA_Video_${Date.now()}.mp4`)}>
                      💾 Download Video
                    </button>
                  </div>
                )}
                {m.role === 'model' && <button className="copy" onClick={() => navigator.clipboard?.writeText(m.text)}><Copy size={13}/></button>}
              </div>
            </div>)}
            {typing && <div className="message model"><div className="avatar"><Bot size={15}/></div><div className="bubble dots">● ● ●</div></div>}
            <div ref={chatEnd}/>
          </div>
        </div>}

        {view === 'aitools' && <div className="aitools-view">
          <div className="panel">
            <div className="section-title">Choose how NOVA should help</div>
            <div className="quick-grid mode-grid">
              <button className={aiMode === 'thinking' ? 'quick-card active' : 'quick-card'} onClick={() => switchAiMode('thinking')}>
                <BrainCircuit size={20}/><b>Deep Thinking</b><span>Thorough, step-by-step reasoning</span>
              </button>
              <button className={aiMode === 'image' ? 'quick-card active' : 'quick-card'} onClick={() => switchAiMode('image')}>
                <ImageIcon size={20}/><b>Image Generation</b><span>Describe it, NOVA draws it</span>
              </button>
              {/* ADDED: Video Generation Button */}
              <button className={aiMode === 'video' ? 'quick-card active' : 'quick-card'} onClick={() => switchAiMode('video')}>
                <Video size={20}/><b>Video Generation</b><span>Generate short videos</span>
              </button>
              <button className={aiMode === 'coder' ? 'quick-card active' : 'quick-card'} onClick={() => switchAiMode('coder')}>
                <Code2 size={20}/><b>Coder X</b><span>Complete, paste-ready code</span>
              </button>
              <button className={aiMode === 'assistant' ? 'quick-card active' : 'quick-card'} onClick={() => switchAiMode('assistant')}>
                <UserCog size={20}/><b>Personal Assistant</b><span>General help &amp; device control</span>
              </button>
            </div>
          </div>
          <div className="panel">
            <div className="section-title">Active mode</div>
            <div className="bar-row"><span>Mode</span><b>{AI_MODE_INFO[aiMode].label}</b></div>
            <p className="panel-text">{AI_MODE_INFO[aiMode].blurb}</p>
            <div className="bar-row"><span>Chat provider</span><b>{settings.aiProvider}</b></div>
            <button className="secondary" onClick={() => setSettingsOpen(true)}><Settings size={14}/> Change provider in Settings</button>
          </div>
        </div>}

        {view === 'devices' && <div className="devices-view">
          <div className="panel">
            <div className="section-title">Device bridge</div>
            <p className="panel-text">{bridgeConnected ? 'Connected — running inside the NOVA Android app. These buttons control your device directly.' : 'Not connected — you are in a normal browser tab. These buttons only work inside the packaged NOVA Android app.'}</p>
            <div className="quick-grid">
              <button className="quick-card" onClick={() => send('enable overlay')}><Sparkles size={20}/><b>Overlay bubble</b></button>
              <button className="quick-card" onClick={() => send('open accessibility')}><Settings size={20}/><b>Accessibility</b></button>
              <button className="quick-card" onClick={() => send('go home')}><Home size={20}/><b>Go home</b></button>
              <button className="quick-card" onClick={() => send('go back')}><Hand size={20}/><b>Go back</b></button>
            </div>
          </div>
        </div>}
      </div>

      <div className="composer-dock">
        <div className="active-mode-indicator">
          {aiMode === 'thinking' && <BrainCircuit size={12}/>}
          {aiMode === 'image' && <ImageIcon size={12}/>}
          {/* ADDED: Video icon for active mode indicator */}
          {aiMode === 'video' && <Video size={12}/>}
          {aiMode === 'coder' && <Code2 size={12}/>}
          {aiMode === 'assistant' && <UserCog size={12}/>}
          <span>{AI_MODE_INFO[aiMode].label}</span>
        </div>

        {attachments.length > 0 && <div className="attachments-row">{attachments.map((a, i) => <span key={i} className="attachment-chip">{a.name}<button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))}><X size={12}/></button></span>)}</div>}
        <input ref={fileInputRef} type="file" multiple hidden onChange={onFilePicked} />
        <div className="composer">
          <button className={listening ? 'active mic' : 'mic'} onClick={toggleMic}>{listening ? <MicOff/> : <Mic/>}</button>
          <button className="mic" title="Attach file" onClick={() => fileInputRef.current?.click()}><Paperclip/></button>
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()} placeholder="Ask me anything…"/>
          <button className="send" onClick={() => send()} disabled={(!input.trim() && attachments.length === 0) || typing}><Send/></button>
        </div>
      </div>
    </div>

    {settingsOpen && <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)}/>}
    
    {notifsOpen && <div className="modal-backdrop"><section className="history-panel">
      <div className="settings-head"><div><b>Notifications</b><span>System alerts &amp; mode tracking</span></div><button onClick={() => setNotifsOpen(false)}><X/></button></div>
      {notifications.length === 0 && <div className="history-empty">No new notifications.</div>}
      {notifications.map(n => <div className="history-item" key={n.id}>
        <div className="load" style={{cursor: 'default'}}>{n.text}<small>{new Date(n.time).toLocaleTimeString()}</small></div>
      </div>)}
    </section></div>}

    {historyOpen && <div className="modal-backdrop"><section className="history-panel">
      <div className="settings-head"><div><b>Chat history</b><span>{history.length} saved conversation{history.length === 1 ? '' : 's'}</span></div><button onClick={() => setHistoryOpen(false)}><X/></button></div>
      {history.length === 0 && <div className="history-empty">No saved conversations yet.</div>}
      {history.sort((a, b) => b.updatedAt - a.updatedAt).map(s => <div className="history-item" key={s.id}>
        <button className="load" onClick={() => { loadSession(s.id); setView('chat'); }}>{s.title}<small>{new Date(s.updatedAt).toLocaleString()}</small></button>
        <button className="del" onClick={() => deleteSession(s.id)}><X size={14}/></button>
      </div>)}
      <div className="settings-foot"><button className="primary" onClick={() => { startNewChat(); setView('chat'); }}><Plus size={16}/> New chat</button></div>
    </section></div>}
  </div>;
}

declare global { interface Window { SpeechRecognition: any; webkitSpeechRecognition: any; } }
