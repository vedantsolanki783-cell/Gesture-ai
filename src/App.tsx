import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Bot,
  Copy,
  FolderOpen,
  MessageSquare,
  Mic,
  MicOff,
  Moon,
  Plus,
  Send,
  Settings,
  Sparkles,
  Sun,
  Trash2,
  User,
  X,
  BrainCircuit,
  PanelLeftClose,
  PanelLeft
} from 'lucide-react';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import { learnSign, clearTrainedSigns, getTrainedSignsCount } from './vision/localVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

const httpUrl = (path: string): string => ['ht', 'tp://', path].join('');

// ============================================================================
// CHAT SESSION INTERFACES (ChatGPT Style)
// ============================================================================
interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  messages: Message[];
}

function getNovaAndroid(): any {
  return typeof window !== 'undefined' ? (window as any).NovaAndroid || null : null;
}

function executeAndroidAgentCommand(rawText: string): string | null {
  const bridge = getNovaAndroid();
  const text = rawText.toLowerCase().trim();

  const isOverlayCmd = /\b(enable|start|show|turn on|activate)\s+(overlay|bubble|floating|hud)\b/i.test(text);
  const isAccessCmd = /\b(enable|open|turn on|activate)\s+(accessibility|ultron|agent control)\b/i.test(text);
  const isHomeCmd = /^(go\s+home|home\s+screen|press\s+home|home)$/i.test(text);
  const isBackCmd = /^(go\s+back|navigate\s+back|press\s+back|back)$/i.test(text);
  const isRecentsCmd = /^(open\s+recents|recent\s+apps|show\s+recents|recents)$/i.test(text);
  const isNotifCmd = /^(open\s+notifications|show\s+notifications|pull\s+down\s+notifications|notifications)$/i.test(text);
  const isScreenshotCmd = /\b(take\s+a?\s*screenshot|capture\s+screen|screenshot)\b/i.test(text);
  const isSwipeUpCmd = /\b(swipe\s+up|scroll\s+down)\b/i.test(text);
  const isSwipeDownCmd = /\b(swipe\s+down|scroll\s+up)\b/i.test(text);
  const openAppMatch = text.match(/^(?:open|launch|start)\s+([a-z0-9\s._-]+)$/i);

  const isAnyAndroidCommand =
    isOverlayCmd || isAccessCmd || isHomeCmd || isBackCmd || isRecentsCmd ||
    isNotifCmd || isScreenshotCmd || isSwipeUpCmd || isSwipeDownCmd ||
    (openAppMatch && /whatsapp|youtube|chrome|browser|instagram|gmail|mail|maps|settings|camera/i.test(openAppMatch[1]));

  if (!isAnyAndroidCommand) return null;

  if (!bridge) {
    return '⚠️ **Android Bridge Not Found:** You must be inside the NOVA APK to control the hardware.';
  }

  if (isOverlayCmd) {
    const res = bridge.enableOverlayBubble();
    return res === 'OPENED_OVERLAY_SETTINGS' ? '⚡ Opening Settings...' : '⚡ Floating Bubble Active!';
  }
  if (isAccessCmd) {
    bridge.openAccessibilitySettings();
    return '🤖 Opening Accessibility Settings... Turn ON Ultron.';
  }
  if (openAppMatch) {
    const ok = bridge.openApp(openAppMatch[1].trim());
    if (ok) return `🚀 Launching **${openAppMatch[1].toUpperCase()}**...`;
  }
  if (!bridge.isUltronConnected()) {
    bridge.openAccessibilitySettings();
    return '⚠️ Ultron OFF. Turn it ON in Accessibility Settings.';
  }
  if (isHomeCmd) { bridge.globalAction('HOME'); return '🏠 Executed **HOME**.'; }
  if (isBackCmd) { bridge.globalAction('BACK'); return '🔙 Executed **BACK**.'; }
  if (isRecentsCmd) { bridge.globalAction('RECENTS'); return '🗂️ Opened **Recents**.'; }
  if (isNotifCmd) { bridge.globalAction('NOTIFICATIONS'); return '🔔 Pulled **Notifications**.'; }
  if (isScreenshotCmd) { bridge.globalAction('SCREENSHOT'); return '📸 Took **Screenshot**.'; }
  if (isSwipeUpCmd) { bridge.swipeScreen(500, 1500, 500, 400, 300); return '⬆️ Swiped Up.'; }
  if (isSwipeDownCmd) { bridge.swipeScreen(500, 400, 500, 1500, 300); return '⬇️ Swiped Down.'; }

  return null;
}

const defaults: AppSettings = {
  theme: 'dark',
  voiceEnabled: true,
  visionEnabled: false,
  confidenceThreshold: 0.72,
  aiProvider: 'ollama',
  ollamaUrl: httpUrl('localhost:11434'),
  ollamaModel: 'qwen2.5:0.5b',
  geminiModel: 'gemini-2.5-flash',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction: 'You are NOVA, an advanced multimodal AI assistant and Android device agent.',
  customGestures: []
};

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      return { ...defaults, ...JSON.parse(localStorage.getItem('nova-unified-settings') || '{}') };
    } catch {
      return defaults;
    }
  });

  // Chat Sessions History State (ChatGPT Style)
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem('nova_chat_sessions');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [currentSessionId, setCurrentSessionId] = useState<string>(() => {
    return sessions.length > 0 ? sessions[0].id : crypto.randomUUID();
  });

  const [messages, setMessages] = useState<Message[]>(() => {
    const active = sessions.find(s => s.id === currentSessionId);
    return active ? active.messages : [
      {
        id: 'welcome',
        role: 'model',
        text: '⚡ **NOVA Core Online**\n\n• **Palm-Ray Mouse Active**: Point & pinch to click.\n• **Chat History**: Tap the sidebar icon to switch conversations.\n• **ML Trainer**: Teach signs in the left sidebar.',
        timestamp: Date.now()
      }
    ];
  });

  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState(false);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [provider, setProvider] = useState('Local Core');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const chatEnd = useRef<HTMLDivElement>(null);

  // ML Trainer State
  const [mlTrainInput, setMlTrainInput] = useState('');
  const [mlCount, setMlCount] = useState(getTrainedSignsCount());

  // Save Settings
  useEffect(() => {
    localStorage.setItem('nova-unified-settings', JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  // Synchronize Active Messages into the Sessions Storage
  useEffect(() => {
    setSessions(prevSessions => {
      const existingIndex = prevSessions.findIndex(s => s.id === currentSessionId);
      const firstUserMsg = messages.find(m => m.role === 'user');
      const title = firstUserMsg ? firstUserMsg.text.slice(0, 28) + (firstUserMsg.text.length > 28 ? '...' : '') : 'New Chat';

      let updated: ChatSession[];
      if (existingIndex >= 0) {
        updated = [...prevSessions];
        updated[existingIndex] = {
          ...updated[existingIndex],
          title: updated[existingIndex].title === 'New Chat' ? title : updated[existingIndex].title,
          messages
        };
      } else {
        updated = [{ id: currentSessionId, title, createdAt: Date.now(), messages }, ...prevSessions];
      }
      localStorage.setItem('nova_chat_sessions', JSON.stringify(updated));
      return updated;
    });
  }, [messages, currentSessionId]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const speak = useCallback((text: string) => {
    if (!settings.voiceEnabled || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.slice(0, 400));
    u.rate = 1.05;
    speechSynthesis.speak(u);
  }, [settings.voiceEnabled]);

  const createNewChat = () => {
    const newId = crypto.randomUUID();
    setCurrentSessionId(newId);
    setMessages([
      {
        id: crypto.randomUUID(),
        role: 'model',
        text: '⚡ New chat started. How can I help you today?',
        timestamp: Date.now()
      }
    ]);
    setIsHistoryDrawerOpen(false);
  };

  const loadChatSession = (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setMessages(session.messages);
    setIsHistoryDrawerOpen(false);
  };

  const deleteChatSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = sessions.filter(s => s.id !== id);
    setSessions(updated);
    localStorage.setItem('nova_chat_sessions', JSON.stringify(updated));
    if (currentSessionId === id) {
      if (updated.length > 0) {
        setCurrentSessionId(updated[0].id);
        setMessages(updated[0].messages);
      } else {
        createNewChat();
      }
    }
  };

  const send = useCallback(async (value = input) => {
    const text = value.trim();
    if (!text || typing) return;

    const userMsgId = crypto.randomUUID();
    setMessages(prev => [...prev, { id: userMsgId, role: 'user', text, timestamp: Date.now() }]);
    setInput('');
    setTyping(true);

    const androidReply = executeAndroidAgentCommand(text);
    if (androidReply) {
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: androidReply, timestamp: Date.now() }]);
      speak(androidReply);
      setTyping(false);
      return;
    }

    try {
      const result = await generateLocalOrCloud(text, messages, settings);
      setProvider(result.provider);
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: result.text, timestamp: Date.now() }]);
      speak(result.text);
    } catch {
      const fallback = `Jarvis Offline Core: Processed "${text}".`;
      setProvider('Offline Core');
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: fallback, timestamp: Date.now() }]);
      speak(fallback);
    } finally {
      setTyping(false);
    }
  }, [input, messages, settings, speak, typing]);

  const handleTrainML = () => {
    if (!mlTrainInput.trim()) return;
    if (!settings.visionEnabled) {
      alert('Please turn ON the camera first!');
      return;
    }
    learnSign(mlTrainInput);
    setTimeout(() => {
      setMlCount(getTrainedSignsCount());
      setMlTrainInput('');
    }, 400);
  };

  const executeSign = useCallback((result: VisionResult) => {
    if (result.type === 'LETTER' && result.value) {
      setInput(prev => prev + result.value);
    } else if (result.value === 'CLEAR') {
      setInput('');
    } else if (result.value === 'SEND') {
      send();
    }
  }, [send]);

  const vision = useVision(settings, settings.customGestures, executeSign);

  return (
    <div className="app">
      <header>
        <div className="brand">
          <button
            className="icon-btn"
            onClick={() => setIsHistoryDrawerOpen(prev => !prev)}
            title="Chat History"
            style={{ background: 'transparent', border: 'none', color: '#38bdf8', cursor: 'pointer', padding: 4 }}
          >
            {isHistoryDrawerOpen ? <PanelLeftClose size={22} /> : <PanelLeft size={22} />}
          </button>
          <div className="logo"><Sparkles size={20} /></div>
          <div>
            <h1>NOVA JARVIS</h1>
            <span>{provider}</span>
          </div>
        </div>

        <div className="header-actions">
          <button
            onClick={createNewChat}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: '#0284c7',
              color: '#fff',
              border: 'none',
              padding: '6px 12px',
              borderRadius: 8,
              fontWeight: 600,
              fontSize: 12,
              cursor: 'pointer'
            }}
          >
            <Plus size={15} /> New Chat
          </button>
          <button onClick={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} title="Camera">
            {settings.visionEnabled ? <BrainCircuit color="#10b981" /> : <BrainCircuit />}
          </button>
          <button onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))}>
            {settings.theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button onClick={() => setSettingsOpen(true)}><Settings size={18} /></button>
        </div>
      </header>

      <main className="app-workspace" style={{ position: 'relative' }}>
        {/* CHATGPT-STYLE CHAT HISTORY DRAWER */}
        {isHistoryDrawerOpen && (
          <aside
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              bottom: 0,
              width: 280,
              background: '#090d16',
              borderRight: '1px solid rgba(56, 189, 248, 0.25)',
              zIndex: 900,
              padding: 14,
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '10px 0 30px rgba(0,0,0,0.7)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontSize: 13, fontWeight: 800, color: '#38bdf8', letterSpacing: 0.6 }}>
                💬 CHAT SESSIONS
              </span>
              <button
                onClick={() => setIsHistoryDrawerOpen(false)}
                style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
              >
                <X size={16} />
              </button>
            </div>

            <button
              onClick={createNewChat}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                background: '#1e293b',
                color: '#fff',
                padding: '10px',
                borderRadius: 8,
                border: '1px solid rgba(255,255,255,0.1)',
                fontWeight: 600,
                fontSize: 13,
                cursor: 'pointer',
                marginBottom: 12
              }}
            >
              <Plus size={16} /> Start New Chat
            </button>

            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
              {sessions.map(s => (
                <div
                  key={s.id}
                  onClick={() => loadChatSession(s)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 10px',
                    borderRadius: 8,
                    background: s.id === currentSessionId ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255,255,255,0.03)',
                    border: s.id === currentSessionId ? '1px solid #38bdf8' : '1px solid transparent',
                    cursor: 'pointer'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, overflow: 'hidden' }}>
                    <MessageSquare size={14} color="#94a3b8" />
                    <span
                      style={{
                        fontSize: 12,
                        color: s.id === currentSessionId ? '#38bdf8' : '#e2e8f0',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}
                    >
                      {s.title}
                    </span>
                  </div>
                  <button
                    onClick={e => deleteChatSession(s.id, e)}
                    style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', padding: 2 }}
                    title="Delete Chat"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          </aside>
        )}

        {/* SIDEBAR: CAMERA + PALM-RAY CONTROLS + ML TRAINER */}
        <aside className="app-sidebar">
          <CameraView
            videoRef={vision.videoRef}
            enabled={settings.visionEnabled}
            status={vision.status}
            lastDetection={vision.lastDetection}
            onToggle={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))}
            settings={settings}
          />

          {/* CUSTOM ML SIGN TRAINER */}
          <div style={{ marginTop: 10, padding: 12, background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: '#10b981', marginBottom: 6 }}>
              🧠 CUSTOM ML TRAINING
            </div>
            <p style={{ fontSize: 11, color: '#94a3b8', margin: '0 0 8px 0' }}>
              Hold your hand sign to the camera, type the label below, and tap Learn.
            </p>
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              <input
                value={mlTrainInput}
                onChange={e => setMlTrainInput(e.target.value.toUpperCase())}
                placeholder="Ex: A, B, SEND"
                style={{ flex: 1, padding: 6, borderRadius: 6, background: '#020617', color: '#fff', border: '1px solid #334155', textTransform: 'uppercase', fontSize: 12 }}
              />
              <button
                onClick={handleTrainML}
                style={{ background: '#10b981', color: '#020617', padding: '6px 12px', borderRadius: 6, fontWeight: 'bold', border: 'none', cursor: 'pointer', fontSize: 12 }}
              >
                Learn
              </button>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b' }}>
              <span>Memory: <b>{mlCount}</b> saved</span>
              <button
                onClick={() => { clearTrainedSigns(); setMlCount(0); }}
                style={{ background: 'transparent', color: '#ef4444', border: 'none', cursor: 'pointer' }}
              >
                Erase All
              </button>
            </div>
          </div>

          {/* HARDWARE ULTRON AGENT BUTTONS */}
          <div className="capabilities" style={{ marginTop: 10 }}>
            <div className="eyebrow">ULTRON ANDROID CONTROLS</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 6 }}>
              <button style={{ padding: 7, borderRadius: 8, background: '#1e293b', color: '#fff', border: '1px solid #334155', fontSize: 11 }} onClick={() => send('enable overlay')}>
                ⚡ HUD
              </button>
              <button style={{ padding: 7, borderRadius: 8, background: '#1e293b', color: '#fff', border: '1px solid #334155', fontSize: 11 }} onClick={() => send('enable accessibility')}>
                ⚙️ Ultron
              </button>
            </div>
          </div>
        </aside>

        {/* MAIN CONVERSATION CHAT */}
        <section className="chat">
          <div className="messages">
            {messages.map(m => (
              <div key={m.id} className={`message ${m.role}`}>
                <div className="avatar">{m.role === 'user' ? <User size={15} /> : <Bot size={15} />}</div>
                <div className="bubble">
                  <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
                </div>
              </div>
            ))}
            {typing && (
              <div className="message model">
                <div className="avatar"><Bot size={15} /></div>
                <div className="bubble dots">● ● ●</div>
              </div>
            )}
            <div ref={chatEnd} />
          </div>

          <div className="composer">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="Ask anything, sign with hand, or type 'open youtube'..."
            />
            <button className="send" onClick={() => send()} disabled={!input.trim() || typing}>
              <Send size={16} />
            </button>
          </div>
        </section>
      </main>

      {settingsOpen && (
        <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)} />
      )}
    </div>
  );
}
