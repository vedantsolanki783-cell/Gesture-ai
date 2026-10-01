import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Bot,
  User,
  Send,
  Settings,
  CameraOff,
  Sun,
  Moon,
  Trash2,
  Mic,
  Paperclip,
  Copy,
  Zap,
  Camera,
  Plus,
  Menu,
  X,
  Sparkles,
  Mouse,
  Eye,
  MessageSquare,
  ChevronDown
} from 'lucide-react';

import { Message, AppSettings, VisionResponse } from './types';
import { generateLocalOrCloud } from './services/aiRouter';
import SettingsPanel from './components/SettingsPanel';
import CameraView from './components/CameraView';
import { useVision } from './hooks/useVision';
import { learnSign } from './vision/localVision';

import './styles.css';

const defaults: AppSettings = {
  theme: 'dark',
  voiceEnabled: true,
  visionEnabled: false,
  confidenceThreshold: 0.72,

  aiProvider: 'ollama',
  ollamaUrl: 'http://localhost:11434',
  ollamaModel: 'qwen2.5:0.5b',

  geminiModel: 'gemini-2.5-flash',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',

  systemInstruction: 'You are NOVA. Answer concisely.',
  customGestures: []
};

function executeAndroidAgentCommand(rawText: string): string | null {
  const bridge =
    typeof window !== 'undefined'
      ? (window as any).NovaAndroid
      : null;

  const text = rawText.toLowerCase().trim();

  const isOverlay =
    /\b(enable|start|show|turn on)\s+(overlay|bubble|hud)\b/i.test(text);

  const isUltron =
    /\b(enable|open|turn on)\s+(accessibility|ultron)\b/i.test(text);

  const isHome = /^(go\s+home|home)$/i.test(text);

  const isBack = /^(go\s+back|back)$/i.test(text);

  const openAppMatch = text.match(
    /^(?:open|launch)\s+([a-z0-9\s._-]+)$/i
  );

  if (
    !bridge &&
    (isOverlay || isUltron || isHome || isBack || openAppMatch)
  ) {
    return '⚠️ Android Bridge Not Found.';
  }

  if (isOverlay) {
    return bridge.enableOverlayBubble() === 'OPENED_OVERLAY_SETTINGS'
      ? '⚡ Opening Settings...'
      : '⚡ Floating Bubble Active!';
  }

  if (isUltron) {
    bridge.openAccessibilitySettings();
    return '🤖 Opening Accessibility Settings...';
  }

  if (
    openAppMatch &&
    bridge.openApp(openAppMatch[1].trim())
  ) {
    return `🚀 Launching ${openAppMatch[1].toUpperCase()}...`;
  }

  if (
    bridge &&
    !bridge.isUltronConnected()
  ) {
    bridge.openAccessibilitySettings();
    return '⚠️ Ultron OFF. Turn it ON.';
  }

  if (isHome) {
    bridge.globalAction('HOME');
    return '🏠 Executed HOME.';
  }

  if (isBack) {
    bridge.globalAction('BACK');
    return '🔙 Executed BACK.';
  }

  return null;
}

export default function App() {
  const [settings, setSettings] =
    useState<AppSettings>(() => {
      try {
        return {
          ...defaults,
          ...JSON.parse(
            localStorage.getItem('nova_settings') || '{}'
          )
        };
      } catch {
        return defaults;
      }
    });

  const [messages, setMessages] = useState<Message[]>([]);

  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [mlInput, setMlInput] = useState('');

  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem(
      'nova_settings',
      JSON.stringify(settings)
    );

    document.documentElement.classList.toggle(
      'dark',
      settings.theme === 'dark'
    );
  }, [settings]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({
      behavior: 'smooth'
    });
  }, [messages, typing]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const newChat = () => {
    setMessages([]);
    setInput('');
    setSidebarOpen(false);
  };

  const send = useCallback(
    async (val = input) => {
      const text = val.trim();

      if (!text || typing) return;

      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content: text,
        timestamp: Date.now()
      };

      setMessages(prev => [
        ...prev,
        userMessage
      ]);

      setInput('');
      setTyping(true);

      const androidReply =
        executeAndroidAgentCommand(text);

      if (androidReply) {
        setMessages(prev => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'model',
            content: androidReply,
            timestamp: Date.now()
          }
        ]);

        setTyping(false);
        return;
      }

      try {
        const res = await generateLocalOrCloud(
          text,
          messages,
          settings
        );

        setMessages(prev => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'model',
            content: res.text,
            timestamp: Date.now()
          }
        ]);
      } catch {
        setMessages(prev => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'model',
            content:
              '⚠️ Local AI Offline. Open Termux and run `ollama serve`.',
            timestamp: Date.now()
          }
        ]);
      } finally {
        setTyping(false);
      }
    },
    [input, messages, settings, typing]
  );

  const handleGestureDetected =
    useCallback(
      (result: VisionResponse) => {
        if (result.type === 'LETTER') {
          setInput(prev => prev + result.value);
        } else if (result.value === 'CLEAR') {
          setInput('');
        } else if (result.value === 'SEND') {
          send();
        }
      },
      [send]
    );

  const vision = useVision(
    settings,
    settings.customGestures,
    handleGestureDetected
  );

  const suggestions = [
    {
      icon: Sparkles,
      title: 'Explain something',
      text: 'Explain a difficult topic simply'
    },
    {
      icon: MessageSquare,
      title: 'Help me write',
      text: 'Write or improve something for me'
    },
    {
      icon: Mouse,
      title: 'Control my device',
      text: 'Open an app or perform an action'
    },
    {
      icon: Eye,
      title: 'Use vision',
      text: 'Recognize my hand gestures'
    }
  ];

  return (
    <div className="nova-app">

      {/* MOBILE OVERLAY */}
      {sidebarOpen && (
        <div
          className="mobile-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* SIDEBAR */}
      <aside
        className={`nova-sidebar ${
          sidebarOpen ? 'sidebar-open' : ''
        }`}
      >

        <div className="sidebar-top">

          <div className="brand-row">

            <div className="nova-logo">
              <Zap size={18} fill="currentColor" />
            </div>

            <div className="brand-text">
              <strong>NOVA</strong>
              <span>GESTURE AI</span>
            </div>

            <button
              className="mobile-close"
              onClick={() => setSidebarOpen(false)}
            >
              <X size={20} />
            </button>

          </div>

          <button
            className="new-chat-btn"
            onClick={newChat}
          >
            <Plus size={18} />
            <span>New chat</span>
          </button>

          <div className="sidebar-section">

            <div className="sidebar-label">
              WORKSPACE
            </div>

            <button className="sidebar-item active">
              <MessageSquare size={17} />
              <span>Assistant</span>
            </button>

            <button className="sidebar-item">
              <Mouse size={17} />
              <span>Gesture Mouse</span>
              <span className="status-dot" />
            </button>

            <button className="sidebar-item">
              <Eye size={17} />
              <span>Vision</span>

              <span
                className={
                  settings.visionEnabled
                    ? 'mini-status on'
                    : 'mini-status'
                }
              >
                {settings.visionEnabled
                  ? 'ON'
                  : 'OFF'}
              </span>
            </button>

          </div>

        </div>

        <div className="sidebar-bottom">

          <div className="local-card">

            <div className="local-icon">
              <Zap size={14} />
            </div>

            <div>
              <strong>Local AI</strong>
              <span>
                {settings.ollamaModel}
              </span>
            </div>

            <span className="online-dot" />

          </div>

          <button
            className="sidebar-item"
            onClick={() =>
              setSettingsOpen(true)
            }
          >
            <Settings size={17} />
            <span>Settings</span>
          </button>

        </div>

      </aside>

      {/* MAIN */}
      <main className="nova-main">

        {/* TOP BAR */}
        <header className="nova-topbar">

          <button
            className="menu-button"
            onClick={() =>
              setSidebarOpen(true)
            }
          >
            <Menu size={20} />
          </button>

          <div className="top-title">

            <div className="top-title-icon">
              <Bot size={17} />
            </div>

            <div>
              <strong>NOVA Assistant</strong>

              <span>
                {settings.ollamaModel}
              </span>
            </div>

          </div>

          <div className="top-actions">

            <button
              className="icon-button"
              onClick={() =>
                setSettings(s => ({
                  ...s,
                  visionEnabled:
                    !s.visionEnabled
                }))
              }
              title="Vision"
            >
              {settings.visionEnabled ? (
                <Camera
                  size={18}
                  className="green"
                />
              ) : (
                <CameraOff size={18} />
              )}
            </button>

            <button
              className="icon-button"
              onClick={() =>
                setSettings(s => ({
                  ...s,
                  theme:
                    s.theme === 'dark'
                      ? 'light'
                      : 'dark'
                }))
              }
            >
              {settings.theme === 'dark' ? (
                <Sun size={18} />
              ) : (
                <Moon size={18} />
              )}
            </button>

            <button
              className="icon-button"
              onClick={() =>
                setSettingsOpen(true)
              }
            >
              <Settings size={18} />
            </button>

          </div>

        </header>

        {/* WORKSPACE */}
        <div className="nova-workspace">

          {/* VISION PANEL */}
          {settings.visionEnabled && (
            <section className="vision-panel">

              <div className="vision-header">

                <div>
                  <strong>
                    <Camera size={15} />
                    Vision
                  </strong>

                  <span>
                    Gesture recognition active
                  </span>
                </div>

                <button
                  className="vision-close"
                  onClick={() =>
                    setSettings(s => ({
                      ...s,
                      visionEnabled: false
                    }))
                  }
                >
                  <X size={16} />
                </button>

              </div>

              <div className="vision-camera">

                <CameraView
                  videoRef={vision.videoRef}
                  enabled={
                    settings.visionEnabled
                  }
                  status={vision.status}
                  lastDetection={
                    vision.lastDetection
                  }
                  onToggle={() =>
                    setSettings(s => ({
                      ...s,
                      visionEnabled:
                        !s.visionEnabled
                    }))
                  }
                  settings={settings}
                />

                <div className="gesture-learning">

                  <input
                    value={mlInput}
                    onChange={e =>
                      setMlInput(
                        e.target.value.toUpperCase()
                      )
                    }
                    placeholder="Sign A-Z"
                  />

                  <button
                    onClick={() => {
                      learnSign(mlInput);
                      setMlInput('');
                    }}
                  >
                    Learn
                  </button>

                </div>

              </div>

            </section>
          )}

          {/* CHAT */}
          <section className="chat-area">

            {messages.length === 0 ? (

              /* WELCOME */
              <div className="welcome-screen">

                <div className="welcome-logo">
                  <Zap
                    size={30}
                    fill="currentColor"
                  />
                </div>

                <h1>
                  How can I help you?
                </h1>

                <p>
                  NOVA is your local AI assistant
                  with gesture and device control.
                </p>

                <div className="suggestion-grid">

                  {suggestions.map(
                    suggestion => {
                      const Icon =
                        suggestion.icon;

                      return (
                        <button
                          key={suggestion.title}
                          className="suggestion-card"
                          onClick={() =>
                            setInput(
                              suggestion.text
                            )
                          }
                        >

                          <Icon size={18} />

                          <div>
                            <strong>
                              {suggestion.title}
                            </strong>

                            <span>
                              {suggestion.text}
                            </span>
                          </div>

                        </button>
                      );
                    }
                  )}

                </div>

              </div>

            ) : (

              /* MESSAGES */
              <div className="messages">

                {messages.map(message => (

                  <div
                    key={message.id}
                    className={`message-row ${
                      message.role === 'user'
                        ? 'user-message'
                        : 'assistant-message'
                    }`}
                  >

                    <div className="message-avatar">

                      {message.role ===
                      'user' ? (
                        <User size={16} />
                      ) : (
                        <Zap
                          size={16}
                          fill="currentColor"
                        />
                      )}

                    </div>

                    <div className="message-content">

                      <div className="message-name">
                        {message.role ===
                        'user'
                          ? 'You'
                          : 'NOVA'}
                      </div>

                      <div className="message-text">
                        {message.content}
                      </div>

                      {message.role ===
                        'model' && (
                        <button
                          className="copy-button"
                          onClick={() =>
                            copyToClipboard(
                              message.content
                            )
                          }
                        >
                          <Copy size={13} />
                          Copy
                        </button>
                      )}

                    </div>

                  </div>

                ))}

                {typing && (
                  <div className="message-row assistant-message">

                    <div className="message-avatar">
                      <Zap
                        size={16}
                        fill="currentColor"
                      />
                    </div>

                    <div className="message-content">

                      <div className="message-name">
                        NOVA
                      </div>

                      <div className="typing">
                        <span />
                        <span />
                        <span />
                      </div>

                    </div>

                  </div>
                )}

                <div ref={chatEnd} />

              </div>

            )}

          </section>

          {/* COMPOSER */}
          <div className="composer-wrapper">

            <div className="composer">

              <button
                className="composer-icon"
              >
                <Paperclip size={19} />
              </button>

              <input
                value={input}
                onChange={e =>
                  setInput(e.target.value)
                }
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    send();
                  }
                }}
                placeholder="Message NOVA..."
              />

              <button
                className="composer-icon"
              >
                <Mic size={19} />
              </button>

              <button
                className={`send-button ${
                  input.trim()
                    ? 'send-active'
                    : ''
                }`}
                onClick={() => send()}
                disabled={
                  !input.trim() || typing
                }
              >
                <Send size={18} />
              </button>

            </div>

            <div className="composer-hint">

              <span>
                NOVA can make mistakes.
                Check important information.
              </span>

              <span className="gesture-hint">
                🖐 Gesture control
              </span>

            </div>

          </div>

        </div>

      </main>

      {/* SETTINGS */}
      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() =>
          setSettingsOpen(false)
        }
        settings={settings}
        onSave={setSettings}
      />

    </div>
  );
}
