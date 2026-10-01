import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Bot,
  Camera,
  CameraOff,
  Check,
  ChevronDown,
  Copy,
  FileText,
  Loader2,
  Mic,
  Paperclip,
  Send,
  Settings,
  Sun,
  Moon,
  Trash2,
  User,
  X,
  Zap,
} from 'lucide-react';
import type {
  AppSettings,
  Message,
  VisionResult,
} from './types';
import { generateLocalOrCloud } from './services/aiRouter';
import { SettingsPanel } from './components/SettingsPanel';
import { CameraView } from './components/CameraView';
import { useVision } from './hooks/useVision';
import { learnSign } from './vision/localVision';
import { parseUploadedFile } from './services/fileReader';
import type { MessageAttachment } from './services/hybridAI';

const defaults: AppSettings = {
  theme: 'dark',
  voiceEnabled: true,
  visionEnabled: false,
  confidenceThreshold: 0.72,
  aiProvider: 'ollama',
  ollamaUrl: 'http://localhost:11434',
  ollamaModel: 'qwen2.5:0.5b',
  geminiModel: 'gemini-2.5-flash',
  cloudFreeBaseUrl: 'https://api.groq.com/openai/v1',
  cloudFreeApiKey: '',
  cloudFreeModel: 'llama-3.3-70b-versatile',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction: 'You are NOVA. Answer concisely.',
  customGestures: [],
};

const WELCOME_MESSAGE = (): Message => ({
  id: 'welcome',
  role: 'model',
  text: 'Hello! How can I assist you today?',
  timestamp: Date.now(),
});

function loadSettings(): AppSettings {
  try {
    const saved = localStorage.getItem('nova_settings');
    return saved
      ? { ...defaults, ...JSON.parse(saved) }
      : defaults;
  } catch {
    return defaults;
  }
}

function executeAndroidAgentCommand(rawText: string): string | null {
  const bridge =
    typeof window !== 'undefined'
      ? (window as any).NovaAndroid
      : null;

  const text = rawText.toLowerCase().trim();

  const isOverlay =
    /\b(enable|start|show|turn on)\s+(overlay|bubble|hud)\b/i.test(
      text,
    );
  const isUltron =
    /\b(enable|open|turn on)\s+(accessibility|ultron)\b/i.test(
      text,
    );
  const isHome = /^(go\s+home|home)$/i.test(text);
  const isBack = /^(go\s+back|back)$/i.test(text);
  const openAppMatch = text.match(
    /^(?:open|launch)\s+([a-z0-9\s._-]+)$/i,
  );

  if (
    !bridge &&
    (isOverlay || isUltron || isHome || isBack || openAppMatch)
  ) {
    return '⚠️ Android Bridge Not Found.';
  }

  if (isOverlay) {
    return bridge.enableOverlayBubble() ===
      'OPENED_OVERLAY_SETTINGS'
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

  if (bridge && !bridge.isUltronConnected()) {
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

function getInitialsOrIcon(role: Message['role']) {
  return role === 'model' ? <Bot size={16} /> : <User size={16} />;
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [messages, setMessages] = useState<Message[]>([
    WELCOME_MESSAGE(),
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [mlInput, setMlInput] = useState('');
  const [attachments, setAttachments] = useState<MessageAttachment[]>([]);
  const [lastProvider, setLastProvider] = useState('Ready');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [attachmentBusy, setAttachmentBusy] = useState(false);

  const chatEnd = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    localStorage.setItem(
      'nova_settings',
      JSON.stringify(settings),
    );
    document.documentElement.classList.toggle(
      'dark',
      settings.theme === 'dark',
    );
  }, [settings]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const copyToClipboard = useCallback(
    async (text: string, id: string) => {
      try {
        await navigator.clipboard.writeText(text);
        setCopiedId(id);
        window.setTimeout(() => setCopiedId(null), 1200);
      } catch {
        // ignore
      }
    },
    [],
  );

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setAttachmentBusy(true);
    try {
      const parsed = await parseUploadedFile(file);
      setAttachments((previous) => [...previous, parsed]);
    } catch (error) {
      console.error('File read error:', error);
    } finally {
      setAttachmentBusy(false);
      event.target.value = '';
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments((previous) =>
      previous.filter((_, itemIndex) => itemIndex !== index),
    );
  };

  const send = useCallback(
    async (value = input) => {
      const text = value.trim();
      if ((!text && attachments.length === 0) || typing) return;

      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        text:
          text ||
          `Attached ${attachments.length} file${attachments.length > 1 ? 's' : ''}`,
        timestamp: Date.now(),
      };

      const historyForAI = messages;
      const currentAttachments = [...attachments];

      setMessages((previous) => [...previous, userMessage]);
      setInput('');
      setAttachments([]);
      setTyping(true);

      const androidReply = executeAndroidAgentCommand(text);

      if (androidReply) {
        setMessages((previous) => [
          ...previous,
          {
            id: crypto.randomUUID(),
            role: 'model',
            text: androidReply,
            timestamp: Date.now(),
          },
        ]);
        setLastProvider('Android Agent');
        setTyping(false);
        return;
      }

      try {
        const response = await generateLocalOrCloud(
          text,
          historyForAI,
          settings,
          currentAttachments,
        );

        setMessages((previous) => [
          ...previous,
          {
            id: crypto.randomUUID(),
            role: 'model',
            text: response.text,
            timestamp: Date.now(),
          },
        ]);
        setLastProvider(response.provider);
      } catch (error) {
        setMessages((previous) => [
          ...previous,
          {
            id: crypto.randomUUID(),
            role: 'model',
            text: `⚠️ NOVA could not reach the selected AI provider.\n\n${String(
              error,
            )}`,
            timestamp: Date.now(),
          },
        ]);
        setLastProvider('Error');
      } finally {
        setTyping(false);
      }
    },
    [attachments, input, messages, settings, typing],
  );

  const handleGestureDetected = useCallback(
    (result: VisionResult) => {
      if (result.type === 'LETTER') {
        setInput((previous) => previous + result.value);
      } else if (result.value === 'CLEAR') {
        setInput('');
      } else if (result.value === 'SEND') {
        void send();
      }
    },
    [send],
  );

  const vision = useVision(
    settings,
    settings.customGestures,
    handleGestureDetected,
  );

  const clearChat = () => {
    setMessages([WELCOME_MESSAGE()]);
    setLastProvider('Ready');
  };

  const modelLabel =
    settings.aiProvider === 'webllm'
      ? settings.webllmModel
      : settings.aiProvider === 'gemini'
        ? settings.geminiModel
        : settings.aiProvider === 'cloudFree'
          ? settings.cloudFreeModel
          : settings.ollamaModel;

  const visionStatus =
    vision.status === 'local'
      ? 'Local vision active'
      : vision.status === 'starting'
        ? 'Starting camera…'
        : vision.status === 'cloud'
          ? 'Cloud vision fallback'
          : vision.status === 'model-missing'
            ? 'Vision model missing'
            : vision.status === 'error'
              ? 'Camera unavailable'
              : 'Vision offline';

  const isOnlineVision =
    vision.status === 'local' || vision.status === 'cloud';

  return (
    <div className="nova-app">
      <header className="nova-header">
        <div className="nova-brand">
          <div className="nova-logo" aria-hidden="true">
            <Zap size={19} fill="currentColor" />
          </div>
          <div className="nova-brand-copy">
            <div className="nova-title">NOVA GESTURE AI</div>
            <div className="nova-subtitle">
              <span className="status-dot" />
              <span>JARVIS + ULTRON CANVAS</span>
              <span className="separator">·</span>
              <span>Local</span>
              <span className="separator">·</span>
              <span className="model-label">{modelLabel}</span>
            </div>
          </div>
        </div>

        <div className="header-actions">
          <button
            className={`header-button ${
              settings.visionEnabled ? 'is-active' : ''
            }`}
            onClick={() =>
              setSettings((previous) => ({
                ...previous,
                visionEnabled: !previous.visionEnabled,
              }))
            }
            aria-label="Toggle camera"
            title="Toggle camera"
          >
            {settings.visionEnabled ? (
              <Camera size={18} />
            ) : (
              <CameraOff size={18} />
            )}
          </button>
          <button
            className="header-button"
            onClick={() =>
              setSettings((previous) => ({
                ...previous,
                theme:
                  previous.theme === 'dark' ? 'light' : 'dark',
              }))
            }
            aria-label="Toggle theme"
            title="Toggle theme"
          >
            {settings.theme === 'dark' ? (
              <Sun size={18} />
            ) : (
              <Moon size={18} />
            )}
          </button>
          <button
            className="header-button"
            onClick={() => setSettingsOpen(true)}
            aria-label="Open settings"
            title="Settings"
          >
            <Settings size={18} />
          </button>
        </div>
      </header>

      <main className="nova-shell">
        <section
          className={`vision-card ${
            settings.visionEnabled ? 'vision-enabled' : 'vision-disabled-card'
          }`}
        >
          {settings.visionEnabled ? (
            <>
              <CameraView
                videoRef={vision.videoRef}
                enabled={settings.visionEnabled}
                status={vision.status}
                lastDetection={vision.lastDetection}
                onToggle={() =>
                  setSettings((previous) => ({
                    ...previous,
                    visionEnabled: !previous.visionEnabled,
                  }))
                }
                settings={settings}
              />

              <div className="vision-trainer">
                <input
                  value={mlInput}
                  onChange={(event) =>
                    setMlInput(event.target.value.toUpperCase())
                  }
                  placeholder="Sign (A-Z)"
                  maxLength={20}
                  aria-label="Custom sign label"
                />
                <button
                  onClick={() => {
                    if (!mlInput.trim()) return;
                    learnSign(mlInput);
                    setMlInput('');
                  }}
                >
                  Learn
                </button>
              </div>

              <div className="vision-status-pill">
                <span className={isOnlineVision ? 'live-dot' : 'idle-dot'} />
                {visionStatus}
              </div>
            </>
          ) : (
            <div className="vision-offline">
              <div className="offline-camera-icon">
                <CameraOff size={34} />
              </div>
              <div className="vision-offline-title">Vision is offline</div>
              <div className="vision-offline-subtitle">
                Enable the camera to use hand-gesture control and the wireless mouse.
              </div>
              <button
                className="enable-camera-button"
                onClick={() =>
                  setSettings((previous) => ({
                    ...previous,
                    visionEnabled: true,
                  }))
                }
              >
                Enable camera
              </button>
            </div>
          )}
        </section>

        <section className="assistant-card">
          <div className="assistant-header">
            <div className="assistant-heading">
              <div className="assistant-title-row">
                <div className="assistant-mini-icon">
                  <Bot size={14} />
                </div>
                <div className="assistant-title">Assistant</div>
              </div>
              <div className="assistant-subtitle">
                Jarvis + Ultron Generative Core
              </div>
            </div>

            <div className="assistant-header-right">
              <span className="provider-chip" title={lastProvider}>
                <span className="provider-dot" />
                {lastProvider}
                <ChevronDown size={11} />
              </span>
              <button
                className="clear-button"
                onClick={clearChat}
                title="Clear chat"
              >
                <Trash2 size={13} />
                <span>Clear</span>
              </button>
            </div>
          </div>

          <div className="messages-container">
            <div className="messages-inner">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message-row ${
                    message.role === 'user'
                      ? 'user-row'
                      : 'model-row'
                  }`}
                >
                  {message.role === 'model' && (
                    <div className="message-avatar model-avatar">
                      {getInitialsOrIcon(message.role)}
                    </div>
                  )}

                  <div className="message-column">
                    <div className="message-label">
                      {message.role === 'model' ? 'NOVA' : 'You'}
                    </div>
                    <div className="message-bubble">
                      {message.text}
                    </div>
                    {message.role === 'model' && (
                      <button
                        className="copy-message"
                        onClick={() =>
                          void copyToClipboard(message.text, message.id)
                        }
                        title="Copy response"
                      >
                        {copiedId === message.id ? (
                          <>
                            <Check size={12} />
                            Copied
                          </>
                        ) : (
                          <>
                            <Copy size={12} />
                            Copy
                          </>
                        )}
                      </button>
                    )}
                  </div>

                  {message.role === 'user' && (
                    <div className="message-avatar user-avatar">
                      {getInitialsOrIcon(message.role)}
                    </div>
                  )}
                </div>
              ))}

              {typing && (
                <div className="message-row model-row">
                  <div className="message-avatar model-avatar">
                    <Bot size={16} />
                  </div>
                  <div className="message-column">
                    <div className="message-label">NOVA</div>
                    <div className="typing-bubble" aria-label="NOVA is thinking">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatEnd} />
            </div>
          </div>

          {attachments.length > 0 && (
            <div className="attachment-strip">
              {attachments.map((attachment, index) => (
                <div className="attachment-chip" key={`${attachment.name}-${index}`}>
                  <FileText size={13} />
                  <span>{attachment.name}</span>
                  <button
                    onClick={() => removeAttachment(index)}
                    title="Remove attachment"
                    aria-label={`Remove ${attachment.name}`}
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="composer-area">
            <div className="composer-box">
              <button
                className="composer-icon-button"
                onClick={() => fileInputRef.current?.click()}
                title="Attach a file"
                aria-label="Attach a file"
                disabled={attachmentBusy}
              >
                {attachmentBusy ? (
                  <Loader2 size={18} className="spin" />
                ) : (
                  <Paperclip size={19} />
                )}
              </button>

              <input
                ref={fileInputRef}
                type="file"
                hidden
                accept=".txt,.md,.json,.csv,.js,.ts,.tsx,.py,.html,.css,.pdf,.png,.jpg,.jpeg,.webp"
                onChange={handleFileUpload}
              />

              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === 'Enter' &&
                    !event.shiftKey
                  ) {
                    event.preventDefault();
                    void send();
                  }
                }}
                placeholder="Ask anything..."
                rows={1}
                aria-label="Message NOVA"
              />

              <button
                className={`send-button ${
                  input.trim() || attachments.length > 0
                    ? 'ready'
                    : ''
                }`}
                onClick={() => void send()}
                disabled={
                  typing ||
                  (!input.trim() && attachments.length === 0)
                }
                title="Send message"
                aria-label="Send message"
              >
                <Send size={18} />
              </button>
            </div>

            <div className="composer-footer">
              <span>
                Enter to send · Shift + Enter for a new line
              </span>
              <span className="gesture-hint">
                🖐 Gesture control {settings.visionEnabled ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
        </section>
      </main>

      {settingsOpen && (
        <SettingsPanel
          settings={settings}
          onUpdate={setSettings}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}

