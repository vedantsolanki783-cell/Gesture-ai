import React, {
  useState,
  useRef,
  useEffect,
  useCallback
} from 'react';

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
  Camera
} from 'lucide-react';

import {
  Message,
  AppSettings,
  VisionResponse
} from './types';

import {
  generateLocalOrCloud
} from './services/aiRouter';

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

  ollamaUrl:
    'http://localhost:11434',

  ollamaModel:
    'qwen2.5:0.5b',

  geminiModel:
    'gemini-2.5-flash',

  webllmModel:
    'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',

  systemInstruction:
    'You are NOVA. Answer concisely.',

  customGestures: []
};


/* =========================================
   ANDROID / ULTRON COMMANDS
========================================= */

function executeAndroidAgentCommand(
  rawText: string
): string | null {

  const bridge =
    typeof window !== 'undefined'
      ? (window as any).NovaAndroid
      : null;

  const text =
    rawText.toLowerCase().trim();


  const isOverlay =
    /\b(enable|start|show|turn on)\s+(overlay|bubble|hud)\b/i
      .test(text);


  const isUltron =
    /\b(enable|open|turn on)\s+(accessibility|ultron)\b/i
      .test(text);


  const isHome =
    /^(go\s+home|home)$/i.test(text);


  const isBack =
    /^(go\s+back|back)$/i.test(text);


  const openAppMatch =
    text.match(
      /^(?:open|launch)\s+([a-z0-9\s._-]+)$/i
    );


  if (
    !bridge &&
    (
      isOverlay ||
      isUltron ||
      isHome ||
      isBack ||
      openAppMatch
    )
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
    bridge.openApp(
      openAppMatch[1].trim()
    )
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


/* =========================================
   APP
========================================= */

export default function App() {

  const [settings, setSettings] =
    useState<AppSettings>(() => {

      try {

        return {
          ...defaults,

          ...JSON.parse(
            localStorage.getItem(
              'nova_settings'
            ) || '{}'
          )
        };

      } catch {

        return defaults;
      }

    });


  const [messages, setMessages] =
    useState<Message[]>([
      {
        id: 'welcome',
        role: 'model',
        content:
          'Hello! How can I assist you today?',
        timestamp: Date.now()
      }
    ]);


  const [input, setInput] =
    useState('');


  const [typing, setTyping] =
    useState(false);


  const [settingsOpen, setSettingsOpen] =
    useState(false);


  const [mlInput, setMlInput] =
    useState('');


  const chatEnd =
    useRef<HTMLDivElement>(null);


  /* =========================================
     SAVE SETTINGS
  ========================================= */

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


  /* =========================================
     AUTO SCROLL
  ========================================= */

  useEffect(() => {

    chatEnd.current?.scrollIntoView({
      behavior: 'smooth'
    });

  }, [messages, typing]);


  /* =========================================
     COPY
  ========================================= */

  const copyToClipboard =
    (text: string) => {

      navigator.clipboard.writeText(text);

    };


  /* =========================================
     SEND MESSAGE
  ========================================= */

  const send =
    useCallback(
      async (val = input) => {

        const text =
          val.trim();


        if (
          !text ||
          typing
        ) {
          return;
        }


        const userMessage: Message = {

          id:
            crypto.randomUUID(),

          role:
            'user',

          content:
            text,

          timestamp:
            Date.now()
        };


        setMessages(
          previous => [
            ...previous,
            userMessage
          ]
        );


        setInput('');

        setTyping(true);


        /* Android commands */

        const androidReply =
          executeAndroidAgentCommand(
            text
          );


        if (androidReply) {

          setMessages(
            previous => [
              ...previous,
              {
                id:
                  crypto.randomUUID(),

                role:
                  'model',

                content:
                  androidReply,

                timestamp:
                  Date.now()
              }
            ]
          );


          setTyping(false);

          return;
        }


        /* AI */

        try {

          const response =
            await generateLocalOrCloud(
              text,
              messages,
              settings
            );


          setMessages(
            previous => [
              ...previous,
              {
                id:
                  crypto.randomUUID(),

                role:
                  'model',

                content:
                  response.text,

                timestamp:
                  Date.now()
              }
            ]
          );


        } catch {

          setMessages(
            previous => [
              ...previous,
              {
                id:
                  crypto.randomUUID(),

                role:
                  'model',

                content:
                  '⚠️ Local AI Offline. Open Termux and run `ollama serve`.',

                timestamp:
                  Date.now()
              }
            ]
          );

        } finally {

          setTyping(false);

        }

      },
      [
        input,
        messages,
        settings,
        typing
      ]
    );


  /* =========================================
     GESTURE
  ========================================= */

  const handleGestureDetected =
    useCallback(
      (result: VisionResponse) => {

        if (
          result.type === 'LETTER'
        ) {

          setInput(
            previous =>
              previous + result.value
          );

        }

        else if (
          result.value === 'CLEAR'
        ) {

          setInput('');

        }

        else if (
          result.value === 'SEND'
        ) {

          send();

        }

      },
      [send]
    );


  const vision =
    useVision(
      settings,
      settings.customGestures,
      handleGestureDetected
    );


  /* =========================================
     CLEAR CHAT
  ========================================= */

  const clearChat = () => {

    setMessages([
      {
        id: 'welcome',
        role: 'model',
        content:
          'Hello! How can I assist you today?',
        timestamp: Date.now()
      }
    ]);

  };


  return (

    <div className="nova-app">


      {/* ===================================
          HEADER
      =================================== */}

      <header className="nova-header">

        <div className="nova-brand">

          <div className="nova-logo">

            <Zap
              size={19}
              fill="currentColor"
            />

          </div>


          <div className="nova-brand-text">

            <div className="nova-title">
              NOVA GESTURE AI
            </div>

            <div className="nova-subtitle">

              <span className="online-dot" />

              JARVIS + ULTRON CANVAS
              <span className="separator">
                ·
              </span>

              Local
              <span className="separator">
                ·
              </span>

              {settings.ollamaModel}

            </div>

          </div>

        </div>


        <div className="header-actions">

          <button
            className="header-button"
            onClick={() =>
              setSettings(
                previous => ({
                  ...previous,
                  visionEnabled:
                    !previous.visionEnabled
                })
              )
            }
          >

            {settings.visionEnabled

              ? (
                <Camera
                  size={19}
                  className="active-green"
                />
              )

              : (
                <CameraOff
                  size={19}
                />
              )

            }

          </button>


          <button
            className="header-button"
            onClick={() =>
              setSettings(
                previous => ({
                  ...previous,

                  theme:
                    previous.theme === 'dark'
                      ? 'light'
                      : 'dark'
                })
              )
            }
          >

            {settings.theme === 'dark'

              ? <Sun size={19} />

              : <Moon size={19} />

            }

          </button>


          <button
            className="header-button"
            onClick={() =>
              setSettingsOpen(true)
            }
          >

            <Settings size={19} />

          </button>

        </div>

      </header>


      {/* ===================================
          MAIN
      =================================== */}

      <main className="nova-container">


        {/* =================================
            VISION
        ================================= */}

        <section
          className={
            `vision-card ${
              settings.visionEnabled
                ? 'vision-active'
                : 'vision-off'
            }`
          }
        >

          {settings.visionEnabled ? (

            <>

              <div className="camera-wrapper">

                <CameraView

                  videoRef={
                    vision.videoRef
                  }

                  enabled={
                    settings.visionEnabled
                  }

                  status={
                    vision.status
                  }

                  lastDetection={
                    vision.lastDetection
                  }

                  onToggle={() =>
                    setSettings(
                      previous => ({
                        ...previous,
                        visionEnabled:
                          !previous.visionEnabled
                      })
                    )
                  }

                  settings={
                    settings
                  }

                />

              </div>


              <div className="vision-trainer">

                <input

                  value={mlInput}

                  onChange={event =>
                    setMlInput(
                      event.target.value
                        .toUpperCase()
                    )
                  }

                  placeholder="Sign (A-Z)"

                />


                <button
                  onClick={() => {

                    learnSign(
                      mlInput
                    );

                    setMlInput('');

                  }}
                >
                  Learn
                </button>

              </div>

            </>

          ) : (

            <div className="vision-disabled">

              <CameraOff
                size={44}
              />

              <div className="vision-disabled-title">
                Vision is offline
              </div>

              <button
                className="enable-camera"
                onClick={() =>
                  setSettings(
                    previous => ({
                      ...previous,
                      visionEnabled:
                        true
                    })
                  )
                }
              >
                Enable camera
              </button>

            </div>

          )}

        </section>


        {/* =================================
            ASSISTANT
        ================================= */}

        <section className="assistant-card">


          {/* Assistant header */}

          <div className="assistant-header">

            <div>

              <div className="assistant-title">
                Assistant
              </div>

              <div className="assistant-subtitle">
                Jarvis + Ultron Generative Core
              </div>

            </div>


            <button
              className="clear-button"
              onClick={clearChat}
            >

              <Trash2 size={14} />

              Clear

            </button>

          </div>


          {/* Messages */}

          <div className="messages-container">

            {messages.map(
              message => (

                <div
                  key={message.id}
                  className={
                    `message-row ${
                      message.role === 'user'
                        ? 'message-user'
                        : 'message-model'
                    }`
                  }
                >


                  {message.role === 'model' && (

                    <div className="message-avatar">

                      <Bot size={16} />

                    </div>

                  )}


                  <div className="message-block">

                    <div className="message-bubble">

                      {message.content}

                    </div>


                    {message.role ===
                      'model' && (

                      <button
                        className="copy-message"
                        onClick={() =>
                          copyToClipboard(
                            message.content
                          )
                        }
                      >

                        <Copy size={13} />

                      </button>

                    )}

                  </div>


                  {message.role === 'user' && (

                    <div className="user-avatar">

                      <User size={16} />

                    </div>

                  )}

                </div>

              )
            )}


            {/* Thinking */}

            {typing && (

              <div className="message-row message-model">

                <div className="message-avatar">

                  <Bot size={16} />

                </div>


                <div className="typing-bubble">

                  <span />
                  <span />
                  <span />

                </div>

              </div>

            )}


            <div ref={chatEnd} />

          </div>


          {/* =================================
              COMPOSER
          ================================= */}

          <div className="composer-area">

            <div className="composer">

              <button className="composer-button">

                <Mic size={19} />

              </button>


              <button className="composer-button">

                <Paperclip size={19} />

              </button>


              <input

                value={input}

                onChange={event =>
                  setInput(
                    event.target.value
                  )
                }

                onKeyDown={event => {

                  if (
                    event.key ===
                    'Enter'
                  ) {

                    send();

                  }

                }}

                placeholder="Ask anything..."

              />


              <button

                className={
                  `send-button ${
                    input.trim()
                      ? 'send-ready'
                      : ''
                  }`
                }

                disabled={
                  !input.trim() ||
                  typing
                }

                onClick={() =>
                  send()
                }

              >

                <Send size={18} />

              </button>

            </div>


            <div className="composer-footer">

              <span>
                NOVA can make mistakes.
                Check important information.
              </span>

              <span>
                🖐 Gesture control
              </span>

            </div>

          </div>

        </section>

      </main>


      {/* SETTINGS */}

      <SettingsPanel

        isOpen={
          settingsOpen
        }

        onClose={() =>
          setSettingsOpen(false)
        }

        settings={
          settings
        }

        onSave={
          setSettings
        }

      />

    </div>

  );
  }
