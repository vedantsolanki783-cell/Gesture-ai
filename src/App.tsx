import React, { useEffect, useRef, useState } from 'react';
import { Bot, User, Send, Settings, CameraOff, Sun, Moon, Trash2, Mic, Paperclip, Copy, Zap } from 'lucide-react';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import { learnSign, clearTrainedSigns, getTrainedSignsCount } from './vision/localVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

const defaults: AppSettings = {
  theme: 'dark', voiceEnabled: true, visionEnabled: false, confidenceThreshold: 0.72,
  aiProvider: 'ollama', ollamaUrl: 'http://localhost:11434', ollamaModel: 'qwen2.5:1.5b',
  geminiModel: 'gemini-2.5-flash', webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction: 'You are NOVA. Answer concisely.', customGestures: []
};

function executeAndroidAgentCommand(rawText: string): string | null {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  const text = rawText.toLowerCase().trim();

  const isOverlay = /\b(enable|start|show|turn on)\s+(overlay|bubble|hud)\b/i.test(text);
  const isUltron = /\b(enable|open|turn on)\s+(accessibility|ultron)\b/i.test(text);
  const isHome = /^(go\s+home|home)$/i.test(text);
  const isBack = /^(go\s+back|back)$/i.test(text);
  const openAppMatch = text.match(/^(?:open|launch)\s+([a-z0-9\s._-]+)$/i);

  if (!bridge && (isOverlay || isUltron || isHome || isBack || openAppMatch)) return '⚠️ Android Bridge Not Found.';
  if (isOverlay) return bridge.enableOverlayBubble() === 'OPENED_OVERLAY_SETTINGS' ? '⚡ Opening Settings...' : '⚡ Floating Bubble Active!';
  if (isUltron) { bridge.openAccessibilitySettings(); return '🤖 Opening Accessibility Settings...'; }
  if (openAppMatch && bridge.openApp(openAppMatch[1].trim())) return `🚀 Launching ${openAppMatch[1].toUpperCase()}...`;
  if (bridge && !bridge.isUltronConnected()) { bridge.openAccessibilitySettings(); return '⚠️ Ultron OFF. Turn it ON.'; }
  if (isHome) { bridge.globalAction('HOME'); return '🏠 Executed HOME.'; }
  if (isBack) { bridge.globalAction('BACK'); return '🔙 Executed BACK.'; }
  return null;
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try { return { ...defaults, ...JSON.parse(localStorage.getItem('nova_settings') || '{}') }; } catch { return defaults; }
  });

  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'model', text: 'Hello! How can I assist you today?', timestamp: Date.now() }
  ]);
  
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [mlInput, setMlInput] = useState('');
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => { localStorage.setItem('nova_settings', JSON.stringify(settings)); document.documentElement.dataset.theme = settings.theme; }, [settings]);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, typing]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const send = async (val = input) => {
    const text = val.trim(); if (!text || typing) return;
    setMessages(p => [...p, { id: crypto.randomUUID(), role: 'user', text, timestamp: Date.now() }]);
    setInput(''); setTyping(true);

    const androidReply = executeAndroidAgentCommand(text);
    if (androidReply) {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: androidReply, timestamp: Date.now() }]);
      setTyping(false); return;
    }

    try {
      const res = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: settings.ollamaModel,
          messages: [
            { role: 'system', content: settings.systemInstruction },
            ...messages.map(m => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.text })),
            { role: 'user', content: text }
          ],
          stream: false
        })
      });
      const data = await res.json();
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: data.message.content, timestamp: Date.now() }]);
    } catch {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: '⚠️ Local AI Offline. Open Termux and run `ollama serve`.', timestamp: Date.now() }]);
    } finally { setTyping(false); }
  };

  const vision = useVision(settings, settings.customGestures, (res) => {
    if (res.type === 'LETTER') setInput(p => p + res.value);
    else if (res.value === 'CLEAR') setInput('');
    else if (res.value === 'SEND') send();
  });

  return (
    <div className="flex flex-col h-screen bg-[#090D16] text-gray-100 font-sans">
      
      {/* Header exactly like 2nd image */}
      <header className="flex items-center justify-between p-4 bg-[#090D16]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full border border-cyan-500 shadow-[0_0_12px_rgba(6,182,212,0.4)] flex items-center justify-center bg-black/50">
            <Zap size={20} fill="currentColor" className="text-yellow-500" />
          </div>
          <div>
            <h1 className="font-bold text-[13px] tracking-widest text-white uppercase">NOVA GESTURE AI</h1>
            <p className="text-[10px] text-gray-400 flex items-center gap-1.5 font-medium tracking-wide">
              <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full inline-block"></span> 
              JARVIS + ULTRON CANVAS · Local · {settings.ollamaModel}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} className="p-2 bg-transparent hover:bg-[#1E293B] rounded-lg text-gray-400 transition-colors">
            <CameraOff size={20} strokeWidth={1.5} />
          </button>
          <button onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))} className="p-2 bg-transparent hover:bg-[#1E293B] rounded-lg text-gray-400 transition-colors">
            {settings.theme === 'dark' ? <Sun size={20} strokeWidth={1.5} /> : <Moon size={20} strokeWidth={1.5} />}
          </button>
          <button onClick={() => setSettingsOpen(true)} className="p-2 bg-transparent hover:bg-[#1E293B] rounded-lg text-gray-400 transition-colors">
            <Settings size={20} strokeWidth={1.5} />
          </button>
        </div>
      </header>

      <main className="flex-1 flex flex-col gap-4 overflow-hidden max-w-4xl mx-auto w-full px-4 pb-4">
        
        {/* Top Camera Block */}
        <div className="flex-shrink-0 bg-[#121927] rounded-3xl border border-gray-800/60 flex items-center justify-center relative overflow-hidden" style={{ minHeight: '220px' }}>
          {settings.visionEnabled ? (
            <CameraView videoRef={vision.videoRef} enabled={settings.visionEnabled} status={vision.status} lastDetection={vision.lastDetection} onToggle={() => {}} settings={settings} />
          ) : (
            <div className="flex flex-col items-center gap-4">
              <CameraOff size={42} strokeWidth={1} className="text-gray-500" />
              <p className="text-gray-400 font-medium text-sm">Vision is offline</p>
              <button onClick={() => setSettings(s => ({...s, visionEnabled: true}))} className="px-5 py-2 bg-[#10B981] text-black font-bold text-xs rounded-full shadow-lg shadow-[#10B981]/20 hover:bg-[#059669] transition-colors">
                Enable camera
              </button>
            </div>
          )}

          {/* ML Trainer Overlay (Hidden unless vision is active) */}
          {settings.visionEnabled && (
            <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-md p-2 rounded-xl border border-white/10 flex items-center gap-2">
              <input value={mlInput} onChange={e => setMlInput(e.target.value.toUpperCase())} placeholder="Sign (A-Z)" className="w-24 bg-transparent border-b border-gray-500 text-xs text-white outline-none uppercase pb-1" />
              <button onClick={() => { learnSign(mlInput); setMlInput(''); }} className="bg-[#10B981] text-black text-[10px] font-bold px-2 py-1 rounded">Learn</button>
            </div>
          )}
        </div>

        {/* Chat Section */}
        <div className="flex-1 bg-[#0C111D] rounded-3xl border border-gray-800/60 flex flex-col overflow-hidden relative">
          
          <div className="flex items-center justify-between p-4 border-b border-gray-800/60 bg-[#0C111D]">
            <div>
              <h2 className="text-sm font-bold text-white">Assistant</h2>
              <p className="text-[11px] text-gray-500">Jarvis + Ultron Generative Core</p>
            </div>
            <button onClick={() => setMessages([{ id: '1', role: 'model', text: 'Hello! How can I assist you today?', timestamp: Date.now() }])} className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white px-3 py-1.5 bg-[#1E293B] rounded-lg border border-gray-700/50 transition-colors">
              <Trash2 size={14} /> Clear
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {messages.map(m => (
              <div key={m.id} className={`flex items-start gap-3 w-full ${m.role === 'user' ? 'justify-end' : ''}`}>
                
                {/* AI Bubble */}
                {m.role === 'model' && (
                  <>
                    <div className="w-8 h-8 rounded-full bg-[#10B981]/10 flex items-center justify-center flex-shrink-0 text-[#10B981]">
                      <Bot size={18} strokeWidth={1.5} />
                    </div>
                    <div>
                      <div className="px-4 py-3 rounded-2xl rounded-tl-none max-w-full text-[13px] leading-relaxed bg-[#1E293B] text-gray-200 shadow-sm border border-gray-700/30">
                        <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
                      </div>
                      <button onClick={() => copyToClipboard(m.text)} className="text-gray-500 hover:text-gray-300 mt-1.5 ml-1 p-1 flex items-center gap-1 text-[10px]">
                        <Copy size={12} />
                      </button>
                    </div>
                  </>
                )}

                {/* User Bubble */}
                {m.role === 'user' && (
                  <>
                    <div className="px-4 py-3 rounded-2xl rounded-tr-none max-w-[75%] text-[13px] leading-relaxed bg-[#10B981] text-[#090D16] font-medium shadow-sm">
                      <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
                    </div>
                    <div className="w-8 h-8 rounded-lg bg-[#10B981] flex items-center justify-center flex-shrink-0 text-[#090D16]">
                      <User size={18} strokeWidth={2} />
                    </div>
                  </>
                )}

              </div>
            ))}

            {/* Bouncing Dots Thinking Animation */}
            {typing && (
              <div className="flex items-start gap-3 w-full">
                <div className="w-8 h-8 rounded-full bg-[#10B981]/10 flex items-center justify-center flex-shrink-0 text-[#10B981]">
                  <Bot size={18} strokeWidth={1.5} />
                </div>
                <div className="bg-[#1E293B] px-5 py-4 rounded-2xl rounded-tl-none flex items-center gap-1.5 shadow-sm border border-gray-700/30">
                  <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full animate-bounce"></span>
                  <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            )}
            <div ref={chatEnd} />
          </div>

          {/* Input Bar */}
          <div className="p-3 bg-[#0C111D]">
            <div className="flex items-center gap-2 bg-[#121927] rounded-xl p-1.5 border border-gray-800">
              <button className="p-2.5 text-gray-400 hover:text-white transition-colors"><Mic size={18} /></button>
              <button className="p-2.5 text-gray-400 hover:text-white transition-colors"><Paperclip size={18} /></button>
              <input 
                value={input} 
                onChange={e => setInput(e.target.value)} 
                onKeyDown={e => e.key === 'Enter' && send()} 
                placeholder="Ask anything, 'make presentation on...', 'build calculator in html'..." 
                className="flex-1 bg-transparent border-none text-[13px] text-white placeholder-gray-500 outline-none" 
              />
              <button 
                onClick={() => send()} 
                disabled={!input.trim() || typing} 
                className={`p-2.5 mr-1 rounded-lg transition-colors ${input.trim() ? 'bg-[#10B981] text-[#090D16] hover:bg-[#059669]' : 'bg-transparent text-gray-600'}`}
              >
                <Send size={18} strokeWidth={2} className={input.trim() ? 'translate-x-0.5' : ''} />
              </button>
            </div>
          </div>
        </div>
      </main>

      {settingsOpen && <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}
