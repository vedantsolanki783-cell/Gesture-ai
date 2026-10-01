import React, { useEffect, useRef, useState } from 'react';
import { Bot, User, Send, Settings, Sparkles, CameraOff, Sun, Moon, Trash2, Mic, Paperclip, CheckSquare } from 'lucide-react';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import { learnSign, clearTrainedSigns, getTrainedSignsCount } from './vision/localVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

const defaults: AppSettings = {
  theme: 'dark', voiceEnabled: true, visionEnabled: false, confidenceThreshold: 0.72,
  aiProvider: 'ollama', ollamaUrl: 'http://localhost:11434', ollamaModel: 'qwen2.5:0.5b',
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
      const res = await generateLocalOrCloud(text, messages, settings);
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: res.text, timestamp: Date.now() }]);
    } catch {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: '⚠️ Local AI Offline. Run `ollama serve`.', timestamp: Date.now() }]);
    } finally { setTyping(false); }
  };

  const vision = useVision(settings, settings.customGestures, (res) => {
    if (res.type === 'LETTER') setInput(p => p + res.value);
    else if (res.value === 'CLEAR') setInput('');
    else if (res.value === 'SEND') send();
  });

  return (
    <div className="flex flex-col h-screen bg-[#090D16] text-gray-100 font-sans">
      
      {/* Header (Exactly like Image 2) */}
      <header className="flex items-center justify-between p-4 bg-[#090D16]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-transparent border-2 border-[#10B981] flex items-center justify-center">
            <Sparkles size={18} className="text-[#10B981]" />
          </div>
          <div>
            <h1 className="font-bold text-sm tracking-widest text-white">NOVA GESTURE AI</h1>
            <p className="text-[10px] text-[#10B981] flex items-center gap-1 font-medium tracking-wide">
              <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full inline-block"></span> JARVIS + ULTRON CANVAS · Local · {settings.ollamaModel}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} className="p-2 bg-[#1E293B] rounded-lg text-gray-400 hover:text-white transition-colors">
            <CameraOff size={18} />
          </button>
          <button onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))} className="p-2 bg-[#1E293B] rounded-lg text-gray-400 hover:text-white transition-colors">
            {settings.theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <button onClick={() => setSettingsOpen(true)} className="p-2 bg-[#1E293B] rounded-lg text-gray-400 hover:text-white transition-colors">
            <Settings size={18} />
          </button>
        </div>
      </header>

      <main className="flex-1 flex flex-col p-4 gap-4 overflow-hidden max-w-4xl mx-auto w-full">
        
        {/* Top Camera Block */}
        <div className="flex-shrink-0 bg-[#111827] rounded-3xl border border-gray-800 flex items-center justify-center relative overflow-hidden" style={{ minHeight: '220px' }}>
          {settings.visionEnabled ? (
            <CameraView videoRef={vision.videoRef} enabled={settings.visionEnabled} status={vision.status} lastDetection={vision.lastDetection} onToggle={() => {}} settings={settings} />
          ) : (
            <div className="flex flex-col items-center gap-3">
              <CameraOff size={32} className="text-gray-500" />
              <p className="text-gray-400 font-medium">Vision is offline</p>
              <button onClick={() => setSettings(s => ({...s, visionEnabled: true}))} className="px-4 py-1.5 bg-[#10B981] text-black font-bold text-sm rounded-full shadow-lg shadow-[#10b981]/20">
                Enable camera
              </button>
            </div>
          )}

          {/* Discreet ML Trainer Overlay */}
          {settings.visionEnabled && (
            <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-md p-2 rounded-xl border border-white/10 flex items-center gap-2">
              <input value={mlInput} onChange={e => setMlInput(e.target.value.toUpperCase())} placeholder="Sign (A-Z)" className="w-24 bg-transparent border-b border-gray-500 text-xs text-white outline-none uppercase pb-1" />
              <button onClick={() => { learnSign(mlInput); setMlInput(''); }} className="bg-[#10B981] text-black text-[10px] font-bold px-2 py-1 rounded">Learn</button>
            </div>
          )}
        </div>

        {/* Chat Section */}
        <div className="flex-1 bg-[#111827] rounded-3xl border border-gray-800 flex flex-col overflow-hidden relative">
          <div className="flex items-center justify-between p-4 border-b border-gray-800">
            <div>
              <h2 className="text-sm font-bold text-white">Assistant</h2>
              <p className="text-xs text-gray-500">Jarvis + Ultron Generative Core</p>
            </div>
            <button onClick={() => setMessages([{ id: '1', role: 'model', text: 'Hello! How can I assist you today?', timestamp: Date.now() }])} className="flex items-center gap-1 text-xs text-gray-400 hover:text-white px-3 py-1.5 bg-[#1E293B] rounded-lg">
              <Trash2 size={14} /> Clear
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {messages.map(m => (
              <div key={m.id} className={`flex items-end gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}>
                {m.role === 'model' && (
                  <div className="w-8 h-8 rounded-full bg-[#10B981]/10 border border-[#10B981]/30 flex items-center justify-center text-[#10B981] flex-shrink-0">
                    <Bot size={16} />
                  </div>
                )}
                
                <div className={`px-4 py-3 rounded-2xl max-w-[75%] text-sm leading-relaxed ${m.role === 'user' ? 'bg-[#10B981] text-black rounded-br-none font-medium' : 'bg-[#1E293B] text-gray-200 rounded-bl-none'}`}>
                  <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
                </div>

                {m.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-[#10B981] flex items-center justify-center text-black flex-shrink-0">
                    <User size={16} />
                  </div>
                )}
              </div>
            ))}

            {/* THE NEW THINKING ANIMATION */}
            {typing && (
              <div className="flex items-end gap-3">
                <div className="w-8 h-8 rounded-full bg-[#10B981]/10 border border-[#10B981]/30 flex items-center justify-center text-[#10B981] flex-shrink-0">
                  <Bot size={16} />
                </div>
                <div className="bg-[#1E293B] px-5 py-4 rounded-2xl rounded-bl-none flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full animate-bounce"></span>
                  <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-1.5 h-1.5 bg-[#10B981] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            )}
            <div ref={chatEnd} />
          </div>

          {/* Input Bar */}
          <div className="p-4 bg-[#111827]">
            <div className="flex items-center gap-2 bg-[#1E293B] rounded-2xl p-1 border border-gray-700">
              <button className="p-3 text-gray-400 hover:text-white"><Mic size={18} /></button>
              <button className="p-3 text-gray-400 hover:text-white"><Paperclip size={18} /></button>
              <input 
                value={input} 
                onChange={e => setInput(e.target.value)} 
                onKeyDown={e => e.key === 'Enter' && send()} 
                placeholder="Ask anything, 'make presentation on...', 'build calculator in html'..." 
                className="flex-1 bg-transparent border-none text-sm text-white placeholder-gray-500 outline-none" 
              />
              <button 
                onClick={() => send()} 
                disabled={!input.trim() || typing} 
                className={`p-3 mr-1 rounded-xl transition-colors ${input.trim() ? 'bg-[#10B981] text-black shadow-lg shadow-[#10B981]/20' : 'bg-transparent text-gray-500'}`}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </main>

      {settingsOpen && <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}
