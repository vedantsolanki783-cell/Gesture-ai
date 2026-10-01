import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Bot, User, Send, Settings, Sparkles, BrainCircuit, Plus, MessageSquare, Trash2, Menu, Camera } from 'lucide-react';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import { learnSign, clearTrainedSigns, getTrainedSignsCount } from './vision/localVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

interface ChatSession { id: string; title: string; messages: Message[]; }

const defaults: AppSettings = {
  theme: 'dark', voiceEnabled: true, visionEnabled: false, confidenceThreshold: 0.72,
  aiProvider: 'ollama', ollamaUrl: 'http://localhost:11434', ollamaModel: 'qwen2.5:0.5b',
  geminiModel: 'gemini-2.5-flash', webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction: 'You are NOVA. Answer concisely.', customGestures: []
};

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try { return { ...defaults, ...JSON.parse(localStorage.getItem('nova_settings') || '{}') }; } catch { return defaults; }
  });

  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try { return JSON.parse(localStorage.getItem('nova_sessions') || '[]'); } catch { return []; }
  });

  const [sessionId, setSessionId] = useState(() => sessions[0]?.id || crypto.randomUUID());
  const [messages, setMessages] = useState<Message[]>(() => sessions[0]?.messages || [{ id: '1', role: 'model', text: 'How can I help you today?', timestamp: Date.now() }]);
  
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [mlInput, setMlInput] = useState('');
  const [mlCount, setMlCount] = useState(getTrainedSignsCount());
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => { localStorage.setItem('nova_settings', JSON.stringify(settings)); document.documentElement.dataset.theme = settings.theme; }, [settings]);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, typing]);

  useEffect(() => {
    setSessions(prev => {
      const idx = prev.findIndex(s => s.id === sessionId);
      const title = messages.find(m => m.role === 'user')?.text.slice(0, 24) || 'New Chat';
      const updated = idx >= 0 ? [...prev] : [{ id: sessionId, title, messages }, ...prev];
      if (idx >= 0) { updated[idx].messages = messages; updated[idx].title = updated[idx].title === 'New Chat' ? title : updated[idx].title; }
      localStorage.setItem('nova_sessions', JSON.stringify(updated));
      return updated;
    });
  }, [messages, sessionId]);

  const newChat = () => { setSessionId(crypto.randomUUID()); setMessages([{ id: crypto.randomUUID(), role: 'model', text: 'How can I help you today?', timestamp: Date.now() }]); };
  const loadChat = (s: ChatSession) => { setSessionId(s.id); setMessages(s.messages); };
  const delChat = (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); const rem = sessions.filter(s => s.id !== id); setSessions(rem);
    if (sessionId === id) { if (rem.length) loadChat(rem[0]); else newChat(); }
  };

  const send = useCallback(async (val = input) => {
    const text = val.trim(); if (!text || typing) return;
    setMessages(p => [...p, { id: crypto.randomUUID(), role: 'user', text, timestamp: Date.now() }]);
    setInput(''); setTyping(true);

    const bridge = (window as any).NovaAndroid;
    const isCmd = /^(open|go home|back|recents|enable overlay)/i.test(text);
    if (isCmd && bridge) {
      if (/overlay/i.test(text)) bridge.enableOverlayBubble();
      else if (/home/i.test(text)) bridge.globalAction('HOME');
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: `Executed system command: ${text}`, timestamp: Date.now() }]);
      setTyping(false); return;
    }

    try {
      const res = await generateLocalOrCloud(text, messages, settings);
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: res.text, timestamp: Date.now() }]);
    } catch {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: 'Offline Core Processed.', timestamp: Date.now() }]);
    } finally { setTyping(false); }
  }, [input, messages, settings, typing]);

  const vision = useVision(settings, settings.customGestures, (res) => {
    if (res.type === 'LETTER') setInput(p => p + res.value);
    else if (res.value === 'CLEAR') setInput('');
    else if (res.value === 'SEND') send();
  });

  return (
    <div className="flex h-screen bg-white dark:bg-[#212121] text-gray-800 dark:text-gray-100 font-sans overflow-hidden">
      
      {/* Sidebar - Chat History & Controls */}
      <aside className={`${drawerOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-gray-50 dark:bg-[#171717] border-r border-gray-200 dark:border-gray-800 flex flex-col overflow-hidden`}>
        <div className="p-3">
          <button onClick={newChat} className="w-full flex items-center gap-2 p-3 bg-white dark:bg-[#212121] border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-sm font-medium">
            <Plus size={16} /> New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          {sessions.map(s => (
            <div key={s.id} onClick={() => loadChat(s)} className={`flex items-center justify-between p-3 rounded-lg cursor-pointer text-sm ${s.id === sessionId ? 'bg-gray-200 dark:bg-[#2f2f2f]' : 'hover:bg-gray-100 dark:hover:bg-[#212121]'}`}>
              <div className="flex items-center gap-3 truncate"><MessageSquare size={16} className="opacity-70"/> <span className="truncate">{s.title}</span></div>
              <button onClick={e => delChat(s.id, e)} className="text-gray-400 hover:text-red-500"><Trash2 size={14}/></button>
            </div>
          ))}
        </div>
        
        {/* ML Trainer Module */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-800">
          <div className="text-xs font-bold text-gray-500 mb-2 uppercase">ML Sign Trainer</div>
          <div className="flex gap-2 mb-2">
            <input value={mlInput} onChange={e => setMlInput(e.target.value)} placeholder="Sign (A-Z)" className="w-full p-2 text-sm bg-white dark:bg-[#2f2f2f] border border-gray-300 dark:border-gray-600 rounded outline-none uppercase" />
            <button onClick={() => { learnSign(mlInput); setTimeout(() => setMlCount(getTrainedSignsCount()), 200); }} className="px-3 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">Learn</button>
          </div>
          <div className="text-xs text-gray-500 flex justify-between">
            <span>{mlCount} Saved</span>
            <button onClick={() => { clearTrainedSigns(); setMlCount(0); }} className="text-red-400 hover:underline">Clear</button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative h-full">
        <header className="h-14 flex items-center justify-between px-4 border-b border-gray-200 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <button onClick={() => setDrawerOpen(!drawerOpen)} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md"><Menu size={20}/></button>
            <span className="font-semibold text-lg flex items-center gap-2"><Sparkles size={18} className="text-blue-500"/> NOVA</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} className={`p-2 rounded-md ${settings.visionEnabled ? 'text-green-500 bg-green-50 dark:bg-green-900/20' : 'hover:bg-gray-100 dark:hover:bg-gray-800'}`}><Camera size={20}/></button>
            <button onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md">{settings.theme === 'dark' ? <Sun size={20}/> : <Moon size={20}/>}</button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:px-24 lg:px-48 pb-32">
          {messages.map(m => (
            <div key={m.id} className={`flex gap-4 mb-6 ${m.role === 'user' ? 'justify-end' : ''}`}>
              {m.role === 'model' && <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white flex-shrink-0"><Bot size={18}/></div>}
              <div className={`px-4 py-3 rounded-2xl max-w-[80%] text-[15px] leading-relaxed ${m.role === 'user' ? 'bg-gray-100 dark:bg-[#2f2f2f]' : 'bg-transparent'}`}>
                {m.text}
              </div>
            </div>
          ))}
          {typing && <div className="flex gap-4 mb-6"><div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white"><Bot size={18}/></div><div className="px-4 py-3">● ● ●</div></div>}
          <div ref={chatEnd} />
        </div>

        {/* Floating Camera Preview */}
        <div className={`absolute left-4 bottom-28 transition-all ${settings.visionEnabled ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
          <CameraView videoRef={vision.videoRef} enabled={settings.visionEnabled} status={vision.status} lastDetection={vision.lastDetection} onToggle={() => {}} settings={settings} />
        </div>

        {/* Input Bar */}
        <div className="absolute bottom-0 left-0 w-full p-4 md:px-24 lg:px-48 bg-gradient-to-t from-white via-white dark:from-[#212121] dark:via-[#212121] to-transparent">
          <div className="relative flex items-center bg-gray-100 dark:bg-[#2f2f2f] rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm focus-within:ring-1 focus-within:ring-gray-400">
            <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()} placeholder="Message NOVA..." className="flex-1 bg-transparent border-none py-4 px-5 outline-none" />
            <button onClick={() => send()} disabled={!input.trim() || typing} className={`p-2 mr-2 rounded-lg transition-colors ${input.trim() ? 'bg-blue-600 text-white' : 'bg-transparent text-gray-400 dark:text-gray-500'}`}>
              <Send size={18} />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
