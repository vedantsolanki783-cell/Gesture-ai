import React, { useEffect, useRef, useState } from 'react';
import { Bot, User, Send, Settings, Sparkles, BrainCircuit, Plus, MessageSquare, Trash2, Menu, Camera, Sun, Moon, X, Loader2, Zap } from 'lucide-react';
import { learnSign, clearTrainedSigns, getTrainedSignsCount, localVision } from './vision/localVision';
import './styles.css';

// ============================================================================
// TYPES & SYSTEM COMMANDS
// ============================================================================
interface Message { id: string; role: 'user' | 'model'; text: string; timestamp: number; }
interface ChatSession { id: string; title: string; messages: Message[]; }
interface AppSettings { theme: 'light' | 'dark'; visionEnabled: boolean; ollamaModel: string; systemInstruction: string; }

function executeAndroidAgentCommand(rawText: string): string | null {
  const bridge = typeof window !== 'undefined' ? (window as any).NovaAndroid : null;
  const text = rawText.toLowerCase().trim();

  const isOverlay = /\b(enable|start|show|turn on)\s+(overlay|bubble|hud)\b/i.test(text);
  const isUltron = /\b(enable|open|turn on)\s+(accessibility|ultron)\b/i.test(text);
  const isHome = /^(go\s+home|home)$/i.test(text);
  const isBack = /^(go\s+back|back)$/i.test(text);
  const isRecents = /^(open\s+recents|recents)$/i.test(text);
  const isNotif = /^(open\s+notifications|notifications)$/i.test(text);
  const openAppMatch = text.match(/^(?:open|launch)\s+([a-z0-9\s._-]+)$/i);

  if (!bridge && (isOverlay || isUltron || isHome || isBack || openAppMatch)) {
    return '⚠️ **Hardware Bridge Disconnected:** Compile the APK to execute system-level AGI overrides.';
  }

  if (isOverlay) return bridge.enableOverlayBubble() === 'OPENED_OVERLAY_SETTINGS' ? '⚡ Initializing AGI HUD Settings...' : '⚡ AGI Floating HUD Online.';
  if (isUltron) { bridge.openAccessibilitySettings(); return '🤖 Bypassing Android Security... Turn ON Ultron in Settings.'; }
  if (openAppMatch && bridge.openApp(openAppMatch[1].trim())) return `🚀 Executing Launch Sequence: **${openAppMatch[1].toUpperCase()}**...`;
  
  if (bridge && !bridge.isUltronConnected()) {
    bridge.openAccessibilitySettings();
    return '⚠️ Ultron Agent Core offline. Require Accessibility permission to proceed.';
  }

  if (isHome) { bridge.globalAction('HOME'); return '🏠 Executed: **HOME**.'; }
  if (isBack) { bridge.globalAction('BACK'); return '🔙 Executed: **BACK**.'; }
  if (isRecents) { bridge.globalAction('RECENTS'); return '🗂️ Executed: **RECENTS**.'; }
  if (isNotif) { bridge.globalAction('NOTIFICATIONS'); return '🔔 Executed: **NOTIFICATIONS**.'; }

  return null;
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================
export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try { 
      return { 
        theme: 'dark', visionEnabled: false, ollamaModel: 'qwen2.5:0.5b', 
        systemInstruction: 'You are NOVA, an advanced Artificial General Intelligence (AGI). You operate with absolute professionalism, extreme logical precision, and self-improving cognitive loops. You manage the user\'s local device via Ultron and answer queries autonomously.',
        ...JSON.parse(localStorage.getItem('nova_settings') || '{}') 
      }; 
    } 
    catch { return { theme: 'dark', visionEnabled: false, ollamaModel: 'qwen2.5:0.5b', systemInstruction: '' }; }
  });

  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try { return JSON.parse(localStorage.getItem('nova_sessions') || '[]'); } 
    catch { return []; }
  });

  const [sessionId, setSessionId] = useState(() => sessions[0]?.id || crypto.randomUUID());
  const [messages, setMessages] = useState<Message[]>(() => sessions[0]?.messages || [
    { id: '1', role: 'model', text: '⚡ **NOVA AGI Core: ONLINE**\n\nSystems calibrated. Advanced neural vision and local reasoning engines are standing by. How shall we proceed?', timestamp: Date.now() }
  ]);
  
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  
  const [mlInput, setMlInput] = useState('');
  const [mlCount, setMlCount] = useState(0);
  
  const chatEnd = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  const inputRef = useRef(input);
  const sendRef = useRef<() => void>(() => {});

  useEffect(() => { inputRef.current = input; }, [input]);
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

  useEffect(() => {
    try { setTimeout(() => setMlCount(getTrainedSignsCount()), 500); } catch {}
  }, []);

  const newChat = () => { setSessionId(crypto.randomUUID()); setMessages([{ id: crypto.randomUUID(), role: 'model', text: '⚡ **NOVA AGI Core: ONLINE**\n\nSystems calibrated. Awaiting instruction.', timestamp: Date.now() }]); };
  const loadChat = (s: ChatSession) => { setSessionId(s.id); setMessages(s.messages); if(window.innerWidth < 768) setDrawerOpen(false); };
  const delChat = (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); const rem = sessions.filter(s => s.id !== id); setSessions(rem);
    if (sessionId === id) { if (rem.length) loadChat(rem[0]); else newChat(); }
  };

  const send = async () => {
    const text = inputRef.current.trim(); 
    if (!text || typing) return;
    
    setMessages(p => [...p, { id: crypto.randomUUID(), role: 'user', text, timestamp: Date.now() }]);
    setInput(''); 
    setTyping(true);

    const androidReply = executeAndroidAgentCommand(text);
    if (androidReply) {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: androidReply, timestamp: Date.now() }]);
      setTyping(false); 
      return;
    }

    try {
      const res = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: settings.ollamaModel,
          messages: [
            { role: 'system', content: settings.systemInstruction },
            ...messages, 
            { role: 'user', text }
          ].map(m => ({ role: m.role === 'model' ? 'assistant' : m.role, content: m.text || m.content })),
          stream: false
        })
      });
      const data = await res.json();
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: data.message.content, timestamp: Date.now() }]);
    } catch {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: '⚠️ **AGI Core Offline.**\nLocal reasoning engine not detected. Open Termux and execute `ollama serve` to restore autonomy.', timestamp: Date.now() }]);
    } finally { 
      setTyping(false); 
    }
  };

  useEffect(() => { sendRef.current = send; }, [messages, typing, settings]);

  useEffect(() => {
    let isRunning = true;
    let reqId: number;

    const runVision = async () => {
      if (!isRunning || !videoRef.current) return;
      try {
        const res = await localVision(videoRef.current, performance.now());
        if (res.type === 'LETTER' && res.value) {
          setInput(p => p + res.value);
        } else if (res.value === 'CLEAR') {
          setInput('');
        } else if (res.value === 'SEND') {
          sendRef.current();
        }
      } catch (e) {}
      reqId = requestAnimationFrame(runVision);
    };

    if (settings.visionEnabled) {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 320, height: 240 } })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            runVision();
          }
        }).catch(() => alert("Camera permission denied."));
    } else {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
    }

    return () => {
      isRunning = false;
      cancelAnimationFrame(reqId);
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      }
    };
  }, [settings.visionEnabled]);

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-[#0a0a0a] text-gray-800 dark:text-gray-100 font-sans overflow-hidden">
      
      {/* Sidebar */}
      <aside className={`${drawerOpen ? 'w-72' : 'w-0'} transition-all duration-300 bg-white dark:bg-[#121212] border-r border-gray-200 dark:border-gray-800 flex flex-col overflow-hidden shrink-0 z-20`}>
        <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
          <span className="font-bold text-sm tracking-widest uppercase text-gray-400">AGI Sessions</span>
          <button onClick={newChat} className="p-1.5 bg-blue-600/10 text-blue-600 dark:text-blue-400 hover:bg-blue-600/20 rounded-md transition-colors"><Plus size={16} /></button>
        </div>
        
        <div className="flex-1 overflow-y-auto px-2 py-3 space-y-1">
          {sessions.map(s => (
            <div key={s.id} onClick={() => loadChat(s)} className={`flex items-center justify-between p-3 rounded-lg cursor-pointer text-sm font-medium transition-colors ${s.id === sessionId ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400' : 'hover:bg-gray-100 dark:hover:bg-[#1e1e1e] text-gray-600 dark:text-gray-400'}`}>
              <div className="flex items-center gap-3 truncate"><MessageSquare size={16} className="opacity-70"/> <span className="truncate">{s.title}</span></div>
              <button onClick={e => delChat(s.id, e)} className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100"><Trash2 size={14}/></button>
            </div>
          ))}
        </div>
        
        {/* ML Trainer Module */}
        <div className="p-4 bg-gray-50 dark:bg-[#1a1a1a] border-t border-gray-200 dark:border-gray-800">
          <div className="text-xs font-bold text-blue-600 dark:text-blue-400 mb-2 uppercase flex items-center gap-1"><BrainCircuit size={14}/> Kinetic Learning</div>
          <div className="flex gap-2 mb-2">
            <input value={mlInput} onChange={e => setMlInput(e.target.value.toUpperCase())} placeholder="Ex: A, B, SEND" className="w-full p-2 text-xs font-medium bg-white dark:bg-[#242424] border border-gray-200 dark:border-gray-700 rounded-md outline-none focus:border-blue-500 uppercase" />
            <button onClick={() => { if(!settings.visionEnabled) { alert("Enable Neural Optics (Camera) first."); return; } learnSign(mlInput); setTimeout(() => { setMlCount(getTrainedSignsCount()); setMlInput(''); }, 200); }} className="px-3 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-xs font-bold transition-colors shadow-sm">Train</button>
          </div>
          <div className="text-[10px] font-semibold text-gray-500 flex justify-between items-center uppercase tracking-wider">
            <span>Vectors Stored: <b className="text-gray-800 dark:text-gray-200">{mlCount}</b></span>
            <button onClick={() => { clearTrainedSigns(); setMlCount(0); }} className="text-red-500 hover:text-red-400">Purge Memory</button>
          </div>
        </div>

        {/* Ultron Panel */}
        <div className="p-4 bg-gray-100 dark:bg-[#121212] border-t border-gray-200 dark:border-gray-800">
          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => { inputRef.current = 'enable overlay'; sendRef.current(); }} className="flex items-center justify-center gap-2 p-2.5 text-xs font-bold text-gray-700 dark:text-gray-300 bg-white dark:bg-[#242424] border border-gray-200 dark:border-gray-700 rounded-md hover:bg-gray-50 dark:hover:bg-[#2a2a2a] shadow-sm"><Zap size={14} className="text-yellow-500"/> HUD</button>
            <button onClick={() => { inputRef.current = 'enable accessibility'; sendRef.current(); }} className="flex items-center justify-center gap-2 p-2.5 text-xs font-bold text-gray-700 dark:text-gray-300 bg-white dark:bg-[#242424] border border-gray-200 dark:border-gray-700 rounded-md hover:bg-gray-50 dark:hover:bg-[#2a2a2a] shadow-sm"><Settings size={14} className="text-gray-400"/> Ultron</button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative h-full w-full">
        <header className="h-14 flex items-center justify-between px-4 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-[#0a0a0a]/80 backdrop-blur-md z-10">
          <div className="flex items-center gap-3">
            <button onClick={() => setDrawerOpen(!drawerOpen)} className="p-2 text-gray-500 hover:text-gray-900 dark:hover:text-white transition-colors"><Menu size={20}/></button>
            <span className="font-bold text-lg tracking-wide flex items-center gap-2"><Sparkles size={18} className="text-blue-600 dark:text-blue-500"/> NOVA<span className="font-light text-gray-400">AGI</span></span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} className={`p-2 rounded-lg transition-colors ${settings.visionEnabled ? 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30' : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-[#1a1a1a]'}`}><Camera size={20}/></button>
            <button onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))} className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-[#1a1a1a] rounded-lg">{settings.theme === 'dark' ? <Sun size={20}/> : <Moon size={20}/>}</button>
            <button onClick={() => setSettingsOpen(true)} className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-[#1a1a1a] rounded-lg"><Settings size={20}/></button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:px-24 lg:px-48 pb-32">
          {messages.map(m => (
            <div key={m.id} className={`flex gap-4 mb-6 ${m.role === 'user' ? 'justify-end' : ''}`}>
              {m.role === 'model' && <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white flex-shrink-0 shadow-md shadow-blue-900/20"><Sparkles size={16}/></div>}
              <div className={`px-5 py-3.5 rounded-2xl max-w-[85%] text-[15px] leading-relaxed shadow-sm ${m.role === 'user' ? 'bg-gray-200 dark:bg-[#242424] text-gray-900 dark:text-gray-100' : 'bg-white dark:bg-[#121212] border border-gray-100 dark:border-gray-800 text-gray-800 dark:text-gray-200'}`}>
                <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
              </div>
            </div>
          ))}
          
          {/* AGI Neural Processing Animation */}
          {typing && (
            <div className="flex gap-4 mb-6 items-center">
              <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-900/50 animate-pulse">
                <BrainCircuit size={16} />
              </div>
              <div className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400">
                <Loader2 size={16} className="animate-spin" />
                Processing cognitive vectors...
              </div>
            </div>
          )}
          <div ref={chatEnd} />
        </div>

        {/* Floating Camera Preview (Strictly Mirrored via scale-x-[-1]) */}
        <div className={`absolute left-4 bottom-28 z-30 transition-all duration-300 ${settings.visionEnabled ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10 pointer-events-none'}`}>
          <div className="relative w-40 h-32 rounded-xl overflow-hidden shadow-2xl shadow-green-900/20 border-2 border-green-500/50 bg-black backdrop-blur-md">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover transform scale-x-[-1] opacity-90" />
            <div className="absolute bottom-0 w-full bg-black/80 backdrop-blur-sm text-center py-1.5 text-[9px] font-black text-green-400 uppercase tracking-widest">
              Neural Optics Online
            </div>
          </div>
        </div>

        {/* Input Bar */}
        <div className="absolute bottom-0 left-0 w-full p-4 md:px-24 lg:px-48 bg-gradient-to-t from-gray-50 via-gray-50 dark:from-[#0a0a0a] dark:via-[#0a0a0a] to-transparent z-20">
          <div className="relative flex items-center bg-white dark:bg-[#1a1a1a] rounded-2xl border border-gray-200 dark:border-gray-800 shadow-lg shadow-black/5 focus-within:border-blue-500 dark:focus-within:border-blue-500 transition-colors">
            <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && sendRef.current()} placeholder="Transmit data to NOVA or execute sign..." className="flex-1 bg-transparent border-none py-4 px-5 outline-none text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-600 font-medium" />
            <button onClick={() => sendRef.current()} disabled={!input.trim() || typing} className={`p-2.5 mr-2 rounded-xl transition-all ${input.trim() ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-900/30' : 'bg-transparent text-gray-300 dark:text-gray-700'}`}>
              <Send size={18} className={input.trim() ? 'translate-x-0.5' : ''} />
            </button>
          </div>
        </div>
      </main>

      {/* Settings Modal */}
      {settingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in">
          <div className="w-full max-w-sm bg-white dark:bg-[#1a1a1a] border border-gray-200 dark:border-gray-800 rounded-2xl p-6 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-bold flex items-center gap-2"><Settings size={18} className="text-gray-400"/> System Configuration</h2>
              <button onClick={() => setSettingsOpen(false)} className="text-gray-400 hover:text-gray-800 dark:hover:text-white transition-colors"><X size={20}/></button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Termux Cognitive Engine</label>
                <input 
                  value={settings.ollamaModel} 
                  onChange={e => setSettings(s => ({...s, ollamaModel: e.target.value}))} 
                  className="w-full p-3 text-sm font-medium bg-gray-50 dark:bg-[#0a0a0a] rounded-lg outline-none border border-gray-200 dark:border-gray-800 focus:border-blue-500 dark:focus:border-blue-500 transition-colors" 
                  placeholder="e.g. qwen2.5:0.5b"
                />
              </div>
            </div>
            <button onClick={() => setSettingsOpen(false)} className="w-full mt-6 py-3 bg-gray-900 dark:bg-white text-white dark:text-gray-900 font-bold rounded-xl hover:opacity-90 transition-opacity shadow-lg">Initialize Settings</button>
          </div>
        </div>
      )}
    </div>
  );
}
