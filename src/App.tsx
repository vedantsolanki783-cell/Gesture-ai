import React, { useEffect, useRef, useState } from 'react';
import { Bot, User, Send, Settings, Sparkles, BrainCircuit, Plus, MessageSquare, Trash2, Menu, Camera, Sun, Moon, X } from 'lucide-react';
import { learnSign, clearTrainedSigns, getTrainedSignsCount, localVision } from './vision/localVision';
import './styles.css';

// ============================================================================
// TYPES & SYSTEM COMMANDS
// ============================================================================
interface Message { id: string; role: 'user' | 'model'; text: string; timestamp: number; }
interface ChatSession { id: string; title: string; messages: Message[]; }
interface AppSettings { theme: 'light' | 'dark'; visionEnabled: boolean; ollamaModel: string; }

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
    return '⚠️ **Android Bridge Not Found:** You must be inside the NOVA APK to control the hardware.';
  }

  if (isOverlay) return bridge.enableOverlayBubble() === 'OPENED_OVERLAY_SETTINGS' ? '⚡ Opening Settings...' : '⚡ Floating Bubble Active!';
  if (isUltron) { bridge.openAccessibilitySettings(); return '🤖 Opening Accessibility Settings...'; }
  if (openAppMatch && bridge.openApp(openAppMatch[1].trim())) return `🚀 Launching **${openAppMatch[1].toUpperCase()}**...`;
  
  if (bridge && !bridge.isUltronConnected()) {
    bridge.openAccessibilitySettings();
    return '⚠️ Ultron OFF. Turn it ON in Accessibility Settings.';
  }

  if (isHome) { bridge.globalAction('HOME'); return '🏠 Executed **HOME**.'; }
  if (isBack) { bridge.globalAction('BACK'); return '🔙 Executed **BACK**.'; }
  if (isRecents) { bridge.globalAction('RECENTS'); return '🗂️ Opened **Recents**.'; }
  if (isNotif) { bridge.globalAction('NOTIFICATIONS'); return '🔔 Pulled **Notifications**.'; }

  return null;
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================
export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try { return { theme: 'dark', visionEnabled: false, ollamaModel: 'qwen2.5:0.5b', ...JSON.parse(localStorage.getItem('nova_settings') || '{}') }; } 
    catch { return { theme: 'dark', visionEnabled: false, ollamaModel: 'qwen2.5:0.5b' }; }
  });

  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try { return JSON.parse(localStorage.getItem('nova_sessions') || '[]'); } 
    catch { return []; }
  });

  const [sessionId, setSessionId] = useState(() => sessions[0]?.id || crypto.randomUUID());
  const [messages, setMessages] = useState<Message[]>(() => sessions[0]?.messages || [
    { id: '1', role: 'model', text: '⚡ **NOVA Core Active**\n\n• Point & pinch to use the Wireless Mouse.\n• Open Termux offline to chat.\n• Train your hands in the ML sidebar.', timestamp: Date.now() }
  ]);
  
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  
  const [mlInput, setMlInput] = useState('');
  const [mlCount, setMlCount] = useState(0);
  
  const chatEnd = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  // Stable refs for the camera loop to access latest state without re-triggering the camera
  const inputRef = useRef(input);
  const sendRef = useRef<() => void>(() => {});

  // Sync state to refs
  useEffect(() => { inputRef.current = input; }, [input]);
  useEffect(() => { localStorage.setItem('nova_settings', JSON.stringify(settings)); document.documentElement.dataset.theme = settings.theme; }, [settings]);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, typing]);

  // Sync sessions
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

  // Initialize ML Count
  useEffect(() => {
    try { setTimeout(() => setMlCount(getTrainedSignsCount()), 500); } catch {}
  }, []);

  const newChat = () => { setSessionId(crypto.randomUUID()); setMessages([{ id: crypto.randomUUID(), role: 'model', text: 'How can I help you today?', timestamp: Date.now() }]); };
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
      // Direct Offline Termux Fetch
      const res = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: settings.ollamaModel,
          messages: [...messages, { role: 'user', text }].map(m => ({
            role: m.role === 'model' ? 'assistant' : 'user',
            content: m.text
          })),
          stream: false
        })
      });
      const data = await res.json();
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: data.message.content, timestamp: Date.now() }]);
    } catch {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: 'model', text: '⚠️ **Offline AI Not Reachable.**\nOpen Termux and run `ollama serve` to turn your offline brain back on.', timestamp: Date.now() }]);
    } finally { 
      setTyping(false); 
    }
  };

  // Sync send function to ref for the camera loop
  useEffect(() => { sendRef.current = send; }, [messages, typing, settings]);

  // Built-in Camera Loop (Zero missing imports)
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
    <div className="flex h-screen bg-white dark:bg-[#212121] text-gray-800 dark:text-gray-100 font-sans overflow-hidden">
      
      {/* Sidebar - Chat History & Controls */}
      <aside className={`${drawerOpen ? 'w-72' : 'w-0'} transition-all duration-300 bg-gray-50 dark:bg-[#171717] border-r border-gray-200 dark:border-gray-800 flex flex-col overflow-hidden shrink-0 z-20`}>
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
          <div className="text-xs font-bold text-blue-500 mb-2 uppercase">🧠 ML Sign Trainer</div>
          <div className="flex gap-2 mb-2">
            <input value={mlInput} onChange={e => setMlInput(e.target.value.toUpperCase())} placeholder="Ex: A, B, SEND" className="w-full p-2 text-sm bg-white dark:bg-[#2f2f2f] border border-gray-300 dark:border-gray-600 rounded outline-none uppercase" />
            <button onClick={() => { if(!settings.visionEnabled) { alert("Turn Camera ON first!"); return; } learnSign(mlInput); setTimeout(() => { setMlCount(getTrainedSignsCount()); setMlInput(''); }, 200); }} className="px-3 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium transition-colors">Learn</button>
          </div>
          <div className="text-xs text-gray-500 flex justify-between items-center">
            <span>Memory: <b className="text-gray-800 dark:text-gray-300">{mlCount}</b> saved</span>
            <button onClick={() => { clearTrainedSigns(); setMlCount(0); }} className="text-red-400 hover:underline">Erase</button>
          </div>
        </div>

        {/* Ultron Panel */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-800">
          <div className="text-xs font-bold text-gray-500 mb-2 uppercase">🤖 Android Controls</div>
          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => { inputRef.current = 'enable overlay'; sendRef.current(); }} className="p-2 text-xs font-medium bg-white dark:bg-[#2f2f2f] border border-gray-200 dark:border-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-800">⚡ HUD</button>
            <button onClick={() => { inputRef.current = 'enable accessibility'; sendRef.current(); }} className="p-2 text-xs font-medium bg-white dark:bg-[#2f2f2f] border border-gray-200 dark:border-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-800">⚙️ Ultron</button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative h-full w-full">
        <header className="h-14 flex items-center justify-between px-4 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-[#212121] z-10">
          <div className="flex items-center gap-3">
            <button onClick={() => setDrawerOpen(!drawerOpen)} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md"><Menu size={20}/></button>
            <span className="font-semibold text-lg flex items-center gap-2"><Sparkles size={18} className="text-blue-500"/> NOVA</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))} className={`p-2 rounded-md transition-colors ${settings.visionEnabled ? 'text-green-500 bg-green-50 dark:bg-green-900/20' : 'hover:bg-gray-100 dark:hover:bg-gray-800'}`}><Camera size={20}/></button>
            <button onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md">{settings.theme === 'dark' ? <Sun size={20}/> : <Moon size={20}/>}</button>
            <button onClick={() => setSettingsOpen(true)} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md"><Settings size={20}/></button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:px-24 lg:px-48 pb-32">
          {messages.map(m => (
            <div key={m.id} className={`flex gap-4 mb-6 ${m.role === 'user' ? 'justify-end' : ''}`}>
              {m.role === 'model' && <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white flex-shrink-0"><Bot size={18}/></div>}
              <div className={`px-4 py-3 rounded-2xl max-w-[85%] text-[15px] leading-relaxed shadow-sm ${m.role === 'user' ? 'bg-gray-100 dark:bg-[#2f2f2f]' : 'bg-transparent border border-gray-100 dark:border-gray-800'}`}>
                <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
              </div>
            </div>
          ))}
          {typing && <div className="flex gap-4 mb-6"><div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white"><Bot size={18}/></div><div className="px-4 py-3">● ● ●</div></div>}
          <div ref={chatEnd} />
        </div>

        {/* Floating Camera Preview */}
        <div className={`absolute left-4 bottom-28 z-30 transition-all duration-300 ${settings.visionEnabled ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10 pointer-events-none'}`}>
          <div className="relative w-40 h-32 rounded-xl overflow-hidden shadow-2xl border-2 border-green-500 bg-black">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover transform scale-x-[-1]" />
            <div className="absolute bottom-0 w-full bg-black/60 text-center py-1 text-[10px] font-bold text-green-400 uppercase tracking-widest">
              Camera Active
            </div>
          </div>
        </div>

        {/* Input Bar */}
        <div className="absolute bottom-0 left-0 w-full p-4 md:px-24 lg:px-48 bg-gradient-to-t from-white via-white dark:from-[#212121] dark:via-[#212121] to-transparent z-20">
          <div className="relative flex items-center bg-gray-100 dark:bg-[#2f2f2f] rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm focus-within:ring-1 focus-within:ring-gray-400">
            <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && sendRef.current()} placeholder="Message NOVA or sign..." className="flex-1 bg-transparent border-none py-4 px-5 outline-none" />
            <button onClick={() => sendRef.current()} disabled={!input.trim() || typing} className={`p-2 mr-2 rounded-lg transition-colors ${input.trim() ? 'bg-blue-600 text-white' : 'bg-transparent text-gray-400 dark:text-gray-500'}`}>
              <Send size={18} />
            </button>
          </div>
        </div>
      </main>

      {/* Settings Modal */}
      {settingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="w-full max-w-sm bg-white dark:bg-[#212121] rounded-2xl p-6 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-bold">System Settings</h2>
              <button onClick={() => setSettingsOpen(false)} className="text-gray-400 hover:text-gray-800 dark:hover:text-white"><X size={20}/></button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">Termux Ollama Model</label>
                <input 
                  value={settings.ollamaModel} 
                  onChange={e => setSettings(s => ({...s, ollamaModel: e.target.value}))} 
                  className="w-full p-3 bg-gray-100 dark:bg-[#2f2f2f] rounded-lg outline-none border border-transparent focus:border-blue-500" 
                  placeholder="e.g. qwen2.5:0.5b"
                />
              </div>
            </div>
            <button onClick={() => setSettingsOpen(false)} className="w-full mt-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl">Save & Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
