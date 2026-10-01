import React, { useEffect, useRef, useState } from 'react';
import { User, Send, Settings, Sparkles, Camera, Cpu, X, Save, Volume2, VolumeX, Loader2, Trash2 } from 'lucide-react';
import { learnSign, clearTrainedSigns, getTrainedSignsCount, localVision } from './vision/localVision';
import './styles.css';

interface Message { id: string; role: 'user' | 'model'; content: string; timestamp: number; }
interface AppSettings { theme: 'light' | 'dark'; voiceEnabled: boolean; ollamaModel: string; systemInstruction: string; }

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
    const saved = localStorage.getItem('ai_settings');
    const defaults: AppSettings = { theme: 'dark', voiceEnabled: true, ollamaModel: 'qwen2.5:0.5b', systemInstruction: 'You are GestureGenius AI.' };
    return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
  });

  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'model', content: "Hello! I am your GestureGenius AI. \n\nI can read Sign Language (ASL) and respond to air gestures.\n\n  Show me 'A', 'B', 'C' to type.\n  Pinch your fingers to use the Wireless Mouse.\n  Show 'Thumb Down' to clear text.", timestamp: Date.now() }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  
  // Settings Panel Local State
  const [localSettings, setLocalSettings] = useState(settings);
  const [mlInput, setMlInput] = useState('');
  const [mlCount, setMlCount] = useState(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const inputRef = useRef(input);
  const sendRef = useRef<() => void>(() => {});

  useEffect(() => { inputRef.current = input; }, [input]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isTyping]);
  useEffect(() => {
    localStorage.setItem('ai_settings', JSON.stringify(settings));
    if (settings.theme === 'dark') document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [settings]);

  useEffect(() => { if (showSettings) { setLocalSettings(settings); setMlCount(getTrainedSignsCount()); } }, [showSettings, settings]);

  const handleSendMessage = async () => {
    const text = inputRef.current.trim();
    if (!text || isTyping) return;

    setMessages(prev => [...prev, { id: Date.now().toString(), role: 'user', content: text, timestamp: Date.now() }]);
    setInput('');
    setIsTyping(true);

    const androidReply = executeAndroidAgentCommand(text);
    if (androidReply) {
      setMessages(p => [...p, { id: Date.now().toString(), role: 'model', content: androidReply, timestamp: Date.now() }]);
      setIsTyping(false);
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
            ...messages.map(m => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.content })),
            { role: 'user', content: text }
          ],
          stream: false
        })
      });
      const data = await res.json();
      setMessages(p => [...p, { id: Date.now().toString(), role: 'model', content: data.message.content, timestamp: Date.now() }]);
    } catch {
      setMessages(p => [...p, { id: Date.now().toString(), role: 'model', content: '⚠️ Offline AI Not Reachable. Open Termux and run `ollama serve`.', timestamp: Date.now() }]);
    } finally {
      setIsTyping(false);
    }
  };

  useEffect(() => { sendRef.current = handleSendMessage; }, [messages, isTyping, settings]);

  useEffect(() => {
    let isRunning = true;
    let reqId: number;

    const runVision = async () => {
      if (!isRunning || !videoRef.current) return;
      try {
        const res = await localVision(videoRef.current, performance.now());
        if (res.type === 'LETTER' && res.value) setInput(p => p + res.value);
        else if (res.value === 'CLEAR') setInput('');
        else if (res.value === 'SEND') sendRef.current();
      } catch (e) {}
      reqId = requestAnimationFrame(runVision);
    };

    if (isCameraActive) {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 320, height: 240 } })
        .then(stream => { if (videoRef.current) { videoRef.current.srcObject = stream; runVision(); } })
        .catch(() => alert("Camera permission denied."));
    } else {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
    }
    return () => { isRunning = false; cancelAnimationFrame(reqId); if (videoRef.current?.srcObject) (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop()); };
  }, [isCameraActive]);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 transition-colors duration-500">
      
      {/* Header */}
      <header className="flex-none p-4 flex justify-between items-center glass-panel z-10 mx-4 mt-4 rounded-2xl shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-500/30">
            <Sparkles className="text-white w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="font-bold text-lg dark:text-white leading-tight">GestureGenius</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">Sign Language & Air Gestures</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setIsCameraActive(!isCameraActive)} className={`p-2 rounded-xl transition-all flex items-center gap-2 ${isCameraActive ? 'bg-indigo-600 text-white shadow-indigo-500/20 shadow-lg' : 'hover:bg-black/5 dark:hover:bg-white/10 dark:text-slate-300'}`}>
            <Camera size={20} />
            <span className="hidden md:inline text-xs font-medium">{isCameraActive ? "ON" : "OFF"}</span>
          </button>
          <button onClick={() => setShowSettings(true)} className="p-2 hover:bg-black/5 dark:hover:bg-white/10 rounded-xl dark:text-slate-300 transition-colors">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative max-w-5xl w-full mx-auto overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-hide pb-32">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex items-start gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'user' ? 'bg-indigo-500 text-white' : 'bg-pink-500 text-white'}`}>
                {msg.role === 'user' ? <User size={16} /> : <Cpu size={16} />}
              </div>
              <div className={`max-w-[80%] rounded-2xl p-4 shadow-sm relative ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-tl-none border border-slate-100 dark:border-slate-700'}`}>
                <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
              </div>
            </div>
          ))}
          
          {/* NEW THINKING ANIMATION */}
          {isTyping && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-pink-500 text-white flex items-center justify-center">
                <Cpu size={16} />
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-2xl rounded-tl-none p-4 border border-slate-100 dark:border-slate-700 flex items-center gap-3 text-pink-500 dark:text-pink-400 font-medium">
                <Loader2 size={18} className="animate-spin" />
                Thinking...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Floating Camera Preview */}
        <div className={`absolute left-4 bottom-24 transition-all duration-300 z-20 ${isCameraActive ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10 pointer-events-none'}`}>
          <div className="relative overflow-hidden rounded-xl border-2 shadow-lg bg-black border-indigo-500">
             <video ref={videoRef} autoPlay playsInline muted className="w-48 h-36 object-cover transform scale-x-[-1]" />
             <div className="absolute bottom-0 left-0 right-0 p-2 text-center bg-gradient-to-t from-black/80 to-transparent">
               <p className="text-[10px] text-white font-medium uppercase tracking-widest truncate px-1">Scanning...</p>
             </div>
          </div>
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 w-full p-4 mb-2">
          <div className="glass-panel p-2 rounded-2xl flex items-end gap-2 shadow-lg dark:shadow-black/20 relative z-30">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendRef.current()}
              placeholder={isCameraActive ? "Sign letters or gestures..." : "Type your message..."}
              className="flex-1 bg-transparent border-none focus:ring-0 p-3 max-h-32 dark:text-white placeholder-slate-400 outline-none"
            />
            <button onClick={() => setSettings(prev => ({...prev, voiceEnabled: !prev.voiceEnabled}))} className={`p-3 rounded-xl transition-colors ${settings.voiceEnabled ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400'}`}>
              {settings.voiceEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            </button>
            <button onClick={() => sendRef.current()} disabled={!input.trim() || isTyping} className="p-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-400 rounded-xl text-white transition-all active:scale-95 shadow-lg shadow-indigo-500/25">
              <Send size={20} />
            </button>
          </div>
        </div>
      </main>

      {/* Settings Panel */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="glass-panel w-full max-w-lg rounded-2xl shadow-2xl p-6 text-slate-900 dark:text-white transform transition-all scale-100 max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold">Personalize AI</h2>
              <button onClick={() => setShowSettings(false)} className="p-1 hover:bg-white/10 rounded-full"><X size={24} /></button>
            </div>
            
            <div className="space-y-6">
              <section className="space-y-2">
                <h3 className="text-sm font-semibold uppercase opacity-70 mb-2">Preferences</h3>
                <div className="flex items-center justify-between p-3 rounded-lg bg-white/50 dark:bg-slate-800/50">
                  <span className="text-sm font-medium">Dark Mode</span>
                  <button onClick={() => setLocalSettings({...localSettings, theme: localSettings.theme === 'dark' ? 'light' : 'dark'})} className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localSettings.theme === 'dark' ? 'bg-indigo-600' : 'bg-slate-400'}`}>
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localSettings.theme === 'dark' ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>
                
                <div>
                  <span className="block text-sm font-medium mb-1">Termux Offline Model</span>
                  <input value={localSettings.ollamaModel} onChange={e => setLocalSettings({...localSettings, ollamaModel: e.target.value})} className="w-full p-2 bg-white/50 dark:bg-slate-800/50 rounded outline-none border border-slate-300 dark:border-slate-600 text-sm" placeholder="qwen2.5:0.5b" />
                </div>
              </section>

              {/* Gesture Lab (ML Trainer) */}
              <section>
                <h3 className="text-sm font-semibold uppercase opacity-70 mb-2">Gesture Lab (ML)</h3>
                <div className="bg-white/50 dark:bg-slate-800/50 rounded-lg p-3 space-y-3">
                  <p className="text-xs text-slate-500">Hold your hand to the camera, type a letter, and tap Learn.</p>
                  <div className="flex gap-2">
                    <input value={mlInput} onChange={e => setMlInput(e.target.value.toUpperCase())} placeholder="Ex: A, B, SEND" className="flex-1 p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 outline-none text-sm uppercase" />
                    <button onClick={() => { if(!isCameraActive) { alert("Turn Camera ON first!"); return; } learnSign(mlInput); setTimeout(() => { setMlCount(getTrainedSignsCount()); setMlInput(''); }, 200); }} className="px-4 bg-indigo-600 text-white rounded text-sm font-bold">Learn</button>
                  </div>
                  <div className="flex justify-between items-center text-xs text-slate-500 pt-2">
                    <span>{mlCount} signs memorized</span>
                    <button onClick={() => { clearTrainedSigns(); setMlCount(0); }} className="text-red-500 hover:text-red-400 flex items-center gap-1"><Trash2 size={12}/> Clear</button>
                  </div>
                </div>
              </section>
            </div>

            <div className="mt-8 flex justify-end">
              <button onClick={() => { setSettings(localSettings); setShowSettings(false); }} className="flex items-center gap-2 px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors font-medium shadow-lg shadow-indigo-500/30">
                <Save size={18} /> Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
