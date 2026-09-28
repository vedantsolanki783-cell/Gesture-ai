import React, { useState } from 'react';
import { X, Save, Trash2, Cpu, Cloud, Info } from 'lucide-react';
import type { AppSettings, CustomGesture } from '../types';

interface Props { settings: AppSettings; onUpdate: (s: AppSettings) => void; onClose: () => void; }

export function SettingsPanel({ settings, onUpdate, onClose }: Props) {
  const [local, setLocal] = useState(settings);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [action, setAction] = useState<CustomGesture['action']>('SEND_MESSAGE');
  const patch = (p: Partial<AppSettings>) => setLocal(s => ({ ...s, ...p }));
  const addGesture = () => {
    if (!name.trim()) return;
    const g: CustomGesture = { id: crypto.randomUUID(), name: name.trim(), description: description.trim() || 'User-defined hand gesture', action };
    patch({ customGestures: [...local.customGestures, g] }); setName(''); setDescription('');
  };
  return <div className="modal-backdrop"><section className="settings">
    <div className="settings-head"><div><b>System Settings</b><span>Unified NOVA + GestureGenius controls</span></div><button onClick={onClose}><X/></button></div>
    <label>AI routing<select value={local.aiProvider} onChange={e => patch({ aiProvider: e.target.value as AppSettings['aiProvider'] })}><option value="cloudFree">Free-tier cloud (Groq/OpenRouter)</option><option value="webllm">On-device (offline, in-browser)</option><option value="ollama">Local first — Ollama</option><option value="auto">Local → free cloud → Gemini</option><option value="gemini">Gemini cloud</option></select></label>
    <label>On-device model<select value={local.webllmModel} onChange={e => patch({ webllmModel: e.target.value })}><option value="Qwen2.5-0.5B-Instruct-q4f16_1-MLC">Qwen2.5 0.5B (smallest, ~400MB, best chance on a tablet)</option><option value="Llama-3.2-1B-Instruct-q4f16_1-MLC">Llama 3.2 1B (~900MB, smarter but heavier)</option></select></label>
    <div className="info"><Cpu size={16}/><span>On-device: first reply downloads the model (needs internet + patience, one time only, ~400MB–900MB). After that it answers with zero internet, forever. Needs a fairly recent Chrome browser.</span></div>
    <div className="grid2"><label>Free-tier API base URL<input value={local.cloudFreeBaseUrl} onChange={e => patch({ cloudFreeBaseUrl: e.target.value })} placeholder="https://api.groq.com/openai/v1"/></label><label>Free-tier model<input value={local.cloudFreeModel} onChange={e => patch({ cloudFreeModel: e.target.value })} placeholder="llama-3.3-70b-versatile"/></label></div>
    <label>Free-tier API key<input type="password" value={local.cloudFreeApiKey} onChange={e => patch({ cloudFreeApiKey: e.target.value })} placeholder="Paste your Groq or OpenRouter key"/></label>
    <div className="info"><Cloud size={16}/><span>Get a free key at console.groq.com or openrouter.ai — this key is yours, stays only on this device, and is sent only to that provider.</span></div>
    <div className="grid2"><label>Local endpoint<input value={local.ollamaUrl} onChange={e => patch({ ollamaUrl: e.target.value })}/></label><label>Local model<input value={local.ollamaModel} onChange={e => patch({ ollamaModel: e.target.value })}/></label></div>
    <div className="grid2"><label>Gemini model<input value={local.geminiModel} onChange={e => patch({ geminiModel: e.target.value })}/></label><label>Vision confidence<input type="number" min="0.5" max="0.99" step="0.01" value={local.confidenceThreshold} onChange={e => patch({ confidenceThreshold: Number(e.target.value) })}/></label></div>
    <label>System instruction<textarea rows={3} value={local.systemInstruction} onChange={e => patch({ systemInstruction: e.target.value })}/></label>
    <div className="info"><Cpu size={16}/><span>Ollama needs a computer running in the background — skip it if you're on phone/tablet only.</span></div>
    <div className="gesture-lab"><div className="section-title">Gesture Lab</div><div className="grid2"><input placeholder="Gesture name" value={name} onChange={e => setName(e.target.value)}/><select value={action} onChange={e => setAction(e.target.value as CustomGesture['action'])}><option value="SEND_MESSAGE">Send message</option><option value="CLEAR">Clear input</option><option value="THEME_SWITCH">Switch theme</option><option value="COPY_LAST">Copy last</option><option value="TOGGLE_MIC">Toggle mic</option></select></div><input placeholder="Description for cloud fallback" value={description} onChange={e => setDescription(e.target.value)}/><button className="secondary" onClick={addGesture}>Add custom gesture</button>{local.customGestures.map(g => <div className="gesture-row" key={g.id}><div><b>{g.name}</b><small>{g.description}</small></div><button onClick={() => patch({ customGestures: local.customGestures.filter(x => x.id !== g.id) })}><Trash2 size={15}/></button></div>)}</div>
    <div className="settings-foot"><button className="secondary" onClick={onClose}>Cancel</button><button className="primary" onClick={() => { onUpdate(local); onClose(); }}><Save size={16}/> Save settings</button></div>
    <div className="info muted"><Info size={15}/><span>The offline hand model is loaded from <code>/public/models/hand_landmarker.task</code>. The repository includes the adapter but not the binary model.</span></div>
  </section></div>;
}
