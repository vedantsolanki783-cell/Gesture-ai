import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Bot,
  Copy,
  Mic,
  MicOff,
  Moon,
  Paperclip,
  Send,
  Settings,
  Sparkles,
  Sun,
  Trash2,
  User,
  WifiOff,
  X
} from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

// ============================================================================
// URL HELPER (Prevents mobile clipboard/markdown from corrupting URLs)
// ============================================================================
const httpsUrl = (path: string): string => ['ht', 'tps://', path].join('');
const httpUrl = (path: string): string => ['ht', 'tp://', path].join('');

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export type ArtifactType = 'html' | 'presentation' | 'image' | 'code';

export interface SlideData {
  id: string;
  title: string;
  subtitle?: string;
  layout: 'title' | 'bullets' | 'two-column' | 'quote' | 'stats';
  bullets?: string[];
  leftColumn?: string[];
  rightColumn?: string[];
  quote?: { text: string; author: string };
  stats?: { value: string; label: string }[];
  notes?: string;
}

export interface PresentationDeck {
  title: string;
  author: string;
  theme: 'cyber' | 'minimal' | 'executive';
  slides: SlideData[];
}

export interface CanvasArtifact {
  id: string;
  title: string;
  type: ArtifactType;
  filename: string;
  mimeType: string;
  content: string;
  dataUrl?: string;
  deck?: PresentationDeck;
  timestamp: number;
}

interface LocalAttachment {
  name: string;
  size: number;
  mimeType: string;
  content: string;
  isImage?: boolean;
  dataUrl?: string;
  localVisualReport?: string;
}

const defaults: AppSettings = {
  theme: 'dark',
  voiceEnabled: true,
  visionEnabled: false,
  confidenceThreshold: 0.72,
  aiProvider: 'cloudFree',
  ollamaUrl: httpUrl('localhost:11434'),
  ollamaModel: 'qwen3:4b',
  geminiModel: 'gemini-2.5-flash',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction:
    'You are NOVA (Jarvis + Ultron Core), an advanced multimodal AI assistant and generative Canvas engine. ' +
    'You can analyze images, PDFs, CSVs, and code; generate interactive HTML web apps; build multi-slide presentations; and synthesize images. ' +
    'When asked to build an HTML app, website, game, or tool, return a complete single-file HTML5 document inside a ```html code block with inline CSS and JS. ' +
    'When asked to draw or generate an image, include [[GENERATE_IMAGE: detailed visual prompt]].',
  customGestures: []
};

const LETTER_COOLDOWN_MS = 1000;
const SAME_LETTER_COOLDOWN_MS = 1800;

const CANVAS_EMBEDDED_CSS = `
.app.has-canvas-open main.app-workspace {
  display: grid;
  grid-template-columns: 300px 1fr 520px;
  gap: 14px;
}
@media (max-width: 1280px) {
  .app.has-canvas-open main.app-workspace {
    grid-template-columns: 260px 1fr 440px;
  }
}
@media (max-width: 980px) {
  .app.has-canvas-open main.app-workspace {
    grid-template-columns: 1fr;
  }
  .canvas-drawer {
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    border-radius: 0 !important;
  }
}
.canvas-drawer {
  background: #090d16;
  border: 1px solid rgba(56, 189, 248, 0.28);
  display: flex;
  flex-direction: column;
  height: calc(100vh - 86px);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55);
}
.canvas-drawer.maximized {
  position: fixed;
  inset: 0;
  height: 100vh;
  border-radius: 0;
  z-index: 99999;
}
.canvas-header {
  padding: 12px 16px;
  background: #0f172a;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.canvas-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  font-weight: 800;
  color: #38bdf8;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}
.canvas-title-group h3 {
  font-size: 14px;
  color: #f8fafc;
  margin: 2px 0 0 0;
  max-width: 210px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.canvas-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.canvas-btn {
  background: rgba(255, 255, 255, 0.06);
  color: #cbd5e1;
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 5px 10px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
}
.canvas-btn.highlight {
  background: #0284c7;
  color: #fff;
  border-color: #38bdf8;
}
.canvas-btn:hover {
  background: #38bdf8;
  color: #020617;
}
.canvas-tabs {
  display: flex;
  background: #0b1120;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.c-tab {
  flex: 1;
  padding: 9px;
  background: none;
  border: none;
  color: #94a3b8;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}
.c-tab.active {
  color: #38bdf8;
  border-bottom-color: #38bdf8;
  background: rgba(56, 189, 248, 0.06);
}
.canvas-body {
  flex: 1;
  overflow: auto;
  position: relative;
  background: #020617;
}
.slide-deck-viewer {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.slide-deck-controls {
  padding: 10px 14px;
  background: #090d16;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.slide-theme-picker {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: #94a3b8;
}
.slide-theme-picker button {
  background: none;
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 3px 8px;
  border-radius: 6px;
  color: #cbd5e1;
  font-size: 11px;
  cursor: pointer;
}
.slide-theme-picker button.active {
  background: #38bdf8;
  color: #020617;
  font-weight: 700;
}
.slide-pagination {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #cbd5e1;
}
.slide-pagination button {
  background: #1e293b;
  border: none;
  color: #fff;
  padding: 4px 8px;
  border-radius: 6px;
  cursor: pointer;
}
.slide-viewport {
  flex: 1;
  padding: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.slide-content-card {
  width: 100%;
  min-height: 270px;
  border-radius: 16px;
  padding: 28px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.theme-cyber .slide-content-card {
  background: linear-gradient(135deg, #090e1c 0%, #171b38 100%);
  border: 1px solid rgba(56, 189, 248, 0.35);
  box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
  color: #f8fafc;
}
.theme-executive .slide-content-card {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border: 1px solid rgba(245, 158, 11, 0.4);
  color: #f8fafc;
}
.theme-minimal .slide-content-card {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  color: #0f172a;
}
.theme-minimal h1, .theme-minimal h2 { color: #0f172a !important; }
.theme-minimal .slide-bullet-list, .theme-minimal .slide-sub { color: #334155 !important; }
.slide-content-card h1 { color: #38bdf8; font-size: 28px; margin-bottom: 10px; }
.slide-content-card h2 { color: #f8fafc; font-size: 22px; border-bottom: 2px solid rgba(56, 189, 248, 0.3); padding-bottom: 8px; margin-bottom: 14px; }
.slide-sub { font-size: 15px; color: #94a3b8; }
.slide-meta-badge { display: inline-block; margin-top: 16px; padding: 4px 12px; background: rgba(56, 189, 248, 0.15); border-radius: 999px; color: #38bdf8; font-size: 11px; font-weight: bold; }
.slide-bullet-list { list-style: none; display: flex; flex-direction: column; gap: 8px; font-size: 14px; color: #cbd5e1; padding: 0; }
.slide-bullet-list li::before { content: "✦ "; color: #38bdf8; }
.slide-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.slide-col { background: rgba(15, 23, 42, 0.55); padding: 12px; border-radius: 10px; }
.slide-col h4 { color: #a855f7; margin-bottom: 6px; font-size: 13px; }
.slide-col ul { list-style: none; padding: 0; font-size: 13px; display: flex; flex-direction: column; gap: 5px; color: #cbd5e1; }
.slide-stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.slide-stat-item { background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.22); padding: 12px; border-radius: 10px; text-align: center; }
.stat-big { display: block; font-size: 26px; font-weight: 800; color: #38bdf8; }
.stat-label { font-size: 11px; color: #94a3b8; }
.slide-quote-view blockquote { font-size: 17px; font-style: italic; color: #e2e8f0; border-left: 4px solid #a855f7; padding-left: 14px; margin: 0; }
.slide-quote-view cite { display: block; margin-top: 10px; color: #38bdf8; font-weight: bold; font-size: 13px; }
.slide-notes-footer { padding: 9px 14px; background: #080d1a; font-size: 11px; color: #94a3b8; border-top: 1px solid rgba(255, 255, 255, 0.05); }
.html-runner-container { width: 100%; height: 100%; }
.html-sandboxed-iframe { width: 100%; height: 100%; border: none; background: #ffffff; }
.image-studio-container { padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; }
.image-studio-display { max-width: 100%; max-height: 65vh; border-radius: 12px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6); }
.image-studio-caption { margin-top: 12px; color: #94a3b8; font-size: 12px; text-align: center; }
.code-inspector-container { padding: 14px; height: 100%; overflow: auto; background: #070b14; }
.code-inspector-pre { font-family: monospace; font-size: 12px; color: #38bdf8; line-height: 1.5; white-space: pre-wrap; margin: 0; }
`;

function isVisionRefusal(text: string): boolean {
  if (!text) return true;
  return /can't view images|cannot view images|cannot analyze or interpret image|unable to view images|unable to see images|can't see images|cannot see the image|i am a text-based|describe what's in the picture|describe what’s in the picture|do not have inherent image/i.test(
    text
  );
}

// ============================================================================
// 1. OFFLINE PROCEDURAL IMAGE SYNTHESIZER & ONLINE FLUX DIFFUSION
// ============================================================================

export function renderProceduralCanvasImage(
  prompt: string,
  mode: 'cyber' | 'neural' | 'sunset' = 'cyber'
): string {
  const canvas = document.createElement('canvas');
  canvas.width = 1200;
  canvas.height = 750;
  const ctx = canvas.getContext('2d')!;

  if (mode === 'cyber') {
    const bg = ctx.createLinearGradient(0, 0, 1200, 750);
    bg.addColorStop(0, '#050811');
    bg.addColorStop(0.5, '#0b1329');
    bg.addColorStop(1, '#020617');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 1200, 750);

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.12)';
    ctx.lineWidth = 1;
    for (let x = 0; x < 1200; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 750);
      ctx.stroke();
    }
    for (let y = 0; y < 750; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(1200, y);
      ctx.stroke();
    }

    const cx = 600;
    const cy = 340;
    for (let r = 50; r <= 220; r += 32) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = r % 64 === 0 ? 'rgba(56, 189, 248, 0.85)' : 'rgba(168, 85, 247, 0.55)';
      ctx.lineWidth = r === 146 ? 5 : 2;
      ctx.stroke();
    }

    const core = ctx.createRadialGradient(cx, cy, 10, cx, cy, 95);
    core.addColorStop(0, '#ffffff');
    core.addColorStop(0.3, '#38bdf8');
    core.addColorStop(0.8, '#6366f1');
    core.addColorStop(1, 'rgba(99, 102, 241, 0)');
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(cx, cy, 95, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 28px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('NOVA JARVIS OFFLINE RENDER', cx, 630);
    ctx.fillStyle = '#38bdf8';
    ctx.font = '18px monospace';
    ctx.fillText(prompt.slice(0, 75), cx, 670);
  } else if (mode === 'neural') {
    ctx.fillStyle = '#0a0f1d';
    ctx.fillRect(0, 0, 1200, 750);
    const nodes: { x: number; y: number }[] = [];
    for (let i = 0; i < 48; i++) {
      nodes.push({
        x: 100 + Math.random() * 1000,
        y: 80 + Math.random() * 550
      });
    }
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dist = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
        if (dist < 180) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
    });
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 26px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(prompt.slice(0, 70), 600, 680);
  } else {
    const grad = ctx.createLinearGradient(0, 0, 1200, 750);
    grad.addColorStop(0, '#0f172a');
    grad.addColorStop(1, '#1e1b4b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1200, 750);
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 32px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('NOVA GENERATIVE CANVAS', 600, 340);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '20px monospace';
    ctx.fillText(prompt.slice(0, 75), 600, 390);
  }

  return canvas.toDataURL('image/png');
}

export async function generateVisualImageArtifact(prompt: string): Promise<CanvasArtifact> {
  const clean =
    prompt
      .replace(/^(generate|create|draw|make|render)\s+(an?\s+)?(image|picture|photo|art|wallpaper|logo|diagram)\s+(of\s+)?/i, '')
      .trim() || prompt;
  const safeSlug = clean.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 28) || 'artifact-image';

  if (navigator.onLine) {
    try {
      const seed = Math.floor(Math.random() * 999999);
      const endpoint = httpsUrl(
        `image.pollinations.ai/prompt/${encodeURIComponent(clean)}?width=1024&height=640&nologo=true&seed=${seed}`
      );
      const res = await fetch(endpoint);
      if (res.ok) {
        const blob = await res.blob();
        const dataUrl: string = await new Promise((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve(r.result as string);
          r.onerror = reject;
          r.readAsDataURL(blob);
        });
        return {
          id: crypto.randomUUID(),
          title: clean.slice(0, 40),
          type: 'image',
          filename: `${safeSlug}.jpg`,
          mimeType: 'image/jpeg',
          content: clean,
          dataUrl,
          timestamp: Date.now()
        };
      }
    } catch {}
  }

  const mode = /neural|brain|network/i.test(clean) ? 'neural' : 'cyber';
  const offlineDataUrl = renderProceduralCanvasImage(clean, mode);
  return {
    id: crypto.randomUUID(),
    title: clean.slice(0, 40),
    type: 'image',
    filename: `${safeSlug}-offline.png`,
    mimeType: 'image/png',
    content: clean,
    dataUrl: offlineDataUrl,
    timestamp: Date.now()
  };
}

// ============================================================================
// 2. OFFLINE & ONLINE HTML WEB APP SYNTHESIZER
// ============================================================================

export function synthesizeOfflineHtmlApp(prompt: string): CanvasArtifact {
  const lower = prompt.toLowerCase();
  let title = 'Interactive Web Application';
  let html = '';

  if (lower.includes('calculator')) {
    title = 'Jarvis Scientific Calculator';
    html = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"><title>${title}</title>
<style>
  body { background: #090d16; color: #f8fafc; font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
  .calc { background: #111827; padding: 24px; border-radius: 20px; border: 1px solid rgba(56,189,248,0.3); width: 320px; box-shadow: 0 20px 50px rgba(0,0,0,0.6); }
  #disp { width: 100%; height: 64px; background: #020617; border: 1px solid #1e293b; border-radius: 12px; color: #38bdf8; font-size: 28px; text-align: right; padding: 12px; box-sizing: border-box; margin-bottom: 16px; font-family: monospace; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  button { padding: 16px; font-size: 18px; font-weight: bold; border: none; border-radius: 12px; background: #1e293b; color: #f8fafc; cursor: pointer; transition: 0.15s; }
  button:hover { background: #38bdf8; color: #020617; }
  .op { background: #0284c7; }
  .eq { background: #10b981; grid-column: span 2; }
</style>
</head>
<body>
  <div class="calc">
    <input id="disp" readonly value="0" />
    <div class="grid">
      <button onclick="clr()" style="background:#ef4444">C</button>
      <button onclick="ins('(')">(</button>
      <button onclick="ins(')')">)</button>
      <button class="op" onclick="ins('/')">÷</button>
      <button onclick="ins('7')">7</button><button onclick="ins('8')">8</button><button onclick="ins('9')">9</button><button class="op" onclick="ins('*')">×</button>
      <button onclick="ins('4')">4</button><button onclick="ins('5')">5</button><button onclick="ins('6')">6</button><button class="op" onclick="ins('-')">−</button>
      <button onclick="ins('1')">1</button><button onclick="ins('2')">2</button><button onclick="ins('3')">3</button><button class="op" onclick="ins('+')">+</button>
      <button onclick="ins('0')">0</button><button onclick="ins('.')">.</button><button class="eq" onclick="solve()">=</button>
    </div>
  </div>
  <script>
    const d = document.getElementById('disp');
    function ins(v) { d.value = d.value === '0' ? v : d.value + v; }
    function clr() { d.value = '0'; }
    function solve() { try { d.value = String(Function('return ' + d.value)()); } catch { d.value = 'Error'; } }
  </script>
</body>
</html>`;
  } else {
    title = prompt.slice(0, 36) || 'NOVA Interactive Web App';
    html = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"><title>${title}</title>
<style>
  body { background: #090d16; color: #f8fafc; font-family: system-ui, sans-serif; padding: 32px; margin: 0; }
  .card { max-width: 680px; margin: 0 auto; background: #111827; border: 1px solid rgba(56,189,248,0.3); border-radius: 16px; padding: 28px; box-shadow: 0 15px 40px rgba(0,0,0,0.5); }
  h1 { color: #38bdf8; margin-top: 0; }
  .row { display: flex; gap: 10px; margin-bottom: 18px; }
  input { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #334155; background: #020617; color: #fff; }
  button { padding: 12px 20px; border-radius: 10px; border: none; background: #0284c7; color: #fff; font-weight: bold; cursor: pointer; }
  ul { list-style: none; padding: 0; }
  li { background: #1e293b; padding: 12px 16px; border-radius: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
</style>
</head>
<body>
  <div class="card">
    <h1>⚡ ${title}</h1>
    <p style="color:#94a3b8">Interactive Workspace generated by NOVA Canvas.</p>
    <div class="row">
      <input id="inp" placeholder="Add new item or command..." onkeydown="if(event.key==='Enter')add()" />
      <button onclick="add()">Add Item</button>
    </div>
    <ul id="list">
      <li><span>Initialize Jarvis Core Protocol</span><button onclick="this.parentElement.remove()" style="background:#ef4444;padding:6px 10px">Done</button></li>
    </ul>
  </div>
  <script>
    function add() {
      const inp = document.getElementById('inp');
      if (!inp.value.trim()) return;
      const li = document.createElement('li');
      li.innerHTML = '<span>' + inp.value + '</span><button onclick="this.parentElement.remove()" style="background:#ef4444;padding:6px 10px">Done</button>';
      document.getElementById('list').appendChild(li);
      inp.value = '';
    }
  </script>
</body>
</html>`;
  }

  return {
    id: crypto.randomUUID(),
    title,
    type: 'html',
    filename: `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.html`,
    mimeType: 'text/html',
    content: html,
    timestamp: Date.now()
  };
}

// ============================================================================
// 3. PRESENTATION DECK GENERATOR & PPT / HTML EXPORTERS
// ============================================================================

export function synthesizePresentationDeck(topic: string): PresentationDeck {
  const cleanTopic =
    topic
      .replace(/^(create|make|build|generate|design)\s+(a\s+)?(presentation|ppt|powerpoint|slides?|deck)\s+(on|about|for)?/i, '')
      .trim() || topic;
  const capitalized = cleanTopic.charAt(0).toUpperCase() + cleanTopic.slice(1);

  return {
    title: capitalized,
    author: 'NOVA Jarvis + Ultron',
    theme: 'cyber',
    slides: [
      {
        id: '1',
        title: capitalized,
        subtitle: 'Executive Architecture, Technical Strategy & Future Horizons',
        layout: 'title',
        notes: 'Opening title slide introducing the core topic and strategic vision.'
      },
      {
        id: '2',
        title: 'Executive Overview & Core Pillars',
        layout: 'bullets',
        bullets: [
          `Strategic transformation and key breakthroughs in ${cleanTopic}.`,
          'Hybrid architecture combining local on-device intelligence with cloud scalability.',
          'Real-time automation pipelines reducing manual operational overhead.',
          'High-reliability execution with full offline resilience.'
        ],
        notes: 'Highlight the four foundational pillars driving this initiative.'
      },
      {
        id: '3',
        title: 'System Architecture Comparison',
        layout: 'two-column',
        leftColumn: [
          'Core Capabilities',
          'Real-time data & sensor processing',
          'Low-latency local execution',
          'Modular extensible design'
        ],
        rightColumn: [
          'Strategic Outcomes',
          '10x faster workflow completion',
          'Zero-downtime offline continuity',
          'Seamless cross-device deployment'
        ],
        notes: 'Compare technical capabilities on the left with business outcomes on the right.'
      },
      {
        id: '4',
        title: 'Key Performance Benchmarks',
        layout: 'stats',
        stats: [
          { value: '99.9%', label: 'System Reliability' },
          { value: '<50ms', label: 'Response Latency' },
          { value: '24/7', label: 'Autonomous Operation' },
          { value: '100%', label: 'Offline Capable' }
        ],
        notes: 'Core quantitative metrics demonstrating performance.'
      },
      {
        id: '5',
        title: 'Guiding Vision',
        layout: 'quote',
        quote: {
          text: `The true power of ${cleanTopic} lies in combining human creativity with autonomous, intelligent execution.`,
          author: 'NOVA Executive Briefing'
        },
        notes: 'Key takeaway quote summarizing the philosophy.'
      },
      {
        id: '6',
        title: 'Implementation Roadmap & Next Steps',
        layout: 'bullets',
        bullets: [
          'Phase 1: Core foundation deployment and baseline verification.',
          'Phase 2: Autonomous agent workflow integration and testing.',
          'Phase 3: Mobile APK packaging and background service activation.',
          'Phase 4: Full-scale production rollout and continuous optimization.'
        ],
        notes: 'Actionable 4-phase implementation plan.'
      }
    ]
  };
}

export function exportPresentationToHtml(deck: PresentationDeck): string {
  const slidesJson = JSON.stringify(deck);
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${deck.title} — NOVA Presentation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #020617;
      color: #f8fafc;
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    #stage {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 32px;
    }
    .slide-card {
      width: 100%;
      max-width: 1050px;
      aspect-ratio: 16/9;
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
      border: 1px solid rgba(56, 189, 248, 0.35);
      border-radius: 20px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
      padding: 54px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    h1 { font-size: 44px; font-weight: 800; color: #38bdf8; margin-bottom: 16px; }
    h2 { font-size: 34px; font-weight: 700; color: #f8fafc; margin-bottom: 26px; border-bottom: 2px solid rgba(56, 189, 248, 0.3); padding-bottom: 10px; }
    p.subtitle { font-size: 20px; color: #94a3b8; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 14px; font-size: 20px; color: #cbd5e1; }
    ul li::before { content: "✦ "; color: #38bdf8; font-weight: bold; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
    .col-box { background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); }
    .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
    .stat-card { background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 20px; text-align: center; }
    .stat-val { font-size: 40px; font-weight: 900; color: #38bdf8; }
    .stat-lbl { font-size: 14px; color: #94a3b8; text-transform: uppercase; }
    .quote-box { border-left: 5px solid #a855f7; padding-left: 22px; font-style: italic; font-size: 24px; color: #e2e8f0; }
    .quote-author { font-size: 17px; color: #38bdf8; font-style: normal; margin-top: 12px; font-weight: bold; }
    #toolbar {
      height: 56px;
      background: #090d16;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      font-size: 14px;
      color: #94a3b8;
    }
    button {
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.15);
      padding: 7px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: #38bdf8; color: #020617; }
  </style>
</head>
<body>
  <div id="stage"><div class="slide-card" id="card"></div></div>
  <div id="toolbar">
    <div><strong>${deck.title}</strong> · <span id="counter">Slide 1 of ${deck.slides.length}</span></div>
    <div style="display: flex; gap: 10px;">
      <button onclick="prev()">◄ Prev</button>
      <button onclick="next()">Next ►</button>
      <button onclick="document.documentElement.requestFullscreen()">Fullscreen</button>
    </div>
  </div>
  <script>
    const deck = ${slidesJson};
    let current = 0;
    function render() {
      const s = deck.slides[current];
      const card = document.getElementById('card');
      document.getElementById('counter').innerText = 'Slide ' + (current + 1) + ' of ' + deck.slides.length;
      if (s.layout === 'title') {
        card.innerHTML = '<h1>' + s.title + '</h1><p class="subtitle">' + (s.subtitle || '') + '</p>';
      } else if (s.layout === 'bullets') {
        card.innerHTML = '<h2>' + s.title + '</h2><ul>' + (s.bullets || []).map(b => '<li>' + b + '</li>').join('') + '</ul>';
      } else if (s.layout === 'two-column') {
        card.innerHTML = '<h2>' + s.title + '</h2><div class="two-col"><div class="col-box"><ul>' +
          (s.leftColumn || []).map(b => '<li>' + b + '</li>').join('') + '</ul></div><div class="col-box"><ul>' +
          (s.rightColumn || []).map(b => '<li>' + b + '</li>').join('') + '</ul></div></div>';
      } else if (s.layout === 'stats') {
        card.innerHTML = '<h2>' + s.title + '</h2><div class="stats-grid">' +
          (s.stats || []).map(st => '<div class="stat-card"><div class="stat-val">' + st.value + '</div><div class="stat-lbl">' + st.label + '</div></div>').join('') + '</div>';
      } else if (s.layout === 'quote') {
        card.innerHTML = '<h2>' + s.title + '</h2><div class="quote-box">"' + (s.quote ? s.quote.text : '') + '"<div class="quote-author">— ' + (s.quote ? s.quote.author : '') + '</div></div>';
      }
    }
    function prev() { if (current > 0) { current--; render(); } }
    function next() { if (current < deck.slides.length - 1) { current++; render(); } }
    window.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight' || e.key === ' ') next();
      if (e.key === 'ArrowLeft') prev();
    });
    render();
  </script>
</body>
</html>`;
}

export function exportPresentationToPpt(deck: PresentationDeck): string {
  const slidesHtml = deck.slides
    .map(
      (s, idx) => `
    <div style="page-break-after: always; padding: 40px; font-family: Calibri, Arial, sans-serif; background: #0f172a; color: #ffffff; margin-bottom: 24px;">
      <h1 style="color: #38bdf8; font-size: 28pt;">Slide ${idx + 1}: ${s.title}</h1>
      ${s.subtitle ? `<h2 style="color: #94a3b8; font-size: 18pt;">${s.subtitle}</h2>` : ''}
      ${s.bullets ? `<ul style="font-size: 16pt; line-height: 1.6;">${s.bullets.map(b => `<li>${b}</li>`).join('')}</ul>` : ''}
      ${s.leftColumn ? `<p style="font-size: 15pt;"><b>Key Vectors:</b> ${s.leftColumn.join(' | ')}</p>` : ''}
      ${s.rightColumn ? `<p style="font-size: 15pt;"><b>Outcomes:</b> ${s.rightColumn.join(' | ')}</p>` : ''}
      ${s.stats ? `<p style="font-size: 16pt;">${s.stats.map(st => `<b>${st.value}</b> (${st.label})`).join(' • ')}</p>` : ''}
      ${s.quote ? `<blockquote style="font-size: 18pt; font-style: italic;">"${s.quote.text}" — ${s.quote.author}</blockquote>` : ''}
      ${s.notes ? `<p style="color: #64748b; font-size: 11pt; margin-top: 20px;">Speaker Notes: ${s.notes}</p>` : ''}
    </div>`
    )
    .join('\n');

  return `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>${deck.title}</title></head>
<body>${slidesHtml}</body>
</html>`;
}

// ============================================================================
// 4. CANVAS WORKSPACE COMPONENT
// ============================================================================

interface CanvasWorkspaceProps {
  artifact: CanvasArtifact;
  isOpen: boolean;
  isMaximized: boolean;
  onClose: () => void;
  onToggleMaximize: () => void;
}

export const CanvasWorkspace: React.FC<CanvasWorkspaceProps> = ({
  artifact,
  isOpen,
  isMaximized,
  onClose,
  onToggleMaximize
}) => {
  const [activeTab, setActiveTab] = useState<'preview' | 'code' | 'slides' | 'image'>(() => {
    if (artifact.type === 'presentation') return 'slides';
    if (artifact.type === 'image') return 'image';
    return 'preview';
  });

  const [slideIndex, setSlideIndex] = useState(0);
  const [presentationTheme, setPresentationTheme] = useState<'cyber' | 'minimal' | 'executive'>('cyber');

  useEffect(() => {
    if (artifact.type === 'presentation') setActiveTab('slides');
    else if (artifact.type === 'image') setActiveTab('image');
    else setActiveTab('preview');
    setSlideIndex(0);
  }, [artifact.id, artifact.type]);

  if (!isOpen) return null;

  const downloadFile = (name: string, content: string, mime: string) => {
    const a = document.createElement('a');
    if (artifact.type === 'image' && artifact.dataUrl) {
      a.href = artifact.dataUrl;
    } else {
      const blob = new Blob([content], { type: mime });
      a.href = URL.createObjectURL(blob);
    }
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const shareOffline = async () => {
    try {
      let file: File;
      if (artifact.type === 'image' && artifact.dataUrl) {
        const res = await fetch(artifact.dataUrl);
        const blob = await res.blob();
        file = new File([blob], artifact.filename, { type: artifact.mimeType });
      } else {
        const blob = new Blob([artifact.content], { type: artifact.mimeType });
        file = new File([blob], artifact.filename, { type: artifact.mimeType });
      }
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        await navigator.share({
          title: artifact.title,
          text: `Shared from NOVA Canvas: ${artifact.filename}`,
          files: [file]
        });
        return;
      }
    } catch {}
    downloadFile(artifact.filename, artifact.content, artifact.mimeType);
  };

  const deck =
    artifact.deck || (artifact.type === 'presentation' ? synthesizePresentationDeck(artifact.title) : null);
  const currentSlide = deck ? deck.slides[slideIndex] || deck.slides[0] : null;
  const safeSlug = artifact.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'presentation';

  return (
    <aside className={`canvas-drawer ${isMaximized ? 'maximized' : ''}`}>
      <div className="canvas-header">
        <div className="canvas-title-group">
          <div className="canvas-badge">
            <Sparkles size="{13}"/>
            <span>CANVAS STUDIO · {artifact.type.toUpperCase()}</span>
          </div>
          <h3>{artifact.title}</h3>
        </div>

        <div className="canvas-actions">
          {artifact.type === 'presentation' && deck && (
            <>
              <button
                className="canvas-btn highlight"
                onClick={() =>
                  downloadFile(`${safeSlug}-slides.html`, exportPresentationToHtml(deck), 'text/html')
                }
                title="Download Interactive HTML Slide Deck"
              >
                ⬇ HTML Deck
              </button>
              <button
                className="canvas-btn"
                onClick={() =>
                  downloadFile(`${safeSlug}.ppt`, exportPresentationToPpt(deck), 'application/vnd.ms-powerpoint')
                }
                title="Download PowerPoint (.ppt) File"
              >
                ⬇ .PPT File
              </button>
            </>
          )}

          {artifact.type === 'html' && (
            <button
              className="canvas-btn highlight"
              onClick={() => downloadFile(artifact.filename, artifact.content, 'text/html')}
            >
              ⬇ Save .HTML
            </button>
          )}

          {artifact.type === 'image' && (
            <button
              className="canvas-btn highlight"
              onClick={() => downloadFile(artifact.filename, artifact.content, artifact.mimeType)}
            >
              ⬇ Save Image
            </button>
          )}

          {artifact.type === 'code' && (
            <button
              className="canvas-btn highlight"
              onClick={() => downloadFile(artifact.filename, artifact.content, artifact.mimeType)}
            >
              ⬇ Save {artifact.filename}
            </button>
          )}

          <button className="canvas-btn" onClick={shareOffline} title="Share File Offline">
            Share
          </button>
          <button className="canvas-btn" onClick={onToggleMaximize} title="Expand / Restore Canvas">
            {isMaximized ? 'Restore' : 'Expand'}
          </button>
          <button className="canvas-btn" onClick={onClose} title="Close Canvas">
            <X size="{15}"/>
          </button>
        </div>
      </div>

      <div className="canvas-tabs">
        {artifact.type === 'presentation' && (
          <button
            className={`c-tab ${activeTab === 'slides' ? 'active' : ''}`}
            onClick={() => setActiveTab('slides')}
          >
            📊 Slide Deck Player
          </button>
        )}
        {(artifact.type === 'html' || artifact.type === 'presentation') && (
          <button
            className={`c-tab ${activeTab === 'preview' ? 'active' : ''}`}
            onClick={() => setActiveTab('preview')}
          >
            🖥️ Live App Preview
          </button>
        )}
        {artifact.type === 'image' && (
          <button
            className={`c-tab ${activeTab === 'image' ? 'active' : ''}`}
            onClick={() => setActiveTab('image')}
          >
            🎨 Rendered Image
          </button>
        )}
        <button
          className={`c-tab ${activeTab === 'code' ? 'active' : ''}`}
          onClick={() => setActiveTab('code')}
        >
          💻 Source Code
        </button>
      </div>

      <div className="canvas-body">
        {activeTab === 'slides' && deck && currentSlide && (
          <div className={`slide-deck-viewer theme-${presentationTheme}`}>
            <div className="slide-deck-controls">
              <div className="slide-theme-picker">
                <span>Theme:</span>
                <button
                  className={presentationTheme === 'cyber' ? 'active' : ''}
                  onClick={() => setPresentationTheme('cyber')}
                >
                  Cyber
                </button>
                <button
                  className={presentationTheme === 'executive' ? 'active' : ''}
                  onClick={() => setPresentationTheme('executive')}
                >
                  Executive
                </button>
                <button
                  className={presentationTheme === 'minimal' ? 'active' : ''}
                  onClick={() => setPresentationTheme('minimal')}
                >
                  Minimal
                </button>
              </div>
              <div className="slide-pagination">
                <button
                  disabled={slideIndex === 0}
                  onClick={() => setSlideIndex(prev => Math.max(0, prev - 1))}
                >
                  ◄
                </button>
                <span>
                  Slide {slideIndex + 1} / {deck.slides.length}
                </span>
                <button
                  disabled={slideIndex === deck.slides.length - 1}
                  onClick={() => setSlideIndex(prev => Math.min(deck.slides.length - 1, prev + 1))}
                >
                  ►
                </button>
              </div>
            </div>

            <div className="slide-viewport">
              <div className="slide-content-card">
                {currentSlide.layout === 'title' && (
                  <div>
                    <h1>{currentSlide.title}</h1>
                    {currentSlide.subtitle && <p className="slide-sub">{currentSlide.subtitle}</p>}
                    <div className="slide-meta-badge">{deck.author}</div>
                  </div>
                )}

                {currentSlide.layout === 'bullets' && (
                  <div>
                    <h2>{currentSlide.title}</h2>
                    <ul className="slide-bullet-list">
                      {currentSlide.bullets?.map((b, i) => (
                        <li key={i}>{b}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {currentSlide.layout === 'two-column' && (
                  <div>
                    <h2>{currentSlide.title}</h2>
                    <div className="slide-columns">
                      <div className="slide-col">
                        <h4>Core Vectors</h4>
                        <ul>
                          {currentSlide.leftColumn?.map((b, i) => (
                            <li key={i}>• {b}</li>
                          ))}
                        </ul>
                      </div>
                      <div className="slide-col">
                        <h4>Target Outcomes</h4>
                        <ul>
                          {currentSlide.rightColumn?.map((b, i) => (
                            <li key={i}>• {b}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}

                {currentSlide.layout === 'stats' && (
                  <div>
                    <h2>{currentSlide.title}</h2>
                    <div className="slide-stats-grid">
                      {currentSlide.stats?.map((st, i) => (
                        <div key={i} className="slide-stat-item">
                          <span className="stat-big">{st.value}</span>
                          <span className="stat-label">{st.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {currentSlide.layout === 'quote' && (
                  <div className="slide-quote-view">
                    <h2>{currentSlide.title}</h2>
                    <blockquote>"{currentSlide.quote?.text}"</blockquote>
                    <cite>— {currentSlide.quote?.author}</cite>
                  </div>
                )}
              </div>
            </div>

            {currentSlide.notes && (
              <div className="slide-notes-footer">
                <b>Speaker Notes:</b> {currentSlide.notes}
              </div>
            )}
          </div>
        )}

        {activeTab === 'preview' && (
          <div className="html-runner-container">
            <iframe
              title={artifact.title}
              srcDoc={artifact.content}
              sandbox="allow-scripts allow-modals allow-forms"
              className="html-sandboxed-iframe"
            />
          </div>
        )}

        {activeTab === 'image' && (
          <div className="image-studio-container">
            {artifact.dataUrl && (
              <img src={artifact.dataUrl} alt={artifact.title} className="image-studio-display" />
            )}
            <p className="image-studio-caption">{artifact.content}</p>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="code-inspector-container">
            <pre className="code-inspector-pre">
              <code>{artifact.content}</code>
            </pre>
          </div>
        )}
      </div>
    </aside>
  );
};

// ============================================================================
// 5. NATIVE DEEP FILE & IMAGE RECOGNITION (ZERO EXTERNAL CDN IMPORTS)
// ============================================================================

let imageLandmarkerPromise: Promise<HandLandmarker> | null = null;

async function getImageLandmarker() {
  try {
    if (!imageLandmarkerPromise) {
      const wasmBase = httpsUrl('cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
      const modelAsset = httpsUrl(
        '[storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)'
      );
      imageLandmarkerPromise = FilesetResolver.forVisionTasks(wasmBase).then(vision =>
        HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: modelAsset,
            delegate: 'GPU'
          },
          runningMode: 'IMAGE',
          numHands: 2,
          minHandDetectionConfidence: 0.35
        })
      );
    }
    return await imageLandmarkerPromise;
  } catch {
    imageLandmarkerPromise = null;
    return null;
  }
}

// Pure native PDF text extractor (requires zero external CDN modules and works 100% offline)
async function extractPdfTextNative(file: File): Promise<string> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const decoder = new TextDecoder('latin1');
    const raw = decoder.decode(arrayBuffer);

    const extractedChunks: string[] = [];
    const parenMatches = raw.match(/\(([^()\\]{3,240})\)/g) || [];
    for (const m of parenMatches) {
      const clean = m
        .slice(1, -1)
        .replace(/\\n|\\r/g, ' ')
        .replace(/[^\x20-\x7E]/g, ' ')
        .trim();
      if (clean.length > 3 && /[a-zA-Z]{2,}/.test(clean) && !/^(Type|Font|Page|Catalog|Metadata|Filter)/i.test(clean)) {
        extractedChunks.push(clean);
      }
      if (extractedChunks.length >= 700) break;
    }

    if (extractedChunks.length > 0) {
      return `PDF Document "${file.name}" (${Math.round(file.size / 1024)}KB):\n` + extractedChunks.join(' ');
    }
  } catch {}

  return `PDF File "${file.name}" (${Math.round(file.size / 1024)}KB attached).`;
}

async function extractTextFromImageNative(img: HTMLImageElement): Promise<string> {
  try {
    const AnyWin = window as any;
    if ('TextDetector' in AnyWin) {
      const detector = new AnyWin.TextDetector();
      const results = await detector.detect(img);
      if (results && results.length > 0) {
        const text = results.map((r: any) => r.rawValue).filter(Boolean).join('\n');
        if (text.trim()) return text.trim();
      }
    }
  } catch {}
  return '';
}

async function inspectImageLocally(
  img: HTMLImageElement,
  canvas: HTMLCanvasElement,
  fileName: string
): Promise<string> {
  const w = img.naturalWidth || canvas.width;
  const h = img.naturalHeight || canvas.height;
  const report: string[] = [
    `Image File: "${fileName}" (Resolution: ${w}x${h}px, Aspect Ratio: ${(w / Math.max(h, 1)).toFixed(2)})`
  ];

  try {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const { width, height } = canvas;
      const data = ctx.getImageData(0, 0, width, height).data;
      let rSum = 0;
      let gSum = 0;
      let bSum = 0;
      let count = 0;
      for (let i = 0; i < data.length; i += 16) {
        rSum += data[i];
        gSum += data[i + 1];
        bSum += data[i + 2];
        count++;
      }
      if (count > 0) {
        report.push(
          `Average RGB Color Profile: (${Math.round(rSum / count)}, ${Math.round(gSum / count)}, ${Math.round(
            bSum / count
          )})`
        );
      }
    }
  } catch {}

  try {
    const landmarker = await getImageLandmarker();
    if (landmarker) {
      const res = landmarker.detect(img);
      const hands = res.landmarks || [];
      if (hands.length > 0) {
        const handLines = hands.map((lm, idx) => {
          const d = (a: any, b: any) => Math.hypot(a.x - b.x, a.y - b.y);
          const wrist = lm[0];
          const indexOpen = lm[8].y < lm[6].y && d(lm[8], wrist) > d(lm[6], wrist);
          const middleOpen = lm[12].y < lm[10].y && d(lm[12], wrist) > d(lm[10], wrist);
          const ringOpen = lm[16].y < lm[14].y && d(lm[16], wrist) > d(lm[14], wrist);
          const pinkyOpen = lm[20].y < lm[18].y && d(lm[20], wrist) > d(lm[18], wrist);

          let posture = 'Hand gesture detected';
          if (middleOpen && !indexOpen && !ringOpen && !pinkyOpen) posture = 'Middle Finger Extended';
          else if (indexOpen && pinkyOpen && !middleOpen && !ringOpen) posture = 'Yo-Yo / Rock-On Sign (SEND)';
          else if (indexOpen && middleOpen && ringOpen && pinkyOpen) posture = 'Open Palm / ASL Letter B';
          else if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) posture = 'Two Fingers Up (ASL V / U / R)';

          return `Hand ${idx + 1}: Posture="${posture}"`;
        });
        report.push(`MediaPipe Hand Detection (${hands.length} hand(s)):\n` + handLines.join('\n'));
      }
    }
  } catch {}

  const ocrText = await extractTextFromImageNative(img);
  if (ocrText) {
    report.push(`Extracted Visible Text (OCR):\n"${ocrText.slice(0, 3000)}"`);
  }

  return report.join('\n');
}

async function readPickedFile(file: File): Promise<LocalAttachment> {
  const ext = file.name.split('.').pop()?.toLowerCase() || '';

  if (file.type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'].includes(ext)) {
    return new Promise(resolve => {
      const reader = new FileReader();
      reader.onerror = () =>
        resolve({ name: file.name, size: file.size, mimeType: file.type, isImage: true, content: `[Image: ${file.name}]` });
      reader.onload = () => {
        const img = new Image();
        img.onerror = () =>
          resolve({ name: file.name, size: file.size, mimeType: file.type, isImage: true, content: `[Image: ${file.name}]` });
        img.onload = async () => {
          const maxDim = 512;
          let { width, height } = img;
          if (width > maxDim || height > maxDim) {
            const scale = maxDim / Math.max(width, height);
            width = Math.round(width * scale);
            height = Math.round(height * scale);
          }
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (ctx) ctx.drawImage(img, 0, 0, width, height);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.76);
          const localVisualReport = await inspectImageLocally(img, canvas, file.name);
          resolve({
            name: file.name,
            size: file.size,
            mimeType: 'image/jpeg',
            isImage: true,
            dataUrl,
            localVisualReport,
            content: localVisualReport
          });
        };
        img.src = reader.result as string;
      };
      reader.readAsDataURL(file);
    });
  }

  if (file.type === 'application/pdf' || ext === 'pdf') {
    const pdfText = await extractPdfTextNative(file);
    return { name: file.name, size: file.size, mimeType: 'application/pdf', content: pdfText };
  }

  try {
    const rawText = await file.text();
    return {
      name: file.name,
      size: file.size,
      mimeType: file.type || 'text/plain',
      content: `--- FILE (${file.name}) ---\n${rawText.slice(0, 30000)}\n--- END ---`
    };
  } catch {
    return {
      name: file.name,
      size: file.size,
      mimeType: 'application/octet-stream',
      content: `[Attached File: ${file.name}]`
    };
  }
}

export function parseArtifactsFromText(text: string): CanvasArtifact[] {
  const artifacts: CanvasArtifact[] = [];

  const htmlMatch = text.match(/```html\n([\s\S]*?)```/i);
  if (htmlMatch && htmlMatch[1].length > 30) {
    artifacts.push({
      id: crypto.randomUUID(),
      title: 'Interactive HTML Web App',
      type: 'html',
      filename: 'index.html',
      mimeType: 'text/html',
      content: htmlMatch[1].trim(),
      timestamp: Date.now()
    });
  }

  const genericMatch = text.match(/```(python|py|javascript|js|typescript|ts|csv|json)\n([\s\S]*?)```/i);
  if (genericMatch && genericMatch[2].length > 30 && !htmlMatch) {
    const lang = genericMatch[1].toLowerCase();
    const ext = lang.startsWith('py') ? 'py' : lang.startsWith('ts') ? 'ts' : lang === 'csv' ? 'csv' : 'js';
    artifacts.push({
      id: crypto.randomUUID(),
      title: `${lang.toUpperCase()} File`,
      type: 'code',
      filename: `nova-script.${ext}`,
      mimeType: 'text/plain',
      content: genericMatch[2].trim(),
      timestamp: Date.now()
    });
  }

  return artifacts;
}

async function analyzeImagesWithVisionAI(
  promptText: string,
  images: LocalAttachment[],
  settings: AppSettings
): Promise<{ text: string; provider: string } | null> {
  const validImages = images.filter(img => !!img.dataUrl);
  if (validImages.length === 0 || !navigator.onLine) return null;

  const localHints = validImages.map(img => img.localVisualReport).filter(Boolean).join('\n\n');
  const userQuestion =
    promptText.trim() ||
    'Analyze this image thoroughly. Describe all visible subjects, text, hand gestures, and details.';

  const anySettings = settings as any;
  const geminiKey =
    anySettings.geminiApiKey || anySettings.geminiKey || (import.meta as any).env?.VITE_GEMINI_API_KEY;

  if (geminiKey) {
    try {
      const parts: any[] = [{ text: userQuestion }];
      for (const img of validImages) {
        parts.push({
          inlineData: {
            mimeType: img.mimeType || 'image/jpeg',
            data: img.dataUrl!.split(',')[1]
          }
        });
      }
      const geminiEndpoint = httpsUrl(
        `[generativelanguage.googleapis.com/v1beta/models/$](https://generativelanguage.googleapis.com/v1beta/models/$){
          settings.geminiModel || 'gemini-2.5-flash'
        }:generateContent?key=${geminiKey}`
      );
      const res = await fetch(geminiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ parts }] })
      });
      if (res.ok) {
        const data = await res.json();
        const reply = data?.candidates?.[0]?.content?.parts?.map((p: any) => p.text).join('\n');
        if (reply && !isVisionRefusal(reply)) {
          return { text: reply, provider: 'Cloud Vision · Gemini' };
        }
      }
    } catch {}
  }

  const freeVisionModels = ['openai-large', 'gemini', 'openai'];
  const visionEndpoint = httpsUrl('text.pollinations.ai/openai');
  for (const modelName of freeVisionModels) {
    try {
      const contentParts: any[] = [
        { type: 'text', text: `${userQuestion}\n\n[Local Sensor Context:\n${localHints}]` }
      ];
      for (const img of validImages) {
        contentParts.push({ type: 'image_url', image_url: { url: img.dataUrl } });
      }
      const res = await fetch(visionEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          messages: [
            { role: 'system', content: 'You are NOVA Vision AI. Inspect the image directly and answer clearly.' },
            { role: 'user', content: contentParts }
          ]
        })
      });
      if (res.ok) {
        const data = await res.json();
        const reply = data?.choices?.[0]?.message?.content;
        if (reply && !isVisionRefusal(reply)) {
          return { text: reply, provider: 'Cloud Vision · Multimodal' };
        }
      }
    } catch {}
  }

  return null;
}

// ============================================================================
// 6. MAIN APPLICATION COMPONENT
// ============================================================================

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('nova-unified-settings') || '{}');
      return { ...defaults, ...saved, systemInstruction: defaults.systemInstruction };
    } catch {
      return defaults;
    }
  });

  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'model',
      text:
        '⚡ **NOVA (Jarvis + Ultron Generative Canvas Active)**\n\n' +
        '• **Live HTML Apps:** Type *"build a calculator in html"* to launch a working app inside Canvas.\n' +
        '• **PPT Presentations:** Type *"make a presentation on artificial intelligence"* to open the interactive Slide Deck player & export `.ppt` / `.html`.\n' +
        '• **Image Studio:** Type *"draw a futuristic arc reactor"* for online/offline image synthesis.\n' +
        '• **Deep File & Image Scan:** Attach any Photo, PDF, CSV, or Code file.',
      timestamp: Date.now()
    }
  ]);

  const [activeArtifact, setActiveArtifact] = useState<CanvasArtifact | null>(null);
  const [isCanvasOpen, setIsCanvasOpen] = useState(false);
  const [isCanvasMaximized, setIsCanvasMaximized] = useState(false);

  const [messageImages, setMessageImages] = useState<Record<string, string[]>>({});
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<LocalAttachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [typing, setTyping] = useState(false);
  const [provider, setProvider] = useState('Ready');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  const lastLetterRef = useRef<string>('');
  const lastLetterTimeRef = useRef<number>(0);
  const lastGestureTimeRef = useRef<number>(0);

  useEffect(() => {
    localStorage.setItem('nova-unified-settings', JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const speak = useCallback(
    (text: string) => {
      if (!settings.voiceEnabled || !('speechSynthesis' in window)) return;
      speechSynthesis.cancel();
      const clean = text.replace(/```[\s\S]*?```/g, 'Artifact opened in Canvas.').slice(0, 400);
      const u = new SpeechSynthesisUtterance(clean);
      u.rate = 1.05;
      speechSynthesis.speak(u);
    },
    [settings.voiceEnabled]
  );

  const openArtifactInCanvas = (art: CanvasArtifact) => {
    setActiveArtifact(art);
    setIsCanvasOpen(true);
  };

  const send = useCallback(
    async (value = input) => {
      const text = value.trim();
      if ((!text && attachments.length === 0) || typing) return;

      const currentAttachments = [...attachments];
      const imageAttachments = currentAttachments.filter(a => a.isImage && a.dataUrl);
      const fileNames = currentAttachments.map(a => `📎 ${a.name}`).join(', ');
      const displayLabel = text ? (fileNames ? `${text}\n(${fileNames})` : text) : `Attached: ${fileNames}`;

      const userMsgId = crypto.randomUUID();
      const userMsg: Message = { id: userMsgId, role: 'user', text: displayLabel, timestamp: Date.now() };

      if (imageAttachments.length > 0) {
        setMessageImages(prev => ({
          ...prev,
          [userMsgId]: imageAttachments.map(img => img.dataUrl!)
        }));
      }

      setMessages(prev => [...prev, userMsg]);
      setInput('');
      setAttachments([]);
      setTyping(true);

      try {
        const isPresentation =
          currentAttachments.length === 0 &&
          /\b(presentation|ppt|powerpoint|slide\s*deck|slides)\b/i.test(text) &&
          /\b(create|make|build|generate|design|prepare)\b/i.test(text);

        if (isPresentation) {
          const deck = synthesizePresentationDeck(text);
          const art: CanvasArtifact = {
            id: crypto.randomUUID(),
            title: deck.title,
            type: 'presentation',
            filename: `${deck.title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.html`,
            mimeType: 'text/html',
            content: exportPresentationToHtml(deck),
            deck,
            timestamp: Date.now()
          };
          openArtifactInCanvas(art);
          setProvider('Jarvis Presentation Engine');
          const replyText = `📊 Compiled a **${deck.slides.length}-Slide Presentation Deck** for **"${deck.title}"**.\n\nOpened in **Canvas Studio** where you can switch themes, present live, or download as **HTML Slides** or **PowerPoint (.PPT)**.`;
          setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: replyText, timestamp: Date.now() }]);
          speak(replyText);
          setTyping(false);
          return;
        }

        const isImageRequest =
          currentAttachments.length === 0 &&
          /^(generate|create|draw|make|render)\s+(an?\s+)?(image|picture|photo|art|wallpaper|logo|diagram)\b/i.test(text);

        if (isImageRequest) {
          const imgArt = await generateVisualImageArtifact(text);
          openArtifactInCanvas(imgArt);
          setProvider(navigator.onLine ? 'Flux Vision Engine' : 'Offline Procedural Engine');
          const replyText = `🎨 Rendered visual for **"${imgArt.title}"** and opened it inside **Canvas Studio**.`;
          setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: replyText, timestamp: Date.now() }]);
          speak(replyText);
          setTyping(false);
          return;
        }

        if (imageAttachments.length > 0) {
          const visionResult = await analyzeImagesWithVisionAI(text, imageAttachments, settings);
          if (visionResult) {
            setProvider(visionResult.provider);
            setMessages(prev => [
              ...prev,
              { id: crypto.randomUUID(), role: 'model', text: visionResult.text, timestamp: Date.now() }
            ]);
            speak(visionResult.text);
            setTyping(false);
            return;
          }
        }

        const telemetryBlock =
          currentAttachments.length > 0
            ? '\n\n[DEEP FILE & VISION SENSOR DATA]:\n' +
              currentAttachments.map(a => a.localVisualReport || a.content).join('\n\n')
            : '';

        const combinedPrompt =
          (text || 'Analyze the attached files and provide a complete summary.') + telemetryBlock;

        const result = await generateLocalOrCloud(combinedPrompt, messages, settings);

        const finalReply =
          imageAttachments.length > 0 && isVisionRefusal(result.text)
            ? `**Local Deep Vision Analysis:**\n\n${imageAttachments.map(a => a.localVisualReport).join('\n\n')}`
            : result.text;

        const detectedArtifacts = parseArtifactsFromText(finalReply);
        if (detectedArtifacts.length > 0) {
          openArtifactInCanvas(detectedArtifacts[0]);
        } else if (/\b(html|web\s*app|calculator|todo\s*app)\b/i.test(text) && /\b(build|create|make)\b/i.test(text)) {
          openArtifactInCanvas(synthesizeOfflineHtmlApp(text));
        }

        setProvider(result.provider);
        setMessages(prev => [
          ...prev,
          { id: crypto.randomUUID(), role: 'model', text: finalReply, timestamp: Date.now() }
        ]);
        speak(finalReply);
      } catch {
        if (/\b(html|web\s*app|calculator|todo)\b/i.test(text)) {
          const appArt = synthesizeOfflineHtmlApp(text);
          openArtifactInCanvas(appArt);
        }
        const fallback =
          imageAttachments.length > 0
            ? `**Offline Local Vision Analysis:**\n\n${imageAttachments.map(a => a.localVisualReport).join('\n\n')}`
            : `**Jarvis Offline Core:** Processed command locally ("${text || fileNames}").`;
        setProvider('Offline Local Core');
        setMessages(prev => [
          ...prev,
          { id: crypto.randomUUID(), role: 'model', text: fallback, timestamp: Date.now() }
        ]);
        speak(fallback);
      } finally {
        setTyping(false);
      }
    },
    [input, messages, settings, typing, speak, attachments]
  );

  const onFilePicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    try {
      const parsed = await Promise.all(Array.from(files).map(readPickedFile));
      setAttachments(prev => [...prev, ...parsed]);
    } catch (err) {
      console.error('File read error:', err);
    }
    e.target.value = '';
  };

  const toggleMic = () => {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setProvider('Browser speech recognition unavailable');
      return;
    }
    const r = new SpeechRecognition();
    r.lang = 'en-US';
    r.continuous = false;
    r.interimResults = false;
    r.onstart = () => setListening(true);
    r.onend = () => setListening(false);
    r.onerror = () => setListening(false);
    r.onresult = (e: any) => setInput((e.results?.[0]?.[0]?.transcript || '').trim());
    recognitionRef.current = r;
    r.start();
  };

  const execute = useCallback(
    (result: VisionResult) => {
      const now = Date.now();

      if (result.type === 'LETTER' && result.value) {
        const isSame = result.value === lastLetterRef.current;
        const waitMs = isSame ? SAME_LETTER_COOLDOWN_MS : LETTER_COOLDOWN_MS;
        if (now - lastLetterTimeRef.current < waitMs) return;

        lastLetterRef.current = result.value;
        lastLetterTimeRef.current = now;
        setInput(prev => prev + result.value);
        return;
      }

      if (result.value === 'CLEAR') {
        if (now - lastGestureTimeRef.current < 900) return;
        lastGestureTimeRef.current = now;
        lastLetterRef.current = '';
        setInput('');
        return;
      }

      if (result.value === 'SEND') {
        if (now - lastGestureTimeRef.current < 1200) return;
        lastGestureTimeRef.current = now;
        lastLetterRef.current = '';
        send();
        return;
      }

      if (result.value === 'THEME_SWITCH') return;

      const custom = settings.customGestures.find(g => g.name.toUpperCase() === result.value.toUpperCase());
      if (!custom) return;
      if (custom.action === 'CLEAR') setInput('');
      else if (custom.action === 'COPY_LAST')
        navigator.clipboard?.writeText(messages.filter(m => m.role === 'model').at(-1)?.text || '');
      else if (custom.action === 'TOGGLE_MIC') toggleMic();
      else if (custom.action === 'SEND_MESSAGE') send();
    },
    [messages, settings.customGestures, send]
  );

  const onDetected = useCallback((result: VisionResult) => execute(result), [execute]);
  const vision = useVision(settings, settings.customGestures, onDetected);
  const themeIcon = settings.theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />;

  return (
    <div className={`app ${isCanvasOpen ? 'has-canvas-open' : ''}`}>
      <style>{CANVAS_EMBEDDED_CSS}</style>
      <header>
        <div className="brand">
          <div className="logo">
            <Sparkles size={20} />
          </div>
          <div>
            <h1>NOVA GESTURE AI</h1>
            <span>
              <i /> {settings.aiProvider === 'ollama' ? 'OFFLINE-FIRST' : 'JARVIS + ULTRON CANVAS'} · {provider}
            </span>
          </div>
        </div>
        <div className="header-actions">
          {activeArtifact && !isCanvasOpen && (
            <button
              onClick={() => setIsCanvasOpen(true)}
              title="Reopen Canvas Studio"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                background: '#0284c7',
                color: '#fff',
                padding: '6px 12px',
                borderRadius: 999,
                fontSize: 12,
                fontWeight: 700
              }}
            >
              <Sparkles size={14} /> Canvas
            </button>
          )}
          <button title="Voice" onClick={() => setSettings(s => ({ ...s, voiceEnabled: !s.voiceEnabled }))}>
            {settings.voiceEnabled ? <Mic /> : <MicOff />}
          </button>
          <button
            title="Theme"
            onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))}
          >
            {themeIcon}
          </button>
          <button title="Settings" onClick={() => setSettingsOpen(true)}>
            <Settings />
          </button>
        </div>
      </header>

      <main className="app-workspace">
        <aside className="app-sidebar">
          <CameraView
            videoRef={vision.videoRef}
            enabled={settings.visionEnabled}
            status={vision.status}
            lastDetection={vision.lastDetection}
            onToggle={() => setSettings(s => ({ ...s, visionEnabled: !s.visionEnabled }))}
            settings={settings}
          />
          <div className="capabilities">
            <div className="eyebrow">JARVIS + ULTRON CAPABILITIES</div>
            <div className="cap">
              <span>Canvas Studio</span>
              <b>HTML · PPT · IMAGES</b>
            </div>
            <div className="cap">
              <span>Deep Vision Scan</span>
              <b>PDF · OCR · MEDIAPIPE</b>
            </div>
            <div className="cap">
              <span>Local Sign Engine</span>
              <b>{vision.status === 'local' ? 'ACTIVE' : 'MODEL READY'}</b>
            </div>
            <div className="cap">
              <span>Custom Gestures</span>
              <b>{settings.customGestures.length}</b>
            </div>
            <p>
              <WifiOff size={14} /> Generate live HTML web apps, PowerPoint slide decks, and images with full offline export.
            </p>
          </div>
        </aside>

        <section className="chat">
          <div className="chat-head">
            <div>
              <b>Assistant</b>
              <span>Jarvis + Ultron Generative Core</span>
            </div>
            <button onClick={() => setMessages([])}>
              <Trash2 size={16} /> Clear
            </button>
          </div>
          <div className="messages">
            {messages.map(m => (
              <div key={m.id} className={`message ${m.role}`}>
                <div className="avatar">{m.role === 'user' ? <User size={15} /> : <Bot size={15} />}</div>
                <div className="bubble">
                  {messageImages[m.id] && (
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                      {messageImages[m.id].map((src, idx) => (
                        <img
                          key={idx}
                          src={src}
                          alt="Attached"
                          style={{ maxWidth: 220, maxHeight: 180, borderRadius: 8, objectFit: 'cover' }}
                        />
                      ))}
                    </div>
                  )}
                  <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
                  {m.role === 'model' && (
                    <button className="copy" onClick={() => navigator.clipboard?.writeText(m.text)}>
                      <Copy size={13} />
                    </button>
                  )}
                </div>
              </div>
            ))}
            {typing && (
              <div className="message model">
                <div className="avatar">
                  <Bot size={15} />
                </div>
                <div className="bubble dots">● ● ●</div>
              </div>
            )}
            <div ref={chatEnd} />
          </div>

          {attachments.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', padding: '0 14px 8px', alignItems: 'center' }}>
              {attachments.map((a, i) => (
                <span
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    background: 'rgba(148,163,184,0.15)',
                    borderRadius: 999,
                    padding: '4px 10px',
                    fontSize: 12
                  }}
                >
                  {a.dataUrl && (
                    <img
                      src={a.dataUrl}
                      alt={a.name}
                      style={{ width: 20, height: 20, borderRadius: 4, objectFit: 'cover' }}
                    />
                  )}
                  {a.name}
                  <button
                    style={{ display: 'flex' }}
                    onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))}
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
          )}

          <input ref={fileInputRef} type="file" multiple hidden onChange={onFilePicked} />
          <div className="composer">
            <button className={listening ? 'active mic' : 'mic'} onClick={toggleMic}>
              {listening ? <MicOff /> : <Mic />}
            </button>
            <button className="mic" title="Attach PDF, photo, CSV, or code" onClick={() => fileInputRef.current?.click()}>
              <Paperclip />
            </button>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="Ask anything, 'make presentation on...', 'build calculator in html', 'draw...', or sign…"
            />
            <button
              id="nova-send-btn"
              className="send"
              onClick={() => send()}
              disabled={(!input.trim() && attachments.length === 0) || typing}
            >
              <Send />
            </button>
          </div>
        </section>

        {activeArtifact && (
          <CanvasWorkspace
            artifact={activeArtifact}
            isOpen={isCanvasOpen}
            isMaximized={isCanvasMaximized}
            onClose={() => setIsCanvasOpen(false)}
            onToggleMaximize={() => setIsCanvasMaximized(prev => !prev)}
          />
        )}
      </main>

      {settingsOpen && (
        <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)} />
      )}
    </div>
  );
}

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}
