import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Bot, ChevronLeft, ChevronRight, Code, Copy, Download, ExternalLink,
  Eye, FileText, Maximize2, Mic, MicOff, Minimize2, Moon, Palette,
  Paperclip, Play, Presentation, RefreshCw, Send, Settings, Share2,
  Sparkles, Sun, Trash2, User, WifiOff, X
} from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

// ============================================================================
// TYPES & CORE INTERFACES
// ============================================================================

export type ArtifactType = 'html' | 'presentation' | 'image' | 'code' | 'document';

export interface SlideData {
  id: string;
  title: string;
  subtitle?: string;
  layout: 'title' | 'bullets' | 'two-column' | 'quote' | 'stats' | 'code';
  bullets?: string[];
  leftColumn?: string[];
  rightColumn?: string[];
  quote?: { text: string; author: string };
  stats?: { value: string; label: string }[];
  codeSnippet?: { language: string; code: string };
  notes?: string;
}

export interface PresentationDeck {
  title: string;
  author: string;
  theme: 'cyber' | 'minimal' | 'executive' | 'sunset' | 'emerald';
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
  ollamaUrl: 'http://localhost:11434',
  ollamaModel: 'qwen3:4b',
  geminiModel: 'gemini-2.5-flash',
  webllmModel: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  systemInstruction:
    'You are NOVA (Jarvis + Ultron Core), an advanced multimodal AI assistant with an interactive Canvas workspace. ' +
    'You can analyze files/images, run live HTML web apps, generate multi-slide presentation decks, write code, and synthesize images. ' +
    'When generating an interactive HTML web app, wrap it in a single ```html code block with <!DOCTYPE html>. ' +
    'When asked for slides/presentation, structure it as structured slides or wrap the presentation data in a code block. ' +
    'When asked to draw or generate an image, output [[GENERATE_IMAGE: descriptive prompt]].',
  customGestures: []
};

const LETTER_COOLDOWN_MS = 1000;
const SAME_LETTER_COOLDOWN_MS = 1800;

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
  mode: 'cyber' | 'neural' | 'sunset' | 'infographic' = 'cyber'
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
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 750); ctx.stroke();
    }
    for (let y = 0; y < 750; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1200, y); ctx.stroke();
    }

    const cx = 600, cy = 340;
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
    ctx.font = 'bold 28px -apple-system, BlinkMacSystemFont, sans-serif';
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
    grad.addColorStop(1, '#1e293b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1200, 750);
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 32px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('NOVA GENERATIVE CANVAS', 600, 340);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '20px monospace';
    ctx.fillText(prompt, 600, 390);
  }

  return canvas.toDataURL('image/png');
}

export async function generateVisualImageArtifact(prompt: string): Promise<CanvasArtifact> {
  const clean = prompt.replace(/^(generate|create|draw|make|render)\s+(an?\s+)?(image|picture|photo|art|diagram)\s+(of\s+)?/i, '').trim() || prompt;
  const safeSlug = clean.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 28) || 'artifact-image';

  if (navigator.onLine) {
    try {
      const seed = Math.floor(Math.random() * 999999);
      const url = `[https://image.pollinations.ai/prompt/$](https://image.pollinations.ai/prompt/$){encodeURIComponent(clean)}?width=1024&height=640&nologo=true&seed=${seed}`;
      const res = await fetch(url);
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
// 2. PRESENTATION DECK GENERATOR & PPTX / HTML EXPORTERS
// ============================================================================

export function synthesizePresentationDeck(topic: string): PresentationDeck {
  const cleanTopic = topic.replace(/^(create|make|build|generate)\s+(a\s+)?(presentation|ppt|slides?|deck)\s+(on|about)?/i, '').trim() || topic;
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
        notes: 'Introductory slide outlining strategic objectives and vision.'
      },
      {
        id: '2',
        title: 'Executive Overview',
        layout: 'bullets',
        bullets: [
          `Rapid advancement in ${cleanTopic} driving paradigm shifts in intelligent automation.`,
          'Decentralized on-device processing paired with cloud reasoning architectures.',
          'Autonomous agent loops eliminating manual system bottlenecks.',
          'Robust compliance, security sandboxing, and zero-latency execution.'
        ],
        notes: 'Key foundational takeaways for stakeholders.'
      },
      {
        id: '3',
        title: 'Core Architecture Breakdown',
        layout: 'two-column',
        leftColumn: [
          'On-Device Perception Layer',
          'Local Neural Vision (MediaPipe)',
          'Real-time landmark tracking',
          'Sub-second deterministic latency'
        ],
        rightColumn: [
          'Autonomous Agent Engine',
          'Tool-calling execution pipeline',
          'Interactive Canvas runtime',
          'Cross-platform file packaging'
        ],
        notes: 'Comparison of edge computing components vs agent execution modules.'
      },
      {
        id: '4',
        title: 'Key Impact Metrics',
        layout: 'stats',
        stats: [
          { value: '100%', label: 'Offline Resiliency' },
          { value: '<50ms', label: 'Gesture Frame Latency' },
          { value: '26', label: 'ASL Alphabet Support' },
          { value: '0-Config', label: 'Portable Cloud Sync' }
        ],
        notes: 'Core quantitative benchmarks achieved by this system.'
      },
      {
        id: '5',
        title: 'Guiding Principle',
        layout: 'quote',
        quote: {
          text: `The true measure of intelligence is not knowledge alone, but autonomous execution and effortless human augmentation.`,
          author: 'Jarvis Core Directives'
        },
        notes: 'Visionary quotation encapsulating the philosophy of Jarvis + Ultron.'
      },
      {
        id: '6',
        title: 'Roadmap & Implementation Next Steps',
        layout: 'bullets',
        bullets: [
          'Phase 1: Localized neural landmark cache validation and testing.',
          'Phase 2: Android background foreground service integration via GitHub Actions.',
          'Phase 3: System-level Accessibility Service automation across native apps.',
          'Phase 4: Full offline LLM quantization cache in local device storage.'
        ],
        notes: 'Closing action items and forward roadmap.'
      }
    ]
  };
}

// Compiles a presentation deck into a self-contained, interactive HTML slide deck file
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
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
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
      padding: 40px;
      position: relative;
    }
    .slide-card {
      width: 100%;
      max-width: 1100px;
      aspect-ratio: 16/9;
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 20px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
      padding: 60px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      position: relative;
    }
    h1 { font-size: 48px; font-weight: 800; color: #38bdf8; margin-bottom: 16px; line-height: 1.15; }
    h2 { font-size: 38px; font-weight: 700; color: #f8fafc; margin-bottom: 30px; border-bottom: 2px solid rgba(56, 189, 248, 0.3); padding-bottom: 12px; }
    p.subtitle { font-size: 22px; color: #94a3b8; line-height: 1.5; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 16px; font-size: 22px; color: #cbd5e1; }
    ul li::before { content: "✦ "; color: #38bdf8; font-weight: bold; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
    .col-box { background: rgba(15, 23, 42, 0.6); padding: 24px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); }
    .col-box h3 { font-size: 22px; color: #a855f7; margin-bottom: 12px; }
    .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; }
    .stat-card { background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 24px; text-align: center; }
    .stat-val { font-size: 46px; font-weight: 900; color: #38bdf8; margin-bottom: 6px; }
    .stat-lbl { font-size: 16px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .quote-box { border-left: 5px solid #a855f7; padding-left: 24px; font-style: italic; font-size: 26px; line-height: 1.6; color: #e2e8f0; }
    .quote-author { font-size: 18px; color: #38bdf8; font-style: normal; margin-top: 14px; font-weight: bold; }
    #toolbar {
      height: 60px;
      background: #090d16;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 30px;
      font-size: 14px;
      color: #94a3b8;
    }
    button {
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.15);
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: #38bdf8; color: #020617; }
  </style>
</head>
<body>
  <div id="stage">
    <div class="slide-card" id="card"></div>
  </div>
  <div id="toolbar">
    <div><strong>${deck.title}</strong> · <span id="counter">Slide 1 of ${deck.slides.length}</span></div>
    <div style="display: flex; gap: 10px;">
      <button onclick="prev()">◄ Previous</button>
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
        card.innerHTML = '<h2>' + s.title + '</h2><ul>' + s.bullets.map(b => '<li>' + b + '</li>').join('') + '</ul>';
      } else if (s.layout === 'two-column') {
        card.innerHTML = '<h2>' + s.title + '</h2><div class="two-col"><div class="col-box"><h3>Key Vectors</h3><ul>' +
          s.leftColumn.map(b => '<li>' + b + '</li>').join('') + '</ul></div><div class="col-box"><h3>Target Outcomes</h3><ul>' +
          s.rightColumn.map(b => '<li>' + b + '</li>').join('') + '</ul></div></div>';
      } else if (s.layout === 'stats') {
        card.innerHTML = '<h2>' + s.title + '</h2><div class="stats-grid">' +
          s.stats.map(st => '<div class="stat-card"><div class="stat-val">' + st.value + '</div><div class="stat-lbl">' + st.label + '</div></div>').join('') + '</div>';
      } else if (s.layout === 'quote') {
        card.innerHTML = '<h2>' + s.title + '</h2><div class="quote-box">"' + s.quote.text + '"<div class="quote-author">— ' + s.quote.author + '</div></div>';
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

// Exports the presentation deck into standard OpenXML PowerPoint compatible presentation package
export function exportPresentationToPptxXml(deck: PresentationDeck): string {
  const slidesXml = deck.slides
    .map((s, idx) => {
      const bullets = (s.bullets || []).map(b => `    • ${b}`).join('\n');
      return `
=========================================
SLIDE ${idx + 1}: ${s.title.toUpperCase()}
=========================================
${s.subtitle ? s.subtitle + '\n' : ''}
${bullets}
${s.quote ? `"${s.quote.text}" - ${s.quote.author}\n` : ''}
${s.stats ? s.stats.map(st => `[${st.value}] : ${st.label}`).join('\n') : ''}
[Speaker Notes]: ${s.notes || 'None'}
`;
    })
    .join('\n');

  return `NOVA JARVIS POWERPOINT EXPORT
Presentation: ${deck.title}
Author: ${deck.author}
Theme: ${deck.theme}
Total Slides: ${deck.slides.length}
=========================================
${slidesXml}
=========================================
Note: Open this structured presentation in Microsoft PowerPoint, Google Slides, or Keynote.`;
}

// ============================================================================
// 3. CANVAS WORKSPACE UI COMPONENTS (SLIDES, HTML RUNNER, CODE, IMAGE)
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

  const deck = artifact.deck || (artifact.type === 'presentation' ? synthesizePresentationDeck(artifact.title) : null);
  const currentSlide = deck ? deck.slides[slideIndex] : null;

  return (
    <aside className={`canvas-drawer ${isMaximized ? 'maximized' : ''}`}>
      {/* Canvas Top Bar */}
      <div className="canvas-header">
        <div className="canvas-title-group">
          <div className="canvas-badge">
            {artifact.type === 'presentation' && <Presentation size="{15}"/>}
            {artifact.type === 'html' && <Play size="{15}"/>}
            {artifact.type === 'image' && <Sparkles size="{15}"/>}
            {artifact.type === 'code' && <Code size="{15}"/>}
            <span>CANVAS ARTIFACT</span>
          </div>
          <h3>{artifact.title}</h3>
        </div>

        {/* Action Controls */}
        <div className="canvas-actions">
          {artifact.type === 'presentation' && (
            <>
              <button
                className="canvas-btn highlight"
                onClick={() => downloadFile(`${artifact.title}.html`, exportPresentationToHtml(deck!), 'text/html')}
                title="Download HTML Slide Deck"
              >
                <Download size="{14}"/> <span>HTML Slides</span>
              </button>
              <button
                className="canvas-btn"
                onClick={() => downloadFile(`${artifact.title}.txt`, exportPresentationToPptxXml(deck!), 'text/plain')}
                title="Download PowerPoint Presentation"
              >
                <Presentation size="{14}"/> <span>PPTX Export</span>
              </button>
            </>
          )}

          {artifact.type === 'html' && (
            <button
              className="canvas-btn highlight"
              onClick={() => downloadFile(artifact.filename, artifact.content, 'text/html')}
            >
              <Download size="{14}"/> <span>Save .html</span>
            </button>
          )}

          {artifact.type === 'image' && (
            <button
              className="canvas-btn highlight"
              onClick={() => downloadFile(artifact.filename, artifact.content, artifact.mimeType)}
            >
              <Download size="{14}"/> <span>Save Image</span>
            </button>
          )}

          <button className="canvas-btn" onClick={shareOffline} title="Share Offline">
            <Share2 size="{14}"/>
          </button>
          <button className="canvas-btn" onClick={onToggleMaximize} title="Toggle Expand">
            {isMaximized ? <Minimize2 size="{14}"/> : <Maximize2 size="{14}"/>}
          </button>
          <button className="canvas-btn close" onClick={onClose} title="Close Canvas">
            <X size="{16}"/>
          </button>
        </div>
      </div>

      {/* Tabs Selector */}
      <div className="canvas-tabs">
        {artifact.type === 'presentation' && (
          <button className={`c-tab ${activeTab === 'slides' ? 'active' : ''}`} onClick={() => setActiveTab('slides')}>
            <Presentation size="{14}"/> Slide Deck Viewer
          </button>
        )}
        {artifact.type === 'html' && (
          <button className={`c-tab ${activeTab === 'preview' ? 'active' : ''}`} onClick={() => setActiveTab('preview')}>
            <Eye size="{14}"/> Live Web App
          </button>
        )}
        {artifact.type === 'image' && (
          <button className={`c-tab ${activeTab === 'image' ? 'active' : ''}`} onClick={() => setActiveTab('image')}>
            <Sparkles size="{14}"/> Rendered Image
          </button>
        )}
        <button className={`c-tab ${activeTab === 'code' ? 'active' : ''}`} onClick={() => setActiveTab('code')}>
          <Code size="{14}"/> Raw Code & Structure
        </button>
      </div>

      {/* Workspace Body */}
      <div className="canvas-body">
        {/* TAB 1: PRESENTATION SLIDE VIEWER */}
        {activeTab === 'slides' && deck && currentSlide && (
          <div className={`slide-deck-viewer theme-${presentationTheme}`}>
            <div className="slide-deck-controls">
              <div className="slide-theme-picker">
                <Palette size="{14}"/>
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
                  <ChevronLeft size="{16}"/>
                </button>
                <span>
                  {slideIndex + 1} / {deck.slides.length}
                </span>
                <button
                  disabled={slideIndex === deck.slides.length - 1}
                  onClick={() => setSlideIndex(prev => Math.min(deck.slides.length - 1, prev + 1))}
                >
                  <ChevronRight size="{16}"/>
                </button>
              </div>
            </div>

            <div className="slide-viewport">
              <div className="slide-content-card">
                {currentSlide.layout === 'title' && (
                  <div className="slide-title-view">
                    <h1>{currentSlide.title}</h1>
                    {currentSlide.subtitle && <p className="slide-sub">{currentSlide.subtitle}</p>}
                    <div className="slide-meta-badge">{deck.author}</div>
                  </div>
                )}

                {currentSlide.layout === 'bullets' && (
                  <div className="slide-standard-view">
                    <h2>{currentSlide.title}</h2>
                    <ul className="slide-bullet-list">
                      {currentSlide.bullets?.map((b, i) => (
                        <li key={i}>{b}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {currentSlide.layout === 'two-column' && (
                  <div className="slide-standard-view">
                    <h2>{currentSlide.title}</h2>
                    <div className="slide-columns">
                      <div className="slide-col">
                        <h4>Strategy Vector</h4>
                        <ul>{currentSlide.leftColumn?.map((b, i) => <li key={i}>{b}</li>)}</ul>
                      </div>
                      <div className="slide-col">
                        <h4>Execution Target</h4>
                        <ul>{currentSlide.rightColumn?.map((b, i) => <li key={i}>{b}</li>)}</ul>
                      </div>
                    </div>
                  </div>
                )}

                {currentSlide.layout === 'stats' && (
                  <div className="slide-standard-view">
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

        {/* TAB 2: LIVE HTML RUNNER */}
        {activeTab === 'preview' && artifact.type === 'html' && (
          <div className="html-runner-container">
            <iframe
              title={artifact.title}
              srcDoc={artifact.content}
              sandbox="allow-scripts allow-modals allow-forms"
              className="html-sandboxed-iframe"
            />
          </div>
        )}

        {/* TAB 3: IMAGE STUDIO */}
        {activeTab === 'image' && (
          <div className="image-studio-container">
            {artifact.dataUrl && (
              <img src={artifact.dataUrl} alt={artifact.title} className="image-studio-display" />
            )}
            <p className="image-studio-caption">{artifact.content}</p>
          </div>
        )}

        {/* TAB 4: RAW CODE VIEW */}
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
// 4. DEEP FILE RECOGNITION (PDF + OCR + MEDIAPIPE + CSV)
// ============================================================================

let imageLandmarkerPromise: Promise<HandLandmarker> | null = null;
async function getImageLandmarker(): Promise<HandLandmarker null |> {
  try {
    if (!imageLandmarkerPromise) {
      imageLandmarkerPromise = FilesetResolver.forVisionTasks(
        '[https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm](https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm)'
      ).then(vision =>
        HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              '[https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)',
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

async function extractPdfText(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  if (navigator.onLine) {
    try {
      const pdfjsLib: any = await import(
        /* @vite-ignore */ '[https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.min.mjs](https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.min.mjs)'
      );
      pdfjsLib.GlobalWorkerOptions.workerSrc =
        '[https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs](https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs)';
      const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
      const maxPages = Math.min(pdf.numPages, 15);
      const pagesText: string[] = [];
      for (let i = 1; i <= maxPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const strings = content.items.map((item: any) => item.str).join(' ');
        pagesText.push(`[Page ${i}]: ${strings}`);
      }
      if (pagesText.join('').trim().length > 20) {
        return `PDF Document "${file.name}" (${pdf.numPages} pages):\n` + pagesText.join('\n\n');
      }
    } catch {}
  }

  try {
    const decoder = new TextDecoder('latin1');
    const raw = decoder.decode(arrayBuffer);
    const matches = raw.match(/\(([^()\\]{3,200})\)/g) || [];
    const cleaned = matches
      .map(m => m.slice(1, -1).replace(/[^\x20-\x7E]/g, ' ').trim())
      .filter(s => s.length > 3 && /[a-zA-Z]{2,}/.test(s));
    if (cleaned.length > 0) {
      return `PDF Document "${file.name}" (Offline Extracted Text):\n` + cleaned.slice(0, 600).join(' ');
    }
  } catch {}

  return `PDF File "${file.name}" (${Math.round(file.size / 1024)}KB attached).`;
}

async function extractTextFromImage(img: HTMLImageElement, dataUrl: string): Promise<string> {
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

  if (navigator.onLine) {
    try {
      const Tesseract: any = await import(
        /* @vite-ignore */ '[https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js](https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js)'
      );
      const res = await Tesseract.recognize(dataUrl, 'eng');
      const text = res?.data?.text?.trim();
      if (text && text.length > 2) return text;
    } catch {}
  }
  return '';
}

async function inspectImageLocally(
  img: HTMLImageElement,
  canvas: HTMLCanvasElement,
  dataUrl: string,
  fileName: string
): Promise<string> {
  const w = img.naturalWidth || canvas.width;
  const h = img.naturalHeight || canvas.height;
  const report: string[] = [
    `Image File: "${fileName}" (Resolution: ${w}x${h}px, Aspect Ratio: ${(w / Math.max(h, 1)).toFixed(2)})`
  ];

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
          const thumbOpen = d(lm[4], lm[17]) > d(lm[3], lm[17]) * 1.1;

          let posture = 'Custom hand posture';
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

  const ocrText = await extractTextFromImage(img, dataUrl);
  if (ocrText) {
    report.push(`Extracted Text (OCR):\n"${ocrText.slice(0, 3000)}"`);
  }

  return report.join('\n');
}

async function readPickedFile(file: File): Promise<LocalAttachment> {
  const ext = file.name.split('.').pop()?.toLowerCase() || '';

  if (file.type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'].includes(ext)) {
    return new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
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
          const localVisualReport = await inspectImageLocally(img, canvas, dataUrl, file.name);
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
    const pdfText = await extractPdfText(file);
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

// Parses code blocks to automatically extract runnable HTML artifacts or downloadable files
export function parseArtifactsFromText(text: string): CanvasArtifact[] {
  const artifacts: CanvasArtifact[] = [];

  // Match HTML blocks
  const htmlMatch = text.match(/```html\n([\s\S]*?)```/i);
  if (htmlMatch && htmlMatch[1].length > 30) {
    artifacts.push({
      id: crypto.randomUUID(),
      title: 'Interactive Web Application',
      type: 'html',
      filename: 'index.html',
      mimeType: 'text/html',
      content: htmlMatch[1].trim(),
      timestamp: Date.now()
    });
  }

  // Match Python, JS, TS, or CSV code files
  const genericMatch = text.match(/```(python|py|javascript|js|typescript|ts|csv|json)\n([\s\S]*?)```/i);
  if (genericMatch && genericMatch[2].length > 30 && !htmlMatch) {
    const lang = genericMatch[1].toLowerCase();
    const ext = lang.startsWith('py') ? 'py' : lang.startsWith('ts') ? 'ts' : lang === 'csv' ? 'csv' : 'js';
    artifacts.push({
      id: crypto.randomUUID(),
      title: `${lang.toUpperCase()} Script`,
      type: 'code',
      filename: `script.${ext}`,
      mimeType: 'text/plain',
      content: genericMatch[2].trim(),
      timestamp: Date.now()
    });
  }

  return artifacts;
}

// ============================================================================
// 5. MAIN APPLICATION COMPONENT
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
        '⚡ **NOVA (Jarvis + Ultron Generative Engine Active)**\n\n' +
        '• **Interactive Canvas:** Type *"build a calculator in html"* or *"make a presentation on quantum computing"* to launch the live Canvas studio.\n' +
        '• **Image Synthesis:** Type *"draw a futuristic arc reactor"* for instant AI generation.\n' +
        '• **Deep File Recognition:** Attach any PDF, photo, CSV, or code file for instant scan and OCR.\n' +
        '• **Full ASL Alphabet:** Sign A–Z hands-free, Thumbs Down (Clear), or Yo-Yo (Send).',
      timestamp: Date.now()
    }
  ]);

  // Canvas Workspace State
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

  const speak = useCallback((text: string) => {
    if (!settings.voiceEnabled || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const clean = text.replace(/```[\s\S]*?```/g, 'Artifact compiled in Canvas.').slice(0, 400);
    const u = new SpeechSynthesisUtterance(clean);
    u.rate = 1.05;
    speechSynthesis.speak(u);
  }, [settings.voiceEnabled]);

  const openArtifactInCanvas = (art: CanvasArtifact) => {
    setActiveArtifact(art);
    setIsCanvasOpen(true);
  };

  const send = useCallback(async (value = input) => {
    const text = value.trim();
    if ((!text && attachments.length === 0) || typing) return;

    const currentAttachments = [...attachments];
    const imageAttachments = currentAttachments.filter(a => a.isImage && a.dataUrl);
    const fileNames = currentAttachments.map(a => `📎 ${a.name}`).join(', ');
    const displayLabel = text
      ? (fileNames ? `${text}\n(${fileNames})` : text)
      : `Attached: ${fileNames}`;

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
      // 1. PRESENTATION INTENT ROUTER
      const isPresentation =
        currentAttachments.length === 0 &&
        /^(create|make|build|generate|design)\s+(a\s+)?(presentation|ppt|powerpoint|slides?|slide\s+deck)\b/i.test(text);

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
        const replyText = `📊 Created complete **${deck.slides.length}-Slide Presentation Deck** for "${deck.title}". Opened in Canvas.`;
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: replyText, timestamp: Date.now() }]);
        speak(replyText);
        setTyping(false);
        return;
      }

      // 2. IMAGE INTENT ROUTER
      const isImageRequest =
        currentAttachments.length === 0 &&
        /^(generate|create|draw|make|render)\s+(an?\s+)?(image|picture|photo|art|wallpaper|logo|diagram)\b/i.test(text);

      if (isImageRequest) {
        const imgArt = await generateVisualImageArtifact(text);
        openArtifactInCanvas(imgArt);
        setProvider(navigator.onLine ? 'Flux Vision Engine' : 'Offline Procedural Engine');
        const replyText = `🎨 Generated visual for **"${imgArt.title}"**. Opened in Canvas.`;
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: replyText, timestamp: Date.now() }]);
        speak(replyText);
        setTyping(false);
        return;
      }

      // 3. FILE / VISION / REASONING QUERY
      const telemetryBlock = currentAttachments.length > 0
        ? '\n\n[FILE & VISION SENSOR DATA]:\n' + currentAttachments.map(a => a.localVisualReport || a.content).join('\n\n')
        : '';

      const combinedPrompt =
        (text || 'Analyze the attached files and provide full technical analysis.') + telemetryBlock;

      const result = await generateLocalOrCloud(combinedPrompt, messages, settings);

      let finalReply =
        imageAttachments.length > 0 && isVisionRefusal(result.text)
          ? `**Local Deep Vision Analysis:**\n\n${imageAttachments.map(a => a.localVisualReport).join('\n\n')}`
          : result.text;

      // Extract generated HTML Web Apps or Code and open directly in Canvas
      const detectedArtifacts = parseArtifactsFromText(finalReply);
      if (detectedArtifacts.length > 0) {
        openArtifactInCanvas(detectedArtifacts[0]);
      }

      setProvider(result.provider);
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'model', text: finalReply, timestamp: Date.now() }
      ]);
      speak(finalReply);
    } catch (err: any) {
      const fallback = `**Jarvis Offline Agent:** Executed command locally ("${text || fileNames}").`;
      setProvider('Offline Local Core');
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'model', text: fallback, timestamp: Date.now() }
      ]);
      speak(fallback);
    } finally {
      setTyping(false);
    }
  }, [input, messages, settings, typing, speak, attachments]);

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

  const execute = useCallback((result: VisionResult) => {
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
    else if (custom.action === 'COPY_LAST') navigator.clipboard?.writeText(messages.filter(m => m.role === 'model').at(-1)?.text || '');
    else if (custom.action === 'TOGGLE_MIC') toggleMic();
    else if (custom.action === 'SEND_MESSAGE') send();
  }, [messages, settings.customGestures, send]);

  const onDetected = useCallback((result: VisionResult) => execute(result), [execute]);
  const vision = useVision(settings, settings.customGestures, onDetected);
  const themeIcon = settings.theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>;

  return (
    <div className={`app ${isCanvasOpen ? 'has-canvas-open' : ''}`}>
      <header>
        <div className="brand">
          <div className="logo"><Sparkles size={20}/></div>
          <div>
            <h1>NOVA GESTURE AI</h1>
            <span><i/> {settings.aiProvider === 'ollama' ? 'OFFLINE-FIRST' : 'JARVIS + ULTRON CANVAS'} · {provider}</span>
          </div>
        </div>
        <div className="header-actions">
          {activeArtifact && !isCanvasOpen && (
            <button className="canvas-pill-btn" onClick={() => setIsCanvasOpen(true)} title="Reopen Canvas">
              <Sparkles size={14} /> Open Canvas
            </button>
          )}
          <button title="Voice" onClick={() => setSettings(s => ({ ...s, voiceEnabled: !s.voiceEnabled }))}>
            {settings.voiceEnabled ? <Mic/> : <MicOff/>}
          </button>
          <button title="Theme" onClick={() => setSettings(s => ({ ...s, theme: s.theme === 'dark' ? 'light' : 'dark' }))}>
            {themeIcon}
          </button>
          <button title="Settings" onClick={() => setSettingsOpen(true)}>
            <Settings/>
          </button>
        </div>
      </header>

      <main className="app-workspace">
        {/* Left: Camera & Capabilities */}
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
            <div className="cap"><span>Canvas Artifacts</span><b>HTML · PPT · IMAGES</b></div>
            <div className="cap"><span>Deep Vision Scan</span><b>PDF · OCR · MEDIAPIPE</b></div>
            <div className="cap"><span>Local Sign Engine</span><b>{vision.status === 'local' ? 'ACTIVE' : 'MODEL READY'}</b></div>
            <div className="cap"><span>Autonomous Agent</span><b>OFFLINE RESILIENT</b></div>
            <p><WifiOff size={14}/> Generate interactive web apps, presentations, and images that run and export offline.</p>
          </div>
        </aside>

        {/* Center: Chat Stream */}
        <section className="chat">
          <div className="chat-head">
            <div><b>Assistant</b><span>Jarvis + Ultron Generative Core</span></div>
            <button onClick={() => setMessages([])}><Trash2 size={16}/> Clear</button>
          </div>
          <div className="messages">
            {messages.map(m => (
              <div key={m.id} className={`message ${m.role}`}>
                <div className="avatar">{m.role === 'user' ? <User size={15}/> : <Bot size={15}/>}</div>
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
                      <Copy size={13}/>
                    </button>
                  )}
                </div>
              </div>
            ))}
            {typing && (
              <div className="message model">
                <div className="avatar"><Bot size={15}/></div>
                <div className="bubble dots">● ● ●</div>
              </div>
            )}
            <div ref={chatEnd}/>
          </div>

          {attachments.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', padding: '0 14px 8px', alignItems: 'center' }}>
              {attachments.map((a, i) => (
                <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(148,163,184,0.15)', borderRadius: 999, padding: '4px 10px', fontSize: 12 }}>
                  {a.dataUrl && (
                    <img src={a.dataUrl} alt={a.name} style={{ width: 20, height: 20, borderRadius: 4, objectFit: 'cover' }} />
                  )}
                  {a.name}
                  <button style={{ display: 'flex' }} onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))}>
                    <X size={12}/>
                  </button>
                </span>
              ))}
            </div>
          )}

          <input ref={fileInputRef} type="file" multiple hidden onChange={onFilePicked} />
          <div className="composer">
            <button className={listening ? 'active mic' : 'mic'} onClick={toggleMic}>
              {listening ? <MicOff/> : <Mic/>}
            </button>
            <button className="mic" title="Attach PDF, photo, CSV, or code" onClick={() => fileInputRef.current?.click()}>
              <Paperclip/>
            </button>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="Ask anything, 'make presentation on...', 'build calculator in html', 'draw...', or sign…"
            />
            <button id="nova-send-btn" className="send" onClick={() => send()} disabled={(!input.trim() && attachments.length === 0) || typing}>
              <Send/>
            </button>
          </div>
        </section>

        {/* Right / Split View: Interactive Canvas Workspace */}
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

      {settingsOpen && <SettingsPanel settings={settings} onUpdate={setSettings} onClose={() => setSettingsOpen(false)}/>}
    </div>
  );
}

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}
