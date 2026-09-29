import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Bot, Copy, Download, FileText, Mic, MicOff, Moon,
  Paperclip, Send, Settings, Share2, Sparkles, Sun,
  Trash2, User, WifiOff, X
} from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { generateLocalOrCloud } from './services/aiRouter';
import { CameraView } from './components/CameraView';
import { SettingsPanel } from './components/SettingsPanel';
import { useVision } from './hooks/useVision';
import type { AppSettings, Message, VisionResult } from './types';
import './styles.css';

interface LocalAttachment {
  name: string;
  size: number;
  mimeType: string;
  content: string;
  isImage?: boolean;
  dataUrl?: string;
  localVisualReport?: string;
}

interface GeneratedArtifact {
  filename: string;
  mimeType: string;
  content: string;
  isImage?: boolean;
  dataUrl?: string;
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
    'You are NOVA (Jarvis + Ultron Edition), an advanced multimodal AI assistant and autonomous agent. ' +
    'You can analyze images, PDFs, datasets, and code; generate complete downloadable files; create images; and execute actions. ' +
    'When the user asks you to create a file or script, wrap the code in a standard markdown code block with the language tag and include the filename on the first line as a comment (e.g. // filename: script.py). ' +
    'When the user asks to generate or draw an image, include [[GENERATE_IMAGE: detailed visual prompt]] in your response.',
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
// 1. DEEP FILE & IMAGE RECOGNITION ENGINE (PDF + OCR + MEDIAPIPE + CSV/CODE)
// ============================================================================

let imageLandmarkerPromise: Promise<HandLandmarker> | null = null;
async function getImageLandmarker(): Promise<HandLandmarker | null> {
  try {
    if (!imageLandmarkerPromise) {
      imageLandmarkerPromise = FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      ).then(vision =>
        HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
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

// Extracts readable text from PDF files (uses pdf.js when online, falls back to raw stream parser offline)
async function extractPdfText(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();

  // Attempt full PDF.js parsing via dynamic CDN import
  if (navigator.onLine) {
    try {
      const pdfjsLib: any = await import(
        /* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.min.mjs'
      );
      pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs';
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

  // Offline Fallback: Extract text blocks between parentheses in raw PDF stream
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

// Runs OCR on an image using native browser TextDetector or Tesseract.js
async function extractTextFromImage(img: HTMLImageElement, dataUrl: string): Promise<string> {
  // 1. Fast Native Browser OCR (Chrome/Android)
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

  // 2. Universal Tesseract.js OCR fallback
  if (navigator.onLine) {
    try {
      const Tesseract: any = await import(
        /* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js'
      );
      const res = await Tesseract.recognize(dataUrl, 'eng');
      const text = res?.data?.text?.trim();
      if (text && text.length > 2) return text;
    } catch {}
  }

  return '';
}

// Deep Local Image Scanner: Pixel Color/Edge Analysis + MediaPipe Hand Detection + OCR
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

  // 1. Pixel Color, Contrast & Region Analysis
  try {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const { width, height } = canvas;
      const data = ctx.getImageData(0, 0, width, height).data;
      let rSum = 0, gSum = 0, bSum = 0, brightSum = 0, edgeCount = 0, count = 0;
      for (let i = 0; i < data.length; i += 16) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        rSum += r; gSum += g; bSum += b;
        brightSum += lum;
        if (i >= 16) {
          const prevLum = 0.299 * data[i - 16] + 0.587 * data[i - 15] + 0.114 * data[i - 14];
          if (Math.abs(lum - prevLum) > 38) edgeCount++;
        }
        count++;
      }
      if (count > 0) {
        const avgR = Math.round(rSum / count);
        const avgG = Math.round(gSum / count);
        const avgB = Math.round(bSum / count);
        const avgBright = Math.round(brightSum / count);
        const edgeDensity = Math.round((edgeCount / count) * 100);
        const lighting =
          avgBright > 205 ? 'High-key / Bright background' :
          avgBright < 60 ? 'Low-key / Dark background' : 'Balanced lighting';
        const complexity =
          edgeDensity > 22 ? 'High detail / dense text or complex scene' :
          edgeDensity > 8 ? 'Moderate detail / clear subject' : 'Minimalist / smooth background';
        report.push(
          `Scene Telemetry: ${lighting}, ${complexity} (Avg RGB: ${avgR},${avgG},${avgB}, Edge Density: ${edgeDensity}%)`
        );
      }
    }
  } catch {}

  // 2. Local MediaPipe Hand & Gesture Recognition
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

          const openFingers: string[] = [];
          if (thumbOpen) openFingers.push('Thumb');
          if (indexOpen) openFingers.push('Index');
          if (middleOpen) openFingers.push('Middle');
          if (ringOpen) openFingers.push('Ring');
          if (pinkyOpen) openFingers.push('Pinky');

          let posture = 'Custom hand posture';
          if (middleOpen && !indexOpen && !ringOpen && !pinkyOpen) posture = 'Middle Finger Extended';
          else if (indexOpen && pinkyOpen && !middleOpen && !ringOpen) posture = 'Yo-Yo / Rock-On Sign (SEND)';
          else if (indexOpen && middleOpen && ringOpen && pinkyOpen) posture = 'Open Palm / ASL Letter B';
          else if (indexOpen && middleOpen && ringOpen && !pinkyOpen) posture = 'Three Fingers Up / ASL Letter W';
          else if (!indexOpen && middleOpen && ringOpen && pinkyOpen) posture = 'OK / ASL Letter F';
          else if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) posture = 'Two Fingers Up (ASL V / U / R)';
          else if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) posture = thumbOpen ? 'ASL Letter L' : 'ASL Letter D / Pointing';
          else if (!indexOpen && !middleOpen && !ringOpen && pinkyOpen) posture = thumbOpen ? 'ASL Letter Y' : 'ASL Letter I';
          else if (!indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            posture = lm[4].y > lm[0].y + 0.05 ? 'Thumbs Down (CLEAR)' : 'Closed Fist (ASL A / S / E / O / T)';
          }

          return `Hand ${idx + 1}: Extended=[${openFingers.join(', ') || 'None'}], Posture="${posture}"`;
        });
        report.push(`MediaPipe Hand Detection (${hands.length} hand(s)):\n` + handLines.join('\n'));
      }
    }
  } catch {}

  // 3. OCR Visible Text Extraction
  const ocrText = await extractTextFromImage(img, dataUrl);
  if (ocrText) {
    report.push(`Extracted Visible Text (OCR):\n"${ocrText.slice(0, 3000)}"`);
  }

  return report.join('\n');
}

// Analyzes CSV files to provide instant row/column summaries
function analyzeCsvContent(name: string, raw: string): string {
  const lines = raw.split(/\r?\n/).filter(l => l.trim().length > 0);
  if (lines.length === 0) return `Empty CSV file: ${name}`;
  const headers = lines[0].split(',').map(h => h.trim());
  const sampleRows = lines.slice(1, 25).join('\n');
  return (
    `--- CSV DATASET: ${name} ---\n` +
    `Total Rows: ${lines.length - 1} | Columns (${headers.length}): ${headers.join(', ')}\n` +
    `Sample Data (First 24 rows):\n${sampleRows}\n` +
    `--- END OF CSV ---`
  );
}

async function readPickedFile(file: File): Promise<LocalAttachment> {
  const ext = file.name.split('.').pop()?.toLowerCase() || '';

  // 1. Image Files
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

  // 2. PDF Documents
  if (file.type === 'application/pdf' || ext === 'pdf') {
    const pdfText = await extractPdfText(file);
    return {
      name: file.name,
      size: file.size,
      mimeType: 'application/pdf',
      content: pdfText
    };
  }

  // 3. CSV Datasets
  if (ext === 'csv') {
    const raw = await file.text();
    return {
      name: file.name,
      size: file.size,
      mimeType: 'text/csv',
      content: analyzeCsvContent(file.name, raw)
    };
  }

  // 4. Code, JSON, Markdown, XML, TXT & General Files
  try {
    const rawText = await file.text();
    const clipped = rawText.slice(0, 30000);
    return {
      name: file.name,
      size: file.size,
      mimeType: file.type || 'text/plain',
      content: `--- FILE (${file.name} | ${Math.round(file.size / 1024)}KB) ---\n${clipped}\n--- END OF FILE ---`
    };
  } catch {
    return {
      name: file.name,
      size: file.size,
      mimeType: file.type || 'application/octet-stream',
      content: `[Binary File Attached: ${file.name} (${Math.round(file.size / 1024)}KB)]`
    };
  }
}

// ============================================================================
// 2. ALL-ROUNDER GENERATION ENGINE (ONLINE/OFFLINE IMAGES + REAL FILES + SHARE)
// ============================================================================

// Renders a high-tech procedural visual offline on an HTML5 Canvas when there is no internet
function generateOfflineCanvasImage(prompt: string): string {
  const canvas = document.createElement('canvas');
  canvas.width = 768;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Deep cyber background gradient
  const grad = ctx.createLinearGradient(0, 0, 768, 512);
  grad.addColorStop(0, '#090d16');
  grad.addColorStop(0.5, '#111827');
  grad.addColorStop(1, '#1e1b4b');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 768, 512);

  // Perspective grid
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.14)';
  ctx.lineWidth = 1;
  for (let x = 0; x < 768; x += 32) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 512); ctx.stroke();
  }
  for (let y = 0; y < 512; y += 32) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(768, y); ctx.stroke();
  }

  // Jarvis Arc Reactor Geometric Rings
  const cx = 384, cy = 230;
  for (let r = 40; r <= 140; r += 25) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = r % 50 === 0 ? 'rgba(56, 189, 248, 0.75)' : 'rgba(168, 85, 247, 0.55)';
    ctx.lineWidth = r === 90 ? 4 : 2;
    ctx.stroke();
  }

  // Glowing core
  const coreGrad = ctx.createRadialGradient(cx, cy, 5, cx, cy, 65);
  coreGrad.addColorStop(0, '#ffffff');
  coreGrad.addColorStop(0.4, '#38bdf8');
  coreGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');
  ctx.fillStyle = coreGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, 65, 0, Math.PI * 2);
  ctx.fill();

  // Prompt caption HUD
  ctx.fillStyle = '#e2e8f0';
  ctx.font = 'bold 20px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('NOVA OFFLINE PROCEDURAL RENDER', cx, 420);
  ctx.fillStyle = '#38bdf8';
  ctx.font = '15px monospace';
  ctx.fillText(prompt.slice(0, 64), cx, 452);

  return canvas.toDataURL('image/png');
}

// Generates an image (uses Pollinations Flux Diffusion when online, or procedural Canvas offline)
async function generateVisualImage(prompt: string): Promise<GeneratedArtifact> {
  const cleanPrompt = prompt.replace(/^(generate|create|draw|make)\s+(an?\s+)?(image|picture|photo|art|diagram)\s+(of\s+)?/i, '').trim() || prompt;
  const safeSlug = cleanPrompt.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 28) || 'nova-image';

  if (navigator.onLine) {
    try {
      const seed = Math.floor(Math.random() * 999999);
      const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(cleanPrompt)}?width=768&height=512&nologo=true&seed=${seed}`;
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
          filename: `${safeSlug}.jpg`,
          mimeType: 'image/jpeg',
          content: cleanPrompt,
          isImage: true,
          dataUrl
        };
      }
    } catch {}
  }

  const offlineDataUrl = generateOfflineCanvasImage(cleanPrompt);
  return {
    filename: `${safeSlug}-offline.png`,
    mimeType: 'image/png',
    content: cleanPrompt,
    isImage: true,
    dataUrl: offlineDataUrl
  };
}

// Extracts code blocks from AI responses and packages them into real downloadable/shareable files
function extractGeneratedFiles(aiText: string): GeneratedArtifact[] {
  const artifacts: GeneratedArtifact[] = [];
  const regex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;
  let index = 1;

  const extMap: Record<string, { ext: string; mime: string }> = {
    python: { ext: 'py', mime: 'text/x-python' },
    py: { ext: 'py', mime: 'text/x-python' },
    javascript: { ext: 'js', mime: 'text/javascript' },
    js: { ext: 'js', mime: 'text/javascript' },
    typescript: { ext: 'ts', mime: 'text/typescript' },
    ts: { ext: 'ts', mime: 'text/typescript' },
    tsx: { ext: 'tsx', mime: 'text/typescript' },
    html: { ext: 'html', mime: 'text/html' },
    css: { ext: 'css', mime: 'text/css' },
    json: { ext: 'json', mime: 'application/json' },
    csv: { ext: 'csv', mime: 'text/csv' },
    markdown: { ext: 'md', mime: 'text/markdown' },
    md: { ext: 'md', mime: 'text/markdown' },
    svg: { ext: 'svg', mime: 'image/svg+xml' },
    xml: { ext: 'xml', mime: 'application/xml' },
    sql: { ext: 'sql', mime: 'application/sql' },
    sh: { ext: 'sh', mime: 'application/x-sh' },
    bash: { ext: 'sh', mime: 'application/x-sh' }
  };

  while ((match = regex.exec(aiText)) !== null) {
    const lang = (match[1] || 'txt').toLowerCase();
    const code = match[2].trim();
    if (code.length < 15) continue;

    // Look for an explicit filename comment on the first line
    const firstLine = code.split('\n')[0];
    const nameMatch = firstLine.match(/(?:filename|file):\s*([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)/i);
    const mapped = extMap[lang] || { ext: 'txt', mime: 'text/plain' };
    const filename = nameMatch ? nameMatch[1] : `nova-output-${index}.${mapped.ext}`;

    artifacts.push({
      filename,
      mimeType: mapped.mime,
      content: code
    });
    index++;
  }

  return artifacts;
}

// Downloads any generated file or image directly to the device
function downloadArtifact(art: GeneratedArtifact) {
  const a = document.createElement('a');
  if (art.isImage && art.dataUrl) {
    a.href = art.dataUrl;
  } else {
    const blob = new Blob([art.content], { type: art.mimeType });
    a.href = URL.createObjectURL(blob);
  }
  a.download = art.filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// Shares any generated file or image using Android's native offline Share Sheet
async function shareArtifactOffline(art: GeneratedArtifact) {
  try {
    let file: File;
    if (art.isImage && art.dataUrl) {
      const res = await fetch(art.dataUrl);
      const blob = await res.blob();
      file = new File([blob], art.filename, { type: art.mimeType });
    } else {
      const blob = new Blob([art.content], { type: art.mimeType });
      file = new File([blob], art.filename, { type: art.mimeType });
    }

    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({
        title: art.filename,
        text: `Shared from NOVA Jarvis AI: ${art.filename}`,
        files: [file]
      });
      return;
    }
  } catch {}
  // Fallback to direct download if native file share sheet is cancelled or unsupported
  downloadArtifact(art);
}

// ============================================================================
// 3. MULTIMODAL CLOUD VISION + LOCAL OFFLINE AGENT BRAIN
// ============================================================================

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
    'Analyze this image thoroughly. Describe all visible subjects, text, hand signs, colors, and technical details.';

  const anySettings = settings as any;
  const geminiKey =
    anySettings.geminiApiKey ||
    anySettings.geminiKey ||
    (import.meta as any).env?.VITE_GEMINI_API_KEY;

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
      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${settings.geminiModel || 'gemini-2.5-flash'}:generateContent?key=${geminiKey}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ contents: [{ parts }] })
        }
      );
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
  for (const modelName of freeVisionModels) {
    try {
      const contentParts: any[] = [
        {
          type: 'text',
          text: `${userQuestion}\n\n[Local Sensor Scan Context:\n${localHints}]`
        }
      ];
      for (const img of validImages) {
        contentParts.push({ type: 'image_url', image_url: { url: img.dataUrl } });
      }
      const res = await fetch('https://text.pollinations.ai/openai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          messages: [
            {
              role: 'system',
              content: 'You are NOVA Multimodal Vision AI. Inspect the image directly and provide a complete, accurate analysis.'
            },
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
        'Hello. I am NOVA (Jarvis + Ultron Core).\n\n' +
        '• Attach any Image, PDF, CSV, or Code file for deep analysis.\n' +
        '• Ask me to generate images ("draw a futuristic arc reactor") or create downloadable files ("create a python calculator script").\n' +
        '• Use ASL hand signs (A–Z), Thumbs Down (Clear), or Yo-Yo (Send).',
      timestamp: Date.now()
    }
  ]);

  const [messageImages, setMessageImages] = useState<Record<string, string[]>>({});
  const [messageArtifacts, setMessageArtifacts] = useState<Record<string, GeneratedArtifact[]>>({});
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
    const clean = text.replace(/```[\s\S]*?
