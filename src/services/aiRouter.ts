import type { Message, AppSettings } from '../types';
import type { MessageAttachment } from './hybridAI';
import { chatWebLLM } from './webllm';

function ollamaUrl(base: string) {
  return `${base.replace(/\/$/, '')}/v1/chat/completions`;
}

function geminiUrl(model: string, key: string) {
  return `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`;
}

function systemPrompt(settings: AppSettings) {
  return `${settings.systemInstruction}\n\nYou are the language/assistant layer of NOVA Gesture AI. Keep responses concise and useful. Do not invent sign-language detections; the vision layer reports gestures separately.`;
}

function attachmentsToPrompt(attachments?: MessageAttachment[]): string {
  if (!attachments?.length) return '';
  const MAX_CHARS = 8000;
  return attachments.map((attachment) => {
    if (attachment.type === 'text') {
      const content = attachment.content.length > MAX_CHARS
        ? `${attachment.content.slice(0, MAX_CHARS)}\n…(truncated)`
        : attachment.content;
      return `\n\n--- Attached file: ${attachment.name} ---\n${content}\n--- End of ${attachment.name} ---`;
    }
    if (attachment.type === 'pdf') {
      return `\n\n[Attached PDF: ${attachment.name}. This provider can only use extracted text when available.]`;
    }
    return `\n\n[Attached image: ${attachment.name}. Use a vision-capable provider to inspect the image.]`;
  }).join('');
}

export async function generateLocalOrCloud(
  text: string,
  history: Message[],
  settings: AppSettings,
  attachments?: MessageAttachment[]
): Promise<{ text: string; provider: string }> {
  const userContent = text + attachmentsToPrompt(attachments);
  const messages = [
    { role: 'system', content: systemPrompt(settings) },
    ...history.slice(-12).map(m => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.text })),
    { role: 'user', content: userContent }
  ];

  const tryOllama = async () => {
    const response = await fetch(ollamaUrl(settings.ollamaUrl), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: settings.ollamaModel, messages, stream: false })
    });
    if (!response.ok) throw new Error(`Local AI HTTP ${response.status}`);
    const data = await response.json();
    return data?.choices?.[0]?.message?.content?.trim() || 'The local model returned no text.';
  };

  const tryCloudFree = async () => {
    const key = settings.cloudFreeApiKey;
    if (!key) throw new Error('No API key set for the free cloud provider. Add one in Settings.');
    const base = (settings.cloudFreeBaseUrl || 'https://api.groq.com/openai/v1').replace(/\/$/, '');
    const response = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body: JSON.stringify({ model: settings.cloudFreeModel || 'llama-3.3-70b-versatile', messages, stream: false })
    });
    if (!response.ok) throw new Error(`Cloud free-tier HTTP ${response.status}: ${await response.text().catch(() => '')}`);
    const data = await response.json();
    return data?.choices?.[0]?.message?.content?.trim() || 'The model returned no text.';
  };

  const tryWebLLM = async () => {
    return chatWebLLM(messages as { role: string; content: string }[], settings.webllmModel || 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC', (text) => {
      window.dispatchEvent(new CustomEvent('nova-webllm-progress', { detail: text }));
    });
  };

  const tryGemini = async () => {
    const key = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;
    if (!key) throw new Error('VITE_GEMINI_API_KEY is not configured');
    const imageParts = (attachments || [])
      .filter((attachment) => attachment.type === 'image')
      .map((attachment) => ({
        inlineData: {
          mimeType: 'image/jpeg',
          data: attachment.content.split(',')[1] || attachment.content,
        },
      }));
    const contents = [
      ...history.slice(-12).map(m => ({ role: m.role === 'model' ? 'model' : 'user', parts: [{ text: m.text }] })),
      { role: 'user', parts: [{ text: userContent }, ...imageParts] }
    ];
    const response = await fetch(geminiUrl(settings.geminiModel, key), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents, systemInstruction: { parts: [{ text: systemPrompt(settings) }] } })
    });
    if (!response.ok) throw new Error(`Gemini HTTP ${response.status}`);
    const data = await response.json();
    return data?.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text || '').join('').trim() || 'Gemini returned no text.';
  };

  if (settings.aiProvider === 'ollama') {
    try { return { text: await tryOllama(), provider: `Local • ${settings.ollamaModel}` }; }
    catch (e) { return { text: `Local AI is not reachable. Start Ollama and make sure ${settings.ollamaModel} is installed.\n\n${String(e)}`, provider: 'Local unavailable' }; }
  }

  if (settings.aiProvider === 'gemini') {
    try { return { text: await tryGemini(), provider: `Cloud • ${settings.geminiModel}` }; }
    catch (e) { return { text: `Cloud AI error: ${String(e)}`, provider: 'Cloud error' }; }
  }

  if (settings.aiProvider === 'cloudFree') {
    try { return { text: await tryCloudFree(), provider: `Cloud (free tier) • ${settings.cloudFreeModel}` }; }
    catch (e) { return { text: `Cloud AI error: ${String(e)}`, provider: 'Cloud error' }; }
  }

  if (settings.aiProvider === 'webllm') {
    try { return { text: await tryWebLLM(), provider: `On-device • ${settings.webllmModel}` }; }
    catch (e) { return { text: `On-device AI error: ${String(e)}`, provider: 'On-device error' }; }
  }

  try { return { text: await tryOllama(), provider: `Local • ${settings.ollamaModel}` }; }
  catch {
    try { return { text: await tryCloudFree(), provider: `Cloud (free tier) • ${settings.cloudFreeModel}` }; }
    catch {
      try { return { text: await tryGemini(), provider: `Cloud • ${settings.geminiModel}` }; }
      catch { return { text: 'No AI provider is reachable. The sign/gesture engine still runs locally either way.', provider: 'Offline' }; }
    }
  }
}

export async function analyzeWithGemini(base64: string, settings: AppSettings, customGestures: AppSettings['customGestures']): Promise<string> {
  const key = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;
  if (!key) throw new Error('VITE_GEMINI_API_KEY is not configured');
  const prompt = `Analyze this mirrored webcam image for a hand gesture. Recognize ASL static letters A, B, C, L, V, Y and these controls: THEME_SWITCH (open palm), CLEAR (thumb down). Custom gestures: ${customGestures.map(g => `${g.name}:${g.description}`).join('; ')}. Return ONLY JSON like {"type":"LETTER|GESTURE|UNKNOWN","value":"A","confidence":0.0}. Do not guess.`;
  const response = await fetch(geminiUrl(settings.geminiModel, key), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: [{ role: 'user', parts: [{ inlineData: { mimeType: 'image/jpeg', data: base64.split(',')[1] || base64 } }, { text: prompt }] }], generationConfig: { responseMimeType: 'application/json' } })
  });
  if (!response.ok) throw new Error(`Gemini vision HTTP ${response.status}`);
  const data = await response.json();
  return data?.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text || '').join('') || '';
}
