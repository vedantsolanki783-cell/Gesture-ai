// Runs a small language model entirely on-device, inside the browser, using
// WebGPU. No server, no internet needed after the model has been downloaded
// once. Requires a browser with WebGPU support (recent Chrome on Android is
// generally fine; older devices or browsers without WebGPU cannot use this).
//
// The library itself is loaded from a CDN at runtime via a plain ESM import
// (not an npm dependency), so no build-tool changes are needed for this file
// to work once it's added to the project.

type ChatMsg = { role: string; content: string };
type ProgressFn = (text: string) => void;

let enginePromise: Promise<any> | null = null;
let loadedModelId: string | null = null;

async function loadWebLLM() {
  // @vite-ignore prevents the bundler from trying to resolve this as a
  // local module — it's a real URL, fetched by the browser at runtime.
  return import(/* @vite-ignore */ 'https://esm.run/@mlc-ai/web-llm');
}

export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!(navigator as any).gpu;
}

async function getEngine(modelId: string, onProgress?: ProgressFn) {
  if (enginePromise && loadedModelId === modelId) return enginePromise;
  loadedModelId = modelId;
  enginePromise = (async () => {
    const webllm = await loadWebLLM();
    return webllm.CreateMLCEngine(modelId, {
      initProgressCallback: (p: { text?: string }) => onProgress?.(p?.text || 'Loading on-device model…')
    });
  })();
  return enginePromise;
}

export async function chatWebLLM(messages: ChatMsg[], modelId: string, onProgress?: ProgressFn): Promise<string> {
  if (!isWebGPUAvailable()) {
    throw new Error('This browser/device does not support WebGPU, so the on-device model cannot run. Try the latest Chrome.');
  }
  const engine = await getEngine(modelId, onProgress);
  const reply = await engine.chat.completions.create({ messages });
  return reply?.choices?.[0]?.message?.content?.trim() || 'The on-device model returned no text.';
}
