export interface MessageAttachment {
  name: string;
  type: 'image' | 'text' | 'pdf';
  content: string; // Plain text or Base64 data URL
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  attachments?: MessageAttachment[];
  imageGeneratedUrl?: string;
}

// 1. FREE ONLINE REASONING (DeepSeek-R1, Qwen-2.5, Llama-3.3)
async function callOnlineLLM(messages: ChatMessage[], apiKey: string): Promise<string> {
  // Format messages and inject file attachments into the system context
  const formattedMessages = messages.map((m) => {
    let text = m.content;
    if (m.attachments && m.attachments.length > 0) {
      const docs = m.attachments
        .map((a) => `\n--- File: ${a.name} ---\n${a.content}\n--- End File ---`)
        .join('\n');
      text = `${text}\n\nAttached Document Data:\n${docs}`;
    }
    return { role: m.role, content: text };
  });

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      // Free high-reasoning model (routes to DeepSeek-R1 or Qwen-2.5)
      model: 'deepseek/deepseek-r1:free',
      messages: [
        {
          role: 'system',
          content:
            'You are an autonomous multimodal assistant. Analyze any attached files carefully, show deep step-by-step reasoning when requested, and provide concise, accurate code and answers.',
        },
        ...formattedMessages,
      ],
      temperature: 0.6,
    }),
  });

  if (!response.ok) {
    throw new Error(`Cloud API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || 'No response generated.';
}

// 2. OFFLINE IN-BROWSER REASONING
// When offline, small local tasks and document search run entirely in memory
function callOfflineAgent(prompt: string, attachments?: MessageAttachment[]): string {
  let fileContext = '';
  if (attachments && attachments.length > 0) {
    fileContext = attachments.map((a) => a.content).join(' ');
  }

  const query = prompt.toLowerCase();

  // If user attached a file offline and asks about it
  if (fileContext) {
    const lines = fileContext.split('\n');
    const matched = lines.filter((l) =>
      query.split(' ').some((word) => word.length > 3 && l.toLowerCase().includes(word))
    );
    if (matched.length > 0) {
      return `[Offline Local Search]: Found relevant lines in your attached file:\n\n${matched.slice(0, 5).join('\n')}`;
    }
    return `[Offline Agent]: File received (${attachments![0].name}). Connect online for DeepSeek-R1 to parse deep insights.`;
  }

  return `[Offline Mode Active]: Currently disconnected from the web. I can inspect offline files, execute device gestures, and vocalize speech. Reconnect to access DeepSeek and Qwen reasoning.`;
}

// 3. MASTER ROUTER
export async function askHybridAI(
  messages: ChatMessage[],
  apiKey: string
): Promise<{ text: string; imageUrl?: string }> {
  const lastMessage = messages[messages.length - 1];
  const prompt = lastMessage.content;

  // Check if user is asking to generate an image
  if (/generate an? image of|draw|create a picture of/i.test(prompt)) {
    const cleanPrompt = prompt
      .replace(/generate an? image of|draw|create a picture of/gi, '')
      .trim();
    // Pollinations.ai generates images completely free with no API key
    const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(cleanPrompt)}?width=1024&height=1024&nologo=true`;
    return {
      text: `Here is the image generated for: "${cleanPrompt}"`,
      imageUrl,
    };
  }

  // Branch online vs offline
  if (navigator.onLine && apiKey) {
    try {
      const reply = await callOnlineLLM(messages, apiKey);
      return { text: reply };
    } catch (err) {
      console.warn('Online API failed, falling back to local engine:', err);
    }
  }

  const offlineReply = callOfflineAgent(prompt, lastMessage.attachments);
  return { text: offlineReply };
}
