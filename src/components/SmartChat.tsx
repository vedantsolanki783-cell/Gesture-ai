import React, { useState, useRef } from 'react';
import { askHybridAI, ChatMessage, MessageAttachment } from '../services/hybridAI';
import { parseUploadedFile } from '../services/fileReader';

interface Props {
  apiKey: string;
  incomingGestureText?: string;
}

export const SmartChat: React.FC<Props> = ({ apiKey, incomingGestureText }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        'Ready! I can think with DeepSeek & Qwen, generate images ("draw a cyber cat"), and read your attached files.',
    },
  ]);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<MessageAttachment[]>([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync text coming from Sign Language / Gesture recognition
  React.useEffect(() => {
    if (incomingGestureText) {
      setInput((prev) => (prev ? `${prev} ${incomingGestureText}` : incomingGestureText));
    }
  }, [incomingGestureText]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    try {
      const parsed = await parseUploadedFile(files[0]);
      setAttachments((prev) => [...prev, parsed]);
    } catch (err) {
      console.error('File read error:', err);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && attachments.length === 0) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      attachments: attachments.length > 0 ? [...attachments] : undefined,
    };

    const newHistory = [...messages, userMessage];
    setMessages(newHistory);
    setInput('');
    setAttachments([]);
    setLoading(true);

    try {
      const result = await askHybridAI(newHistory, apiKey);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: result.text,
          imageGeneratedUrl: result.imageUrl,
        },
      ]);
    } catch (error: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${error.message || 'Failed to process'}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', maxWidth: '650px', margin: '0 auto' }}>
      {/* Chat Messages Log */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {messages.map((m, idx) => (
          <div
            key={idx}
            style={{
              alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
              background: m.role === 'user' ? '#2563eb' : '#1f2937',
              color: '#ffffff',
              padding: '12px 16px',
              borderRadius: '12px',
              maxWidth: '85%',
              wordBreak: 'break-word',
            }}
          >
            {/* Show any attached file tags */}
            {m.attachments?.map((att, i) => (
              <div key={i} style={{ fontSize: '11px', background: 'rgba(255,255,255,0.2)', padding: '2px 6px', borderRadius: '4px', marginBottom: '6px' }}>
                📎 {att.name} ({att.type})
              </div>
            ))}

            {/* Message Body */}
            <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>

            {/* Render Generated Image if prompt requested one */}
            {m.imageGeneratedUrl && (
              <div style={{ marginTop: '10px' }}>
                <img
                  src={m.imageGeneratedUrl}
                  alt="Generated AI"
                  style={{ width: '100%', borderRadius: '8px', border: '1px solid #374151' }}
                  loading="lazy"
                />
              </div>
            )}
          </div>
        ))}
        {loading && <div style={{ color: '#9ca3af', fontStyle: 'italic' }}>Thinking deeply with DeepSeek...</div>}
      </div>

      {/* Uploaded File Badges preview */}
      {attachments.length > 0 && (
        <div style={{ padding: '8px 16px', display: 'flex', gap: '8px', background: '#111827' }}>
          {attachments.map((att, idx) => (
            <span key={idx} style={{ background: '#374151', color: '#60a5fa', padding: '4px 8px', borderRadius: '6px', fontSize: '12px' }}>
              📎 {att.name}
            </span>
          ))}
        </div>
      )}

      {/* Input bar */}
      <div style={{ display: 'flex', gap: '8px', padding: '12px', background: '#111827', borderTop: '1px solid #374151' }}>
        {/* Hidden File Picker */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          style={{ display: 'none' }}
          accept=".txt,.py,.ts,.js,.json,.csv,.md,.png,.jpg,.jpeg"
        />

        {/* Paperclip Button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          style={{ background: '#374151', border: 'none', borderRadius: '8px', padding: '0 14px', color: '#fff', cursor: 'pointer', fontSize: '18px' }}
          title="Upload file or image"
        >
          📎
        </button>

        {/* Text Input */}
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask DeepSeek, attach a file, or type 'draw a space rocket'..."
          style={{ flex: 1, padding: '10px 14px', borderRadius: '8px', background: '#1f2937', color: '#fff', border: '1px solid #4b5563' }}
        />

        {/* Send Button */}
        <button
          onClick={handleSend}
          disabled={loading}
          style={{ background: '#2563eb', border: 'none', borderRadius: '8px', padding: '0 16px', color: '#fff', fontWeight: 'bold', cursor: 'pointer' }}
        >
          Send
        </button>
      </div>
    </div>
  );
};
