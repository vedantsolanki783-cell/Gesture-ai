export type Role = 'user' | 'model';

export interface Message {
  id: string;
  role: Role;
  text: string;
  timestamp: number;
}

export type CustomAction =
  | 'THEME_SWITCH'
  | 'CLEAR'
  | 'SEND_MESSAGE'
  | 'COPY_LAST'
  | 'SCROLL_UP'
  | 'SCROLL_DOWN'
  | 'TOGGLE_MIC';

export interface CustomGesture {
  id: string;
  name: string;
  description: string;
  action: CustomAction;
}

export interface AppSettings {
  theme: 'dark' | 'light';
  voiceEnabled: boolean;
  visionEnabled: boolean;
  confidenceThreshold: number;
  aiProvider: 'ollama' | 'gemini' | 'cloudFree' | 'auto';
  ollamaUrl: string;
  ollamaModel: string;
  geminiModel: string;
  cloudFreeBaseUrl: string;
  cloudFreeApiKey: string;
  cloudFreeModel: string;
  systemInstruction: string;
  customGestures: CustomGesture[];
}

export interface VisionResult {
  type: 'LETTER' | 'GESTURE' | 'UNKNOWN' | 'ERROR';
  value: string;
  confidence: number;
  source: 'local' | 'gemini' | 'none';
}

export interface Landmark {
  x: number;
  y: number;
  z?: number;
}
