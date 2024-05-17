import { useEffect, useState } from "react";
import { v4 as uuid } from "uuid";

import { FILES_URL, useBackend } from "./backend.ts";

export interface UserMessage {
  type: "user";
  id: string;
  text: string;
}

export interface AiMessage {
  type: "ai";
  id: string;
  progress: number;
  url?: string
  error?: string;
}

export type ChatMessage = UserMessage | AiMessage;

export function useMessaging() {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const { send, last } = useBackend();

  useEffect(() => {
    if (last == null) {
      // do nothing
    } else if ('AudioGenerationProgress' in last) {
      const [id, progress] = last.AudioGenerationProgress
      setHistory(prev => {
        const newest = prev[prev.length - 1] as undefined | ChatMessage;
        if (newest?.id === id && newest.type === 'ai') {
          newest.progress = progress
        }
        return [...prev]
      })
    } else if ('AudioGenerationResponse' in last) {
      const [id, relPath] = last.AudioGenerationResponse
      setHistory(prev => {
        const newest = prev[prev.length - 1] as undefined | ChatMessage;
        if (newest?.id === id && newest.type === 'ai') {
          newest.progress = 1
          newest.url = FILES_URL + (relPath.startsWith('/') ? relPath : `/${relPath}`)
        }
        return [...prev]
      })
    } else if ('AudioGenerationFailure' in last) {
      const [id, msg] = last.AudioGenerationFailure
      setHistory(prev => {
        const newest = prev[prev.length - 1] as undefined | ChatMessage;
        if (newest?.id === id && newest.type === 'ai') {
          newest.progress = 1
          newest.error = msg
        }
        return [...prev]
      })
    }
  }, [last])

  function sendMessage(prompt: string, secs: number) {
    const id = uuid();
    send({ AudioGenerationRequest: { id, prompt, secs: clamp(1, secs, 30) } });
    setHistory((prev) => {
      prev.push({ type: "user", id, text: prompt });
      prev.push({ type: "ai", id, progress: 0 });
      return [...prev]
    });
  }

  return { sendMessage, history }
}

function clamp(min: number, num: number, max: number): number {
  return Math.max(Math.min(num, max), min);
}
