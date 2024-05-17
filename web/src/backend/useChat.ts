import { useEffect, useState } from "react";
import { v4 as uuid } from "uuid";

import { FILES_URL, useBackend } from "./useBackend.ts";

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

const DEFAULT_CHAT_ID = '39b099b5-9eaf-4ac3-8d4b-1380369090b5'

export function useChat () {
  const [activeChat,] = useState(DEFAULT_CHAT_ID)

  const [history, setHistory] = useState<ChatMessage[]>([]);
  const { send, last } = useBackend();

  useEffect(() => {
    send({ RetrieveHistory: { chat_id: activeChat } })
  }, [activeChat, send]);

  useEffect(() => {
    if (last == null) {
      // do nothing
    } else if ('Progress' in last) {
      const { id, progress } = last.Progress
      setHistory(prev => {
        const newest = prev[prev.length - 1] as undefined | ChatMessage;
        if (newest?.id === id && newest.type === 'ai') {
          newest.progress = progress
        }
        return [...prev]
      })
    } else if ('Result' in last) {
      const { id, relpath } = last.Result
      setHistory(prev => {
        const newest = prev[prev.length - 1] as undefined | ChatMessage;
        if (newest?.id === id && newest.type === 'ai') {
          newest.progress = 1
          newest.url = FILES_URL + (relpath.startsWith('/') ? relpath : `/${relpath}`)
        }
        return [...prev]
      })
    } else if ('Error' in last) {
      const { id, error } = last.Error
      setHistory(prev => {
        const newest = prev[prev.length - 1] as undefined | ChatMessage;
        if (newest?.id === id && newest.type === 'ai') {
          newest.progress = 1
          newest.error = error
        }
        return [...prev]
      })
    } else if ('History' in last) {
      const [, history] = last.History
      const newHistory: ChatMessage[] = []
      for (const entry of history) {
        if ('User' in entry) {
          newHistory.push({
            type: 'user',
            id: entry.User.id,
            text: entry.User.text,
          })
        } else if ('Ai' in entry) {
          newHistory.push({
            type: 'ai',
            id: entry.Ai.id,
            url: entry.Ai.relpath,
            progress: 1,
          })
        }
      }
      setHistory(newHistory)
    }
  }, [last])

  function sendMessage (prompt: string, secs: number) {
    const id = uuid();
    const chat_id = activeChat;
    send({ GenerateAudio: { id, chat_id, prompt, secs: clamp(1, secs, 30) } });
    setHistory((prev) => {
      prev.push({ type: "user", id, text: prompt });
      prev.push({ type: "ai", id, progress: 0 });
      return [...prev]
    });
  }

  function abortLast () {
    const newest = history[history.length - 1] as undefined | ChatMessage;
    if (newest?.type === "ai" && newest?.progress < 1) {
      send({ AbortGeneration: { id: newest.id, chat_id: activeChat } });
    }
  }

  return { sendMessage, abortLast, history }
}

function clamp (min: number, num: number, max: number): number {
  return Math.max(Math.min(num, max), min);
}
