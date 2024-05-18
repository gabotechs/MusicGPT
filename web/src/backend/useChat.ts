import { useEffect, useRef, useState } from "react";
import { v4 as uuid } from "uuid";

import { useBackend } from "./useBackend.ts";
import {
  AudioGenerationError,
  AudioGenerationProgress,
  AudioGenerationResult,
  AudioGenerationStart,
  Chat,
  ChatEntry
} from './bindings.ts'

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


export function useChat (chat_id: string | undefined, onNewChat: (chat_id: string) => void) {
  const [chatMetadata, setChatMetadata] = useState<Chat>()
  const chatIdRef = useRef(chat_id)

  const [history, setHistory] = useState<ChatHistory>();
  const { send, last } = useBackend();

  useEffect(() => {
    if (chat_id !== undefined) {
      send({ GetChat: { chat_id } })
      chatIdRef.current = chat_id
    } else {
      setHistory(undefined)
    }
  }, [chat_id, send]);

  useEffect(() => {
    if (last == null) {
      // do nothing
    } else if ('Generation' in last && 'Start' in last.Generation) {
      const msg = last.Generation.Start
      setHistory(prev => prev?.audioGenerationStart(msg))
    } else if ('Generation' in last && 'Progress' in last.Generation) {
      const msg = last.Generation.Progress
      setHistory(prev => prev?.audioGenerationProgress(msg))
    } else if ('Generation' in last && 'Result' in last.Generation) {
      const msg = last.Generation.Result
      setHistory(prev => prev?.audioGenerationResultOrError(msg))
    } else if ('Generation' in last && 'Error' in last.Generation) {
      const msg = last.Generation.Error
      setHistory(prev => prev?.audioGenerationResultOrError(msg))
    } else if ('Chat' in last) {
      const [chat, history] = last.Chat
      setChatMetadata(chat)
      setHistory(ChatHistory.fromHistory(chat.chat_id, history))
    }
  }, [last])

  function sendMessage (prompt: string, secs: number) {
    const id = uuid();
    if (chat_id !== undefined) {
      send({ GenerateAudio: { id, chat_id, prompt, secs: clamp(1, secs, 30) } });
    } else {
      const chat_id = uuid()
      send({ GenerateAudioNewChat: { id, chat_id, prompt, secs: clamp(1, secs, 30) } })
      setHistory(new ChatHistory(chat_id))
      onNewChat(chat_id)
    }
  }

  function abortLast () {
    if (chat_id === undefined) return
    const last = history?.lastAi()
    if (last !== undefined && last.progress < 1) {
      send({ AbortGeneration: { id: last.id, chat_id } });
    }
  }

  return { sendMessage, abortLast, history, chatMetadata }
}

class ChatHistory {
  constructor (
    readonly chatId: string,
    readonly list: ChatMessage[] = [],
    // a user msg and a response msg have the same id if
    // one is the response to the other.
    private readonly userDict: Record<string, UserMessage> = {},
    private readonly aiDict: Record<string, AiMessage> = {},
  ) {
  }

  lastAi (): AiMessage | undefined {
    for (let i = this.list.length - 1; i >= 0; --i) {
      if (this.list[i].type === 'ai') return this.list[i] as AiMessage
    }
  }

  audioGenerationStart (msg: AudioGenerationStart) {
    if (msg.chat_id != this.chatId) return this
    if (msg.id in this.userDict) return this
    const userMsg: UserMessage = {
      type: 'user',
      id: msg.id,
      text: msg.prompt
    }
    this.userDict[msg.id] = userMsg
    this.list.push(userMsg)
    return this.shallowCopy()
  }

  audioGenerationProgress (msg: AudioGenerationProgress) {
    if (msg.chat_id != this.chatId) return this
    if (msg.id in this.aiDict) {
      this.aiDict[msg.id].progress = msg.progress
      return this.shallowCopy()
    }
    const aiMsg: AiMessage = {
      type: "ai",
      id: msg.id,
      progress: msg.progress,
    }
    this.aiDict[msg.id] = aiMsg
    this.list.push(aiMsg)
    return this.shallowCopy()
  }

  audioGenerationResultOrError (msg: AudioGenerationResult | AudioGenerationError) {
    if (msg.chat_id != this.chatId) return this
    if (msg.id in this.aiDict) {
      this.aiDict[msg.id].progress = 1
      if ('relpath' in msg) {
        this.aiDict[msg.id].url = msg.relpath
      } else if ('error' in msg) {
        this.aiDict[msg.id].error = msg.error
      }
      return this.shallowCopy()
    }
    const aiMsg: AiMessage = {
      type: "ai",
      id: msg.id,
      progress: 1,
      url: 'relpath' in msg ? msg.relpath : undefined,
      error: 'error' in msg ? msg.error : undefined,
    }
    this.aiDict[msg.id] = aiMsg
    this.list.push(aiMsg)
    return this.shallowCopy()
  }

  shallowCopy () {
    return new ChatHistory(this.chatId, this.list, this.userDict, this.aiDict)
  }

  static fromHistory (chatId: string, entries: ChatEntry[]): ChatHistory {
    const chatHistory = new ChatHistory(chatId)
    for (const entry of entries) {
      if ('User' in entry) {
        const msg: UserMessage = {
          type: 'user',
          id: entry.User.id,
          text: entry.User.text,
        }
        chatHistory.list.push(msg)
        chatHistory.userDict[msg.id] = msg
      } else if ('Ai' in entry) {
        const msg: AiMessage = {
          type: 'ai',
          id: entry.Ai.id,
          url: entry.Ai.relpath || undefined,
          error: entry.Ai.error || undefined,
          progress: 1,
        }
        chatHistory.list.push(msg)
        chatHistory.aiDict[msg.id] = msg
      }
    }
    return chatHistory
  }
}

function clamp (min: number, num: number, max: number): number {
  return Math.max(Math.min(num, max), min);
}
