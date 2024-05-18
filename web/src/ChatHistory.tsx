import UserQuestion from "./components/UserMessage.tsx";
import AudioGenerating from "./components/AudioGenerating.tsx";
import AudioFailure from "./components/AudioFailure.tsx";
import { AudioSuccess } from "./components/AudioSuccess.tsx";
import { ChatMessage } from "./backend/useChat.ts";


export interface ChatHistoryProps {
  messages: ChatMessage[]
  className?: string
}

export function ChatHistory ({ messages, className = '' }: ChatHistoryProps) {
  return <div className={`flex-1 flex flex-col max-w-3xl mx-auto ${className}`}>
    {messages.map(msg => {
        const key = msg.type + msg.id
        if (msg.type === 'user') {
          return <UserQuestion
            className={'ml-16 self-end mb-8'}
            key={key}
            text={msg.text}
          />
        } else if (msg.progress < 1) {
          return <AudioGenerating
            className={'mb-8'}
            key={key}
            progress={msg.progress}
          />
        } else if (msg.error !== undefined) {
          return <AudioFailure
            className={'mr-16 self-start mb-8'}
            key={key}
            msg={msg.error}
          />
        } else if (msg.url !== undefined) {
          return <AudioSuccess
            className={'mr-16 self-start mb-8'}
            key={key}
            src={msg.url}
          />
        } else {
          return null
        }
      }
    )}
  </div>
}
