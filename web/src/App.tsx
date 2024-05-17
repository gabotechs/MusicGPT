import { useEffect, useRef } from "react";

import ChatInput from "./ChatInput";
import { ChatMessage, useChat } from "./useChat.ts";
import { StatusIndicator } from "./StatusIndicator.tsx";
import UserQuestion from "./UserMessage.tsx";
import AudioGenerating from "./AudioGenerating.tsx";
import { AudioSuccess } from "./AudioSuccess.tsx";
import AudioFailure from "./AudioFailure.tsx";
import ThemeToggle from "./ThemeToggle.tsx";
import { useThemeToggle } from "./ThemeToggleHook.tsx";


function App () {
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const { sendMessage, abortLast, history } = useChat()
  const newest = history[history.length - 1] as undefined | ChatMessage;

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [history.length]);

  const { theme, toggleTheme } = useThemeToggle()

  return (
    <div
      className={`h-screen w-full mx-auto flex justify-start align-middle flex-col bg-[var(--background-color)] text-[var(--text-color)] ${theme === 'dark' ? 'dark' : ''}`}>
      <div className="absolute top-0 w-full z-10 backdrop-blur-md flex items-center justify-between px-4 py-2">
        <div className="w-1/3"></div>
        <StatusIndicator className="m-2"/>
        <div className="w-1/3 flex flex-row justify-end">
          <ThemeToggle className="mr-2" onToggle={toggleTheme} theme={theme}/>
        </div>
      </div>
      <div className="overflow-auto px-2" ref={chatContainerRef}>
        <div className="h-20"/>
        <div className="flex-1 flex flex-col max-w-3xl mx-auto">
          {history.map(msg => {
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
        <div className="h-20"/>
      </div>
      <div className="absolute bottom-0 w-full">
        <ChatInput
          className={'max-w-3xl p-2 mx-auto'}
          onSend={sendMessage}
          onCancel={abortLast}
          loading={newest?.type == "ai" && newest.progress < 1}
        />
      </div>
    </div>
  );
}

export default App;
