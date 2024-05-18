import { useEffect, useMemo, useRef, useState } from "react";

import ChatInput from "./components/ChatInput.tsx";
import { useChat } from "./backend/useChat.ts";
import { StatusIndicator } from "./StatusIndicator.tsx";
import ThemeToggle from "./components/ThemeToggle.tsx";
import { useThemeToggle } from "./components/ThemeToggleHook.tsx";
import { ChatHistory } from "./ChatHistory.tsx";
import { useChats } from "./backend/useChats.ts";
import ResponsiveDrawer, { ResponsiveDrawerEntry } from "./components/ResponsiveDrawer.tsx";
import { ToggleButton } from "./components/ToggleButton.tsx";

// const DEFAULT_CHAT_ID = '39b099b5-9eaf-4ac3-8d4b-1380369090b5'

function App () {
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [drawerOpen, setDrawerOpen] = useState(false)

  const { chats, activeChat, setActiveChat, setChatMetadata } = useChats()
  const { sendMessage, abortLast, history } = useChat(activeChat, setActiveChat)

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [history?.list.length]);

  const { theme, toggleTheme } = useThemeToggle()

  const drawerEntries: ResponsiveDrawerEntry[] = useMemo(
    () => chats.map(chat => ({
      id: chat.chat_id,
      name: chat.name,
      date: new Date(chat.created_at),
      onRename: name => setChatMetadata(chat.chat_id, { name })
    })),
    [setChatMetadata, chats]
  )

  return (
    <div
      className={`h-screen w-full mx-auto flex justify-start align-middle flex-col bg-[var(--background-color)] text-[var(--text-color)] ${theme === 'dark' ? 'dark' : ''}`}>
      <ResponsiveDrawer
        open={drawerOpen}
        setOpen={setDrawerOpen}
        entries={drawerEntries}
        selectedEntry={activeChat}
        onSelectEntry={setActiveChat}
      />
      <div className="absolute top-0 w-full z-10 flex items-center justify-between px-4 py-2">
        <div className="w-1/3 flex flex-row justify-start">
          <ToggleButton onClick={() => setDrawerOpen(true)}/>
        </div>
        <StatusIndicator className="m-2 w-fit"/>
        <div className="w-1/3 flex flex-row justify-end">
          <ThemeToggle className="mr-2" onToggle={toggleTheme} theme={theme}/>
        </div>
      </div>
      <div className="overflow-auto px-2" ref={chatContainerRef}>
        <div className="h-20"/>
        <ChatHistory messages={history?.list ?? []}/>
        <div className="h-20"/>
      </div>
      <div className="absolute bottom-0 w-full">
        <ChatInput
          className={'max-w-3xl p-2 mx-auto'}
          inputFocusToken={activeChat}
          onSend={sendMessage}
          onCancel={abortLast}
          loading={(history?.lastAi()?.progress ?? 1) < 1}
        />
      </div>
    </div>
  );
}

export default App;
