import { useEffect, useRef, useState } from "react";

import ChatInput from "./ChatInput";
import { useBackend } from "./backend";
import { ChatMessage, useMessaging } from "./useMessaging.ts";
import { StatusIndicator } from "./StatusIndicator.tsx";
import UserQuestion from "./UserMessage.tsx";
import GeneratingAudio from "./GeneratingAudio.tsx";
import { Audio } from "./Audio.tsx";
import AudioDurationSelector from "./AudioDurationSelector.tsx";


function App () {
  const [audioDuration, setAudioDuration] = useState(10);

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const { send } = useBackend();
  const { sendMessage, history } = useMessaging()
  const newest = history[history.length - 1] as undefined | ChatMessage;

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [history.length]);

  function onSend (prompt: string) {
    sendMessage(prompt, audioDuration)
  }

  function onCancel () {
    if (newest?.type === "ai" && newest?.progress < 1) {
      send({ Abort: newest.id });
    }
  }

  return (
    <div className="h-screen w-full mx-auto flex justify-start align-middle flex-col">
      <div className="absolute top-0 w-full z-10 backdrop-blur-md flex items-center justify-between px-4 py-2">
        <div className="w-1/3"></div>
        <StatusIndicator className="m-2" />
        <div className="w-1/3 flex justify-end">
          <AudioDurationSelector className="m-2" value={audioDuration} onChange={setAudioDuration} />
        </div>
      </div>
      <div className="overflow-auto" ref={chatContainerRef}>
        <div className="h-20"/>
        <div className="flex-1 flex flex-col max-w-3xl mx-auto">
          {history.map(msg => msg.type === 'user'
            ? <UserQuestion
              className={'ml-16 self-end mb-8'}
              key={msg.id + msg.type}
              text={msg.text}
            />
            : msg.progress < 1
              ? <GeneratingAudio
                className={'mb-8'}
                key={msg.id + msg.type}
                progress={msg.progress}
              />
              : <Audio
                className={'mr-16 self-start mb-8'}
                key={msg.id + msg.type}
                src={msg.url}
              />
          )}
        </div>
        <div className="h-20"/>
      </div>
      <div className="absolute bottom-0 w-full">
        <ChatInput
          className={'max-w-3xl m-2 mx-auto'}
          onSend={onSend}
          onCancel={onCancel}
          loading={newest?.type == "ai" && newest.progress < 1}
        />
      </div>
    </div>
  );
}

export default App;
