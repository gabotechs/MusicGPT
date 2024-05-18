import { useCallback, useEffect, useState } from "react";

import { useBackend } from "./useBackend.ts";
import { Chat } from "./bindings.ts";

export function useChats () {
  const [activeChat, setActiveChat] = useState<string>()
  const [chats, setChats] = useState<Chat[]>([])

  const { send, last } = useBackend()

  useEffect(() => {
    if (last == null) {
      // do nothing
    } else if ('Chats' in last) {
      setChats(last.Chats)
    }
  }, [last])

  const setChatMetadata = useCallback((chat_id: string, opts: { name?: string } = {}) => {
    if (Object.keys(opts).length === 0) return
    send({ SetChatMetadata: { chat_id, name: opts.name ?? null } })
  }, [send])

  const deleteChat = useCallback((chat_id: string) => {
    send({ DelChat: { chat_id } })
  }, [send])

  return { activeChat, setActiveChat, chats, setChatMetadata, deleteChat }
}
