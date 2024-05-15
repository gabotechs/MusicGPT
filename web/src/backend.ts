import useWebSocket from "react-use-websocket";
import { BackendAiInboundMsg, BackendAiOutboundMsg } from "./bindings";
import { useState } from "react";

const BACKEND_URL: string = import.meta.env.VITE_BACKEND_URL ?? window.location.origin
export const WS_URL = `${BACKEND_URL.replace('http', 'ws')}/ws`
export const FILES_URL = `${BACKEND_URL}/files`

export function useBackend () {
  const [closeEvent, setCloseEvent] = useState<WebSocketEventMap['close']>()

  const { sendJsonMessage, lastJsonMessage, readyState } =
    useWebSocket<BackendAiOutboundMsg>(WS_URL, {
      share: true,
      onClose: close => setCloseEvent(close)
    });

  function send (msg: BackendAiInboundMsg) {
    sendJsonMessage(msg);
  }

  const last = lastJsonMessage;

  return { send, last, readyState, closeEvent };
}
