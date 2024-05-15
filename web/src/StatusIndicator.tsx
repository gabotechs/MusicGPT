import { useBackend } from "./backend.ts";
import { ReadyState } from "react-use-websocket";

function textAndColor(state: ReadyState)  {
  switch (state) {
    case ReadyState.CLOSING:
    case ReadyState.CLOSED:
    case ReadyState.UNINSTANTIATED:
      return ['bg-red-500', 'Disconnected']
    case ReadyState.CONNECTING:
      return ['bg-yellow-500', 'Connecting']
    case ReadyState.OPEN:
      return ['bg-green-500', 'Ready']
  }
}

export function StatusIndicator({ className }: { className?: string } = {}) {
  const { readyState } = useBackend()
  const [color, status] = textAndColor(readyState)
  return <div className={`p-2 w-36 ${color} text-center text-white font-semibold rounded ${className}`}>
    {status}
  </div>
}
