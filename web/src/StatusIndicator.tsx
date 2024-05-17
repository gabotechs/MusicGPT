import { ReadyState } from "react-use-websocket";

import { useBackend } from "./backend/useBackend.ts";
import { ErrorIcon } from "./Icons/ErrorIcon.tsx";
import { WarningIcon } from "./Icons/WarningIcon.tsx";
import { CheckIcon } from "./Icons/CheckIcon.tsx";

export function StatusIndicator ({ className }: { className?: string } = {}) {
  const { readyState, closeEvent, info } = useBackend()
  const [icon, status] = textAndColor(readyState)
  return <div className={`flex items-center space-x-2 p-2 bg-[var(--card-background-color)] rounded ${className}`}>
    {icon}
    {readyState === ReadyState.OPEN ? (
      <span className="text-[var(--text-color)]">{info != null ? `${info.model} (${info.device})` : ''}</span>
    ) : (
      <span className="text-[var(--text-color)]">
        {closeEvent?.reason && closeEvent.reason.length > 0 ? closeEvent.reason : status}
      </span>
    )}
  </div>
}

function textAndColor (state: ReadyState) {
  switch (state) {
    case ReadyState.CLOSING:
    case ReadyState.CLOSED:
    case ReadyState.UNINSTANTIATED:
      return [<ErrorIcon/>, 'Disconnected']
    case ReadyState.CONNECTING:
      return [<WarningIcon/>, 'Connecting']
    case ReadyState.OPEN:
      return [<CheckIcon/>, 'Ready']
  }
}
