import React from 'react';
import './ResponsiveDrawer.css';

export interface ResponsiveDrawerEntry {
  onRename (newName: string): void;

  id: string
  name: string
  date: Date
}

export interface ChatDrawerProps {
  open: boolean;
  setOpen: (open: boolean | ((prev: boolean) => boolean)) => void;
  entries: ResponsiveDrawerEntry[];
  selectedEntry?: string;
  onSelectEntry: (chatId?: string) => void;
}

const ResponsiveDrawer: React.FC<ChatDrawerProps> = ({ open, setOpen, entries, selectedEntry, onSelectEntry }) => {
  function handleSelectEntry () {
    onSelectEntry(undefined)
    setOpen(false)
  }

  return (
    <>
      <div className={`chat-drawer ${open ? 'open' : ''}`}>
        <div className="chat-drawer-header">
          <button className="new-chat-button" onClick={handleSelectEntry}>
            + New Chat
          </button>
          <button className="close-button" onClick={() => setOpen(false)}>
            &times;
          </button>
        </div>
        <ul className="chat-list">
          {entries.map((entry) => (
            <li
              key={entry.id}
              className={`chat-item ${selectedEntry === entry.id ? 'selected' : ''}`}
              onClick={() => {
                onSelectEntry(entry.id);
                setOpen(false)
              }}
            >
              <div>
                <span className="text-[var(--text-color)] line-clamp-2">{entry.name}</span>
                <span className="text-[var(--text-faded-color)] text-xs">
                  {entry.date.toLocaleString()}
                </span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
};

export default ResponsiveDrawer;
