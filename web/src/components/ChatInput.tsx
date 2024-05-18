import React, { useEffect, useRef, useState } from "react";
import { LoadingIcon } from "../Icons/LoadingIcon.tsx";
import { StopIcon } from "../Icons/StopIcon.tsx";
import { SendIcon } from "../Icons/SendIcon.tsx";

export interface ChatInputProps {
  className?: string;
  loading: boolean;
  inputFocusToken?: string

  onSend (text: string, secs: number): void;

  onCancel (): void;
}

const ChatInput = ({ className = '', inputFocusToken, onSend, loading, onCancel }: ChatInputProps) => {
  const [audioDuration, setAudioDuration] = useState(10)

  const [aborting, setAborting] = useState(false)

  useEffect(() => {
    if (!loading) setAborting(false)
  }, [loading])


  const inputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    inputRef.current?.focus()
  }, [inputFocusToken])

  const handleChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    setInputValue(e.target.value);
  };

  const handleSubmit = (e: { preventDefault (): void }) => {
    e.preventDefault(); // Prevents the default form submission behavior
    if (loading) return;
    if (inputValue.trim()) {
      onSend(inputValue, audioDuration);
      setInputValue(""); // Clears the input after sending
      inputRef.current?.focus()
    }
  };

  const handleCancel = (e: { preventDefault (): void }) => {
    e.preventDefault(); // Prevents the default form submission behavior
    if (!loading) return;
    onCancel()
    setAborting(true)
  }

  function handleAudioChange (event: React.ChangeEvent<HTMLInputElement>) {
    const newValue = parseInt(event.target.value);
    setAudioDuration(newValue);
  }

  return (
    <form onSubmit={handleSubmit} className={`flex relative ${className}`}>
      <input
        type="number"
        id="audioDuration"
        min="1"
        max="30"
        placeholder={"Duration (s)"}
        value={audioDuration}
        onChange={handleAudioChange}
        className="ml-2 px-2 py-1 border rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500 bg-[var(--input-background-color)] text-[var(--input-text-color)] border-[var(--input-border-color)]"
      />
      <label
        htmlFor="audioDuration"
        className="absolute left-3 top-0 transform -translate-y-1/2 text-gray-500 text-xs pointer-events-none"
      >
        Duration (s)
      </label>
      <input
        type="text"
        ref={inputRef}
        value={inputValue}
        onChange={handleChange}
        placeholder="Type your message..."
        className="flex-grow px-4 py-2 mx-2 rounded-lg border focus:outline-none focus:ring-1 focus:ring-blue-500 bg-[var(--input-background-color)] text-[var(--input-text-color)] border-[var(--input-border-color)]"
      />
      <button
        type={loading ? 'button' : 'submit'}
        disabled={aborting}
        onClick={e => {
          if (loading) {
            handleCancel(e)
          } else {
            handleSubmit(e)
          }
        }}
        className="p-2 bg-blue-500 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-500 text-white"
      >
        {(() => {
          if (aborting) return LoadingIcon()
          if (loading) return StopIcon()
          return SendIcon()
        })()}
      </button>
    </form>
  );
};


export default ChatInput;
