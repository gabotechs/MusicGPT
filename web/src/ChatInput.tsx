import React, { useEffect, useRef, useState } from "react";

export interface ChatInputProps {
  className?: string;
  onSend (text: string): void;
  onCancel (): void;
  loading: boolean;
}

const ChatInput = ({ className = '', onSend, loading, onCancel }: ChatInputProps) => {
  const [aborting, setAborting] = useState(false)

  useEffect(() => {
    if (!loading) setAborting(false)
  }, [loading])

  const inputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState("");

  const handleChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    setInputValue(e.target.value);
  };

  const handleSubmit = (e: { preventDefault(): void }) => {
    e.preventDefault(); // Prevents the default form submission behavior
    if (loading) return;
    if (inputValue.trim()) {
      onSend(inputValue);
      setInputValue(""); // Clears the input after sending
      inputRef.current?.focus()
    }
  };

  const handleCancel = (e: { preventDefault(): void }) => {
    e.preventDefault(); // Prevents the default form submission behavior
    if (!loading) return;
    onCancel()
    setAborting(true)
  }

  return (
    <form onSubmit={handleSubmit} className={`p-4 flex ${className}`}>
      <input
        type="text"
        ref={inputRef}
        value={inputValue}
        onChange={handleChange}
        placeholder="Type your message..."
        className="flex-grow px-4 py-2 mr-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
        className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {(() => {
          if (aborting) return Aborting()
          if (loading) return Stop()
          return Send()
        })()}
      </button>
    </form>
  );
};

function Aborting() {
  return (
    <svg
      className="animate-spin h-6 w-6 text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      ></circle>
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      ></path>
    </svg>
  )
}

function Stop () {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="h-6 w-6"
      fill="currentColor"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M5 5h14v14H5z"
      />
    </svg>
  )
}

function Send () {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M5 10l7-7m0 0l7 7m-7-7v18"
      />
    </svg>
  )
}

export default ChatInput;
