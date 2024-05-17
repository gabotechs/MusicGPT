import React from 'react';

interface GeneratingAudioProps {
  className?: string;
  progress: number;
}

const GeneratingAudio: React.FC<GeneratingAudioProps> = ({ className = '', progress }) => {
  const percentProgress = Math.round(progress * 100)
  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center space-x-2 text-gray-500">
        <svg
          className="w-4 h-4 animate-spin"
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
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
        <span>Generating audio response...</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full"
          style={{ width: `${percentProgress}%` }}
        />
      </div>
      <div className="text-right text-gray-500 text-sm">{percentProgress}%</div>
    </div>
  );
};

export default GeneratingAudio;
