import React from 'react';
import { LoadingIcon } from "./Icons/LoadingIcon.tsx";

interface GeneratingAudioProps {
  className?: string;
  progress: number;
}

const AudioGenerating: React.FC<GeneratingAudioProps> = ({ className = '', progress }) => {
  const percentProgress = Math.round(progress * 100)
  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center space-x-2 text-[var(--text-faded-color)]">
        <LoadingIcon/>
        <span>Generating audio response...</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full"
          style={{ width: `${percentProgress}%` }}
        />
      </div>
      <div className="text-right text-[var(--text-faded-color)] text-sm">{percentProgress}%</div>
    </div>
  );
};


export default AudioGenerating;
