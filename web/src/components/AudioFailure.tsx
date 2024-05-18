import React from 'react';
import { ErrorIcon } from "../Icons/ErrorIcon.tsx";

interface AudioFailureProps {
  className?: string;
  msg: string
}

const AudioFailure: React.FC<AudioFailureProps> = ({ className = '', msg }) => {
  return (
    <div className={`bg-red-100 p-4 rounded-b-lg rounded-tr-lg ${className}`}>
      <div className="flex items-center space-x-2">
        <ErrorIcon/>
        <span className="text-red-800 font-semibold">Audio Generation Failed</span>
      </div>
      <p className="mt-2 text-red-700">
        {msg}
      </p>
    </div>
  );
};

export default AudioFailure;
