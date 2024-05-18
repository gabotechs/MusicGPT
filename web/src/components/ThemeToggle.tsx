import React from 'react';
import { LightTheme } from "../Icons/LightTheme.tsx";
import { DarkTheme } from "../Icons/DarkTheme.tsx";

interface ThemeToggleProps {
  className?: string;
  theme: 'light' | 'dark';
  onToggle: () => void;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ className = '', onToggle, theme }) => {
  return (
    <button
      className={`p-2 rounded-full focus:outline-none  ${className}`}
      onClick={onToggle}
    >
      {theme === 'light' ? <LightTheme/> : <DarkTheme/>}
    </button>
  );
};


export default ThemeToggle;
