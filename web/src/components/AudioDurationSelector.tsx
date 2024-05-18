// AudioDurationSelector.tsx
import React, { HTMLProps } from 'react';

interface AudioDurationSelectorProps extends Omit<HTMLProps<HTMLDivElement>, 'value' | 'onChange'> {
  value: number;
  onChange: (value: number) => void;
}

const AudioDurationSelector: React.FC<AudioDurationSelectorProps> = ({ value, onChange, className = '' }) => {
  function handleChange (event: React.ChangeEvent<HTMLInputElement>) {
    const newValue = parseInt(event.target.value);
    onChange(newValue);
  }

  return (
    <div className={className}>
      <label htmlFor="audioDuration" className="text-sm font-medium">
        Audio Duration (seconds):
      </label>
      <input
        type="number"
        id="audioDuration"
        min="1"
        max="30"
        value={value}
        onChange={handleChange}
        className="ml-2 px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
    </div>
  );
};

export default AudioDurationSelector;
