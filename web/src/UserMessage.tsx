import React, { HTMLProps } from 'react';

interface UserQuestionProps extends HTMLProps<HTMLDivElement> {
  text: string;
}

const UserQuestion: React.FC<UserQuestionProps> = ({ text, className = '' }) => {
  return (
    <div className={`bg-gray-50 p-4 rounded-b-lg rounded-tl-lg ${className}`}>
      <p>{text}</p>
    </div>
  );
};

export default UserQuestion;
