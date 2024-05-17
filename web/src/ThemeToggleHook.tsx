import { useEffect, useState } from "react";

const THEME_KEY = 'selectedTheme';

export function useThemeToggle() {
  const [theme, setTheme] = useState<'light' | 'dark'>();
  // ... other state and refs

  useEffect(() => {
    const storedTheme = localStorage.getItem(THEME_KEY) as 'light' | 'dark';
    if (['light', 'dark'].includes(storedTheme)) {
      setTheme(storedTheme);
    } else {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setTheme(prefersDark ? 'dark' : 'light');
    }
  }, []);

  useEffect(() => {
    if (theme != null) localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  return { theme: theme ?? 'light', toggleTheme }
}

