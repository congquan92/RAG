/**
 * Theme Toggle Component
 * ======================
 *
 * Sun/Moon icon button that toggles between light and dark themes.
 */

import { useEffect } from "react";
import { Sun, Moon } from "lucide-react";
import { useThemeStore } from "@/stores/useThemeStore";
import { Button } from "./button";

export function ThemeToggle() {
  const { theme, toggleTheme } = useThemeStore();

  useEffect(() => {
    const root = document.documentElement;
    root.classList.add("theme-transition");
    root.setAttribute("data-theme", theme);
    const timeout = setTimeout(() => {
      root.classList.remove("theme-transition");
    }, 350);
    return () => clearTimeout(timeout);
  }, [theme]);

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      title={theme === "dark" ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
      className="size-8"
    >
      {theme === "dark" ? (
        <Sun className="size-4" />
      ) : (
        <Moon className="size-4" />
      )}
    </Button>
  );
}
