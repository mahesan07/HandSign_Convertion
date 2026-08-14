/**
 * Theme selection.
 *
 * Light (lavender and white) is the default look; dark is available from the
 * header toggle and is used automatically when the operating system asks for
 * it. The choice is written to `data-theme` on <html>, which is what
 * global.css keys off, and remembered between visits.
 */

import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

const STORAGE_KEY = "handsign-theme";

function initialTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "dark" || stored === "light") return stored;
  } catch {
    /* private browsing can refuse localStorage; the default is fine */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#14121f" : "#f5f3fd");
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* not worth surfacing */
    }
  }, [theme]);

  const toggle = useCallback(
    () => setTheme((current) => (current === "dark" ? "light" : "dark")),
    [],
  );

  return { theme, toggle };
}
