import type { Theme } from "../lib/useTheme";

/** Sun/moon toggle. Inline SVG so there is no icon dependency to load. */
export function ThemeToggle({
  theme,
  onToggle,
}: {
  theme: Theme;
  onToggle: () => void;
}) {
  const goingTo = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      aria-label={`Switch to ${goingTo} theme`}
      title={`Switch to ${goingTo} theme`}
    >
      {theme === "dark" ? (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2" fill="currentColor" />
          <g
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinecap="round"
          >
            <path d="M12 2.6v2.3M12 19.1v2.3M21.4 12h-2.3M4.9 12H2.6" />
            <path d="M18.6 5.4l-1.6 1.6M7 17l-1.6 1.6M18.6 18.6L17 17M7 7L5.4 5.4" />
          </g>
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M20.7 14.6A8.6 8.6 0 1 1 9.4 3.3a6.9 6.9 0 0 0 11.3 11.3Z"
            fill="currentColor"
          />
        </svg>
      )}
    </button>
  );
}
