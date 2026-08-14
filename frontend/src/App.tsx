import { useEffect, useState } from "react";
import { ThemeToggle } from "./components/ThemeToggle";
import { Notice } from "./components/ui";
import { api, assetUrl } from "./lib/api";
import { useTheme } from "./lib/useTheme";
import { EMPTY_CATALOG } from "./lib/signAssets";
import type { AppConfig, SignCatalog } from "./lib/types";
import { useRecognitionSocket } from "./lib/useRecognitionSocket";
import { SignToTextView } from "./views/SignToTextView";
import { TextToSignView } from "./views/TextToSignView";

type Mode = "sign-to-text" | "text-to-sign";

export default function App() {
  const [mode, setMode] = useState<Mode>("sign-to-text");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [catalog, setCatalog] = useState<SignCatalog>(EMPTY_CATALOG);
  const [bootError, setBootError] = useState<string | null>(null);
  const { theme, toggle: toggleTheme } = useTheme();

  // The socket stays open across mode switches so the sentence survives a trip
  // to Text to Sign and back.
  const socket = useRecognitionSocket(true);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.config(), api.signCatalog()])
      .then(([loadedConfig, loadedCatalog]) => {
        if (cancelled) return;
        setConfig(loadedConfig);
        setCatalog(loadedCatalog);
        setBootError(null);
      })
      .catch((error: Error) => {
        if (!cancelled) setBootError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app">
      <header className="app__header raised">
        <div className="brand">
          <span className="brand__mark">
            <img src={assetUrl("/signs/H.svg")} alt="" aria-hidden="true" />
          </span>
          <div>
            <h1 className="brand__title">HandSign Conversion</h1>
            <p className="brand__subtitle">
              Spell with hand signs to build sentences, or turn typed English
              back into signs.
            </p>
          </div>
        </div>

        <div className="header__actions">
          <div
            className="mode-switch sunken"
            role="group"
            aria-label="Conversion direction"
          >
            <button
              type="button"
              className="mode-switch__option"
              aria-pressed={mode === "sign-to-text"}
              onClick={() => setMode("sign-to-text")}
            >
              Sign to Text
            </button>
            <button
              type="button"
              className="mode-switch__option"
              aria-pressed={mode === "text-to-sign"}
              onClick={() => setMode("text-to-sign")}
            >
              Text to Sign
            </button>
          </div>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      <main className="stack">
        {bootError && (
          <Notice
            tone="danger"
            title="Could not load the app configuration"
            detail={`${bootError} Start the backend with: uvicorn backend.app.main:app --reload`}
          />
        )}

        {mode === "sign-to-text" ? (
          <SignToTextView socket={socket} config={config} />
        ) : (
          <TextToSignView catalog={catalog} />
        )}
      </main>

      <footer className="app__footer">
        {config
          ? `${config.classes.length} letters recognised · ${
              config.gemini_enabled
                ? `smart suggestions via ${config.gemini_model}`
                : "local suggestions only"
            } · v${config.version}`
          : "Connecting to the recognition server..."}
      </footer>
    </div>
  );
}
