/**
 * Sign -> Text.
 *
 * The camera loop and the suggestion loop are deliberately independent: frames
 * go out on a fixed timer, and suggestions arrive whenever the backend has
 * something new. Neither waits for the other.
 */

import { useCallback, useEffect, useRef } from "react";
import { CameraStage } from "../components/CameraStage";
import { SuggestionPanel } from "../components/SuggestionPanel";
import { TextPanel } from "../components/TextPanel";
import { Card, Notice } from "../components/ui";
import { useCamera } from "../lib/useCamera";
import type { AppConfig } from "../lib/types";
import type { useRecognitionSocket } from "../lib/useRecognitionSocket";

type Socket = ReturnType<typeof useRecognitionSocket>;

interface Props {
  socket: Socket;
  config: AppConfig | null;
}

export function SignToTextView({ socket, config }: Props) {
  const camera = useCamera();
  const { capture, status: cameraStatus } = camera;
  const { sendFrame, sendCommand, connection } = socket;

  const fps = config?.recommended_fps ?? 15;
  const live = cameraStatus === "running" && connection === "open";

  // --- frame pump ----------------------------------------------------
  const liveRef = useRef(live);
  liveRef.current = live;

  useEffect(() => {
    if (!live) return;
    const interval = window.setInterval(() => {
      if (!liveRef.current) return;
      const image = capture();
      if (image) sendFrame(image);
    }, Math.round(1000 / fps));
    return () => window.clearInterval(interval);
  }, [live, fps, capture, sendFrame]);

  // --- keyboard shortcuts --------------------------------------------
  const handleKey = useCallback(
    (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA)$/.test(target.tagName)) return;

      if (event.key === " ") {
        event.preventDefault();
        sendCommand("space");
      } else if (event.key === "Backspace") {
        event.preventDefault();
        sendCommand(event.ctrlKey ? "delete_word" : "backspace");
      } else if (event.key === "Escape") {
        sendCommand("clear");
      }
    },
    [sendCommand],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  const busy = connection !== "open";

  return (
    <>
      {config && !config.gemini_enabled && (
        <Notice
          tone="warn"
          title="Smart suggestions are off"
          detail="No GEMINI_API_KEY is configured, so suggestions come from the built-in engine. Everything else works exactly the same."
        />
      )}

      <div className="workspace">
        <div className="stack">
          <CameraStage
            videoRef={camera.videoRef}
            cameraStatus={camera.status}
            cameraError={camera.error}
            connection={connection}
            frame={socket.frame}
            onStart={camera.start}
            onStop={camera.stop}
            onRetryConnection={socket.retry}
          />

          <Card title="Your sentence">
            <TextPanel
              text={socket.text}
              disabled={busy}
              onSpace={() => sendCommand("space")}
              onBackspace={() => sendCommand("backspace")}
              onDeleteWord={() => sendCommand("delete_word")}
              onClear={() => sendCommand("clear")}
            />
          </Card>
        </div>

        <div className="stack">
          <SuggestionPanel
            suggestions={socket.suggestions}
            onPickWord={(word) => sendCommand("accept_word", word)}
            onPickSentence={(sentence) =>
              sendCommand("accept_sentence", sentence)
            }
          />

          <Card title="How to spell">
            <ol
              style={{
                margin: 0,
                paddingLeft: "1.2rem",
                color: "var(--text-muted)",
                fontSize: "0.92rem",
                display: "grid",
                gap: "0.35rem",
              }}
            >
              <li>Hold a sign steady until the bar under the camera fills.</li>
              <li>
                Change pose between letters - the same held sign is only typed
                once, so LL in HELLO needs two deliberate holds.
              </li>
              <li>
                Tap a suggestion to finish the word instead of spelling it out.
              </li>
              <li>
                Keyboard: <kbd>Space</kbd> finishes a word, <kbd>Backspace</kbd>{" "}
                deletes a letter, <kbd>Esc</kbd> clears everything.
              </li>
            </ol>
          </Card>

          {socket.socketError && (
            <Notice tone="danger" title="Connection problem" detail={socket.socketError} />
          )}
        </div>
      </div>
    </>
  );
}
