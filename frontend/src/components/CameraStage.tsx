/** The live camera, the hand overlay and the recognition read-out. */

import { useEffect, useRef } from "react";
import type { RecognitionFrame } from "../lib/useRecognitionSocket";
import type { CameraError, CameraStatus } from "../lib/useCamera";
import type { ConnectionState } from "../lib/types";
import { Button, Notice } from "./ui";

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** Plain-language status, so nothing is communicated by colour alone. */
const STATUS_TEXT: Record<RecognitionFrame["status"], string> = {
  idle: "Show your hand to the camera",
  low_confidence: "Sign unclear - adjust your hand or the lighting",
  detecting: "Reading your sign - keep holding",
  locking: "Almost there - hold it",
  committed: "Letter added",
  hold_release: "Added - change pose for the next letter",
  cooldown: "Just a moment",
};

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  cameraStatus: CameraStatus;
  cameraError: CameraError | null;
  connection: ConnectionState;
  frame: RecognitionFrame;
  onStart: () => void;
  onStop: () => void;
  onRetryConnection: () => void;
}

export function CameraStage({
  videoRef,
  cameraStatus,
  cameraError,
  connection,
  frame,
  onStart,
  onStop,
  onRetryConnection,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const running = cameraStatus === "running";

  // Draw the skeleton on every new frame. The landmarks arrive in the same
  // mirrored space the video is displayed in, so they need no transform.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const { width, height } = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(width) || canvas.height !== Math.round(height)) {
      canvas.width = Math.round(width);
      canvas.height = Math.round(height);
    }
    context.clearRect(0, 0, canvas.width, canvas.height);

    const points = frame.landmarks;
    if (!points.length) return;

    const scaled = points.map(
      ([x, y]) => [x * canvas.width, y * canvas.height] as const,
    );

    context.strokeStyle = "rgba(155, 140, 255, 0.9)";
    context.lineWidth = 3;
    context.lineCap = "round";
    context.beginPath();
    for (const [start, end] of HAND_CONNECTIONS) {
      const a = scaled[start];
      const b = scaled[end];
      if (!a || !b) continue;
      context.moveTo(a[0], a[1]);
      context.lineTo(b[0], b[1]);
    }
    context.stroke();

    context.fillStyle = "#f472b6";
    for (const [x, y] of scaled) {
      context.beginPath();
      context.arc(x, y, 4, 0, Math.PI * 2);
      context.fill();
    }
  }, [frame.landmarks]);

  const confidencePercent = Math.round(frame.confidence * 100);

  return (
    <div className="stack">
      <div className="stage raised">
        <video
          ref={videoRef}
          className="stage__video"
          playsInline
          muted
          aria-label="Live camera view"
          style={{ visibility: running ? "visible" : "hidden" }}
        />
        <canvas
          ref={canvasRef}
          className="stage__overlay"
          aria-hidden="true"
          style={{ visibility: running ? "visible" : "hidden" }}
        />

        {!running && (
          <div className="stage__placeholder">
            {cameraError ? (
              <>
                <p style={{ fontWeight: 650, fontSize: "1.05rem" }}>
                  {cameraError.title}
                </p>
                <p style={{ color: "#c3cad8" }}>{cameraError.detail}</p>
                <Button variant="primary" onClick={onStart}>
                  Try again
                </Button>
              </>
            ) : (
              <>
                <p style={{ fontWeight: 650, fontSize: "1.05rem" }}>
                  Camera is off
                </p>
                <p style={{ color: "#c3cad8" }}>
                  Turn the camera on to spell words with hand signs. Nothing
                  leaves your machine - frames go to your own local server.
                </p>
                <Button
                  variant="primary"
                  onClick={onStart}
                  disabled={cameraStatus === "starting"}
                >
                  {cameraStatus === "starting" ? "Starting..." : "Start camera"}
                </Button>
              </>
            )}
          </div>
        )}

        {running && (
          <div className="stage__hud">
            <span className="stage__letter" aria-hidden="true">
              {frame.letter ?? "-"}
            </span>
            <div className="stage__meta">
              <span className="stage__status">
                <span
                  className="dot"
                  style={{
                    color: frame.handDetected ? "#6ee7a8" : "#f0a5a5",
                  }}
                  aria-hidden="true"
                />
                {STATUS_TEXT[frame.status]}
                {frame.handDetected && (
                  <span style={{ opacity: 0.75 }}>
                    · {confidencePercent}% sure
                    {/* Show the bar's own number too: if confidence is fine
                        but the bar is not moving, that is worth seeing. */}
                    {frame.progress > 0 &&
                      ` · ${Math.round(frame.progress * 100)}% held`}
                  </span>
                )}
              </span>
              <div
                className="hold-bar"
                role="progressbar"
                aria-label="Hold progress for the current letter"
                aria-valuenow={Math.round(frame.progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="hold-bar__fill"
                  style={{ width: `${Math.round(frame.progress * 100)}%` }}
                />
              </div>
            </div>
            <span className="stage__latency">{Math.round(frame.latencyMs)} ms</span>
          </div>
        )}

        {/* One live region carries recognition state to screen readers. */}
        <p className="sr-only" role="status" aria-live="polite">
          {running
            ? `${STATUS_TEXT[frame.status]}${
                frame.letter ? `. Current letter ${frame.letter}, ${confidencePercent} percent confident.` : ""
              }`
            : "Camera is off."}
        </p>
      </div>

      {connection === "failed" && (
        <Notice
          tone="danger"
          title="Not connected to the recognition server"
          detail="Start the backend with: uvicorn backend.app.main:app"
          action={
            <Button onClick={onRetryConnection}>Reconnect</Button>
          }
        />
      )}
      {connection === "reconnecting" && (
        <Notice
          tone="warn"
          title="Reconnecting to the recognition server..."
          detail="Recognition will resume automatically."
        />
      )}

      <div className="controls">
        {running ? (
          <Button onClick={onStop}>Stop camera</Button>
        ) : (
          <Button variant="primary" onClick={onStart} disabled={cameraStatus === "starting"}>
            Start camera
          </Button>
        )}
      </div>
    </div>
  );
}
