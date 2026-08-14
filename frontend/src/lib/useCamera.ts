/**
 * Webcam access and frame capture.
 *
 * Captured frames are mirrored and downscaled before they leave the browser:
 *
 * - **mirrored**, because the classifier was trained on mirrored frames (the
 *   original capture script ran `cv2.flip(frame, 1)`), and because the video
 *   is displayed mirrored too, so the overlay coordinates line up with no
 *   further maths;
 * - **downscaled to 320x240 JPEG**, because MediaPipe does not benefit from
 *   more and a smaller payload is a shorter round trip.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export const CAPTURE_WIDTH = 320;
export const CAPTURE_HEIGHT = 240;
const JPEG_QUALITY = 0.6;

export type CameraStatus =
  | "idle"
  | "starting"
  | "running"
  | "denied"
  | "missing"
  | "error";

export interface CameraError {
  status: Exclude<CameraStatus, "idle" | "starting" | "running">;
  title: string;
  detail: string;
}

function describe(error: unknown): CameraError {
  const name = (error as DOMException)?.name ?? "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      status: "denied",
      title: "Camera permission was declined",
      detail:
        "HandSign needs the camera to see your signs. Allow camera access in your browser's address bar, then press Start camera again.",
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return {
      status: "missing",
      title: "No camera found",
      detail:
        "Connect a webcam and press Start camera again. You can still use Text to Sign without a camera.",
    };
  }
  if (name === "NotReadableError") {
    return {
      status: "error",
      title: "The camera is busy",
      detail:
        "Another application is using the camera. Close it and press Start camera again.",
    };
  }
  return {
    status: "error",
    title: "The camera could not be started",
    detail:
      (error as Error)?.message ??
      "An unexpected problem stopped the camera. Please try again.",
  };
}

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<CameraError | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError({
        status: "error",
        title: "This browser cannot open a camera",
        detail:
          "Camera capture needs a secure context. Open the app over https, or on localhost.",
      });
      setStatus("error");
      return;
    }

    setStatus("starting");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus("running");
    } catch (caught) {
      const described = describe(caught);
      setError(described);
      setStatus(described.status);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  useEffect(() => stop, [stop]);

  /** Grab the current video frame as bare base64 JPEG, mirrored. */
  const capture = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return null;

    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.width = CAPTURE_WIDTH;
      canvas.height = CAPTURE_HEIGHT;
      canvasRef.current = canvas;
    }

    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return null;

    context.save();
    context.translate(CAPTURE_WIDTH, 0);
    context.scale(-1, 1); // mirror, to match how the model was trained
    context.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    context.restore();

    // Strip the "data:image/jpeg;base64," prefix: the backend accepts either,
    // and dropping it saves bytes on every single frame.
    return canvas.toDataURL("image/jpeg", JPEG_QUALITY).split(",")[1] ?? null;
  }, []);

  return { videoRef, status, error, start, stop, capture };
}
