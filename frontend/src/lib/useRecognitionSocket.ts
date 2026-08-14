/**
 * The live channel to the backend.
 *
 * Owns reconnection (capped exponential backoff), keeps the last recognition
 * frame, the authoritative text state and the current suggestions, and exposes
 * two ways to talk back: send a frame, or send an editing command.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { websocketUrl } from "./api";
import type {
  ConnectionState,
  RecognitionStatus,
  ServerMessage,
  Suggestions,
  TextCommand,
  TextState,
} from "./types";

const EMPTY_TEXT: TextState = { text: "", words: [], current_word: "" };
const EMPTY_SUGGESTIONS: Suggestions = {
  word_suggestions: [],
  sentence_suggestions: [],
  source: "local",
  llm_pending: false,
  notice: null,
};

const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000];

export interface RecognitionFrame {
  handDetected: boolean;
  status: RecognitionStatus;
  letter: string | null;
  confidence: number;
  progress: number;
  stableCount: number;
  requiredFrames: number;
  landmarks: [number, number][];
  latencyMs: number;
}

const IDLE_FRAME: RecognitionFrame = {
  handDetected: false,
  status: "idle",
  letter: null,
  confidence: 0,
  progress: 0,
  stableCount: 0,
  requiredFrames: 1,
  landmarks: [],
  latencyMs: 0,
};

export function useRecognitionSocket(enabled: boolean) {
  const [connection, setConnection] = useState<ConnectionState>("closed");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [frame, setFrame] = useState<RecognitionFrame>(IDLE_FRAME);
  const [text, setText] = useState<TextState>(EMPTY_TEXT);
  const [suggestions, setSuggestions] = useState<Suggestions>(EMPTY_SUGGESTIONS);
  const [lastLetter, setLastLetter] = useState<{ letter: string; at: number } | null>(
    null,
  );
  const [socketError, setSocketError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const sessionRef = useRef<string | null>(null);
  const attemptRef = useRef(0);
  const retryTimerRef = useRef<number | null>(null);
  const closingRef = useRef(false);

  const handleMessage = useCallback((message: ServerMessage) => {
    switch (message.type) {
      case "ready":
        sessionRef.current = message.session_id;
        setSessionId(message.session_id);
        setSocketError(null);
        break;
      case "recognition":
        setFrame({
          handDetected: message.hand_detected,
          status: message.status,
          letter: message.prediction.letter,
          confidence: message.prediction.confidence,
          progress: message.progress,
          stableCount: message.stable_count,
          requiredFrames: message.required_frames,
          landmarks: message.landmarks,
          latencyMs: message.latency_ms,
        });
        setText(message.text);
        if (message.committed_letter) {
          setLastLetter({ letter: message.committed_letter, at: Date.now() });
        }
        break;
      case "suggestions":
        setSuggestions(message.suggestions);
        setText(message.text);
        break;
      case "text":
        setText(message.text);
        break;
      case "error":
        // Per-frame problems are noise; only surface something the user can act on.
        if (message.fatal || message.code !== "bad_frame") {
          setSocketError(message.message);
        }
        break;
      case "pong":
        break;
    }
  }, []);

  const connect = useCallback(() => {
    if (closingRef.current) return;
    setConnection(attemptRef.current === 0 ? "connecting" : "reconnecting");

    const socket = new WebSocket(websocketUrl(sessionRef.current ?? undefined));
    socketRef.current = socket;

    socket.onopen = () => {
      attemptRef.current = 0;
      setConnection("open");
      setSocketError(null);
    };

    socket.onmessage = (event) => {
      try {
        handleMessage(JSON.parse(event.data) as ServerMessage);
      } catch {
        /* a malformed frame must never break the stream */
      }
    };

    socket.onerror = () => {
      /* onclose always follows; retrying is handled there */
    };

    socket.onclose = () => {
      socketRef.current = null;
      if (closingRef.current) {
        setConnection("closed");
        return;
      }
      const attempt = attemptRef.current;
      if (attempt >= RECONNECT_DELAYS_MS.length) {
        setConnection("failed");
        setSocketError(
          "Lost contact with the HandSign server. Check that the backend is running, then retry.",
        );
        return;
      }
      attemptRef.current = attempt + 1;
      setConnection("reconnecting");
      retryTimerRef.current = window.setTimeout(
        connect,
        RECONNECT_DELAYS_MS[attempt],
      );
    };
  }, [handleMessage]);

  useEffect(() => {
    if (!enabled) return;
    closingRef.current = false;
    attemptRef.current = 0;
    connect();

    return () => {
      closingRef.current = true;
      if (retryTimerRef.current) window.clearTimeout(retryTimerRef.current);
      socketRef.current?.close();
      socketRef.current = null;
      setConnection("closed");
    };
  }, [enabled, connect]);

  const send = useCallback((payload: object): boolean => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(payload));
    return true;
  }, []);

  const sendFrame = useCallback(
    (image: string) => send({ type: "frame", image, mirrored: true }),
    [send],
  );

  const sendCommand = useCallback(
    (command: TextCommand, value = "") =>
      send({ type: "command", command, value }),
    [send],
  );

  const retry = useCallback(() => {
    attemptRef.current = 0;
    setSocketError(null);
    if (retryTimerRef.current) window.clearTimeout(retryTimerRef.current);
    connect();
  }, [connect]);

  return {
    connection,
    sessionId,
    frame,
    text,
    suggestions,
    lastLetter,
    socketError,
    sendFrame,
    sendCommand,
    retry,
  };
}
