/**
 * The single place the frontend knows a backend URL.
 *
 * In development Vite proxies /api, /signs and /ws to the FastAPI server, so
 * the base is empty and everything is same-origin. Set VITE_API_BASE when the
 * two are deployed separately.
 */

import type {
  AppConfig,
  SignCatalog,
  SignTranslation,
  Suggestions,
  TextState,
} from "./types";

export const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export const websocketUrl = (sessionId?: string): string => {
  const base =
    API_BASE ||
    `${window.location.protocol}//${window.location.host}`;
  const url = new URL("/ws/recognition", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  if (sessionId) url.searchParams.set("session_id", sessionId);
  return url.toString();
};

/** Resolve an asset path from the backend into something an <img> can load. */
export const assetUrl = (path: string): string =>
  path.startsWith("http") ? path : `${API_BASE}${path}`;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const REQUEST_TIMEOUT_MS = 8000;

async function request<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT_MS,
  );
  // Let the caller cancel too, without losing the timeout.
  signal?.addEventListener("abort", () => controller.abort(), { once: true });

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
      signal: controller.signal,
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const body = await response.json();
        if (typeof body?.detail === "string") detail = body.detail;
      } catch {
        /* the body was not JSON; the status is enough */
      }
      throw new ApiError(detail, response.status);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if ((error as Error)?.name === "AbortError") {
      throw new ApiError("The server did not respond in time.", 0);
    }
    throw new ApiError(
      "Could not reach the HandSign server. Is the backend running?",
      0,
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export const api = {
  config: () => request<AppConfig>("/api/config"),

  signCatalog: () => request<SignCatalog>("/api/signs"),

  translateToSign: (text: string, signal?: AbortSignal) =>
    request<SignTranslation>(
      "/api/translate-to-sign",
      { method: "POST", body: JSON.stringify({ text }) },
      signal,
    ),

  suggestions: (
    body: {
      text: string;
      current_word: string;
      context: string[];
      max_words?: number;
      max_sentences?: number;
    },
    options: { waitForLlm?: boolean; signal?: AbortSignal } = {},
  ) =>
    request<Suggestions>(
      `/api/suggestions?wait_for_llm=${options.waitForLlm ? "true" : "false"}`,
      { method: "POST", body: JSON.stringify(body) },
      options.signal,
    ),

  resetSession: (sessionId?: string) =>
    request<{ session_id: string; text: TextState }>("/api/session/reset", {
      method: "POST",
      body: JSON.stringify({ command: "clear", session_id: sessionId ?? null }),
    }),
};
