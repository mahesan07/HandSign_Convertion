/** Shapes mirrored from the backend's Pydantic schemas. */

export type RecognitionStatus =
  | "idle"
  | "low_confidence"
  | "detecting"
  | "locking"
  | "committed"
  | "hold_release"
  | "cooldown";

export interface TextState {
  text: string;
  words: string[];
  current_word: string;
}

export interface Prediction {
  letter: string | null;
  confidence: number;
  alternatives: [string, number][];
}

export interface Suggestions {
  word_suggestions: string[];
  sentence_suggestions: string[];
  source: "local" | "gemini" | "cache";
  llm_pending: boolean;
  notice: string | null;
}

export interface AppConfig {
  app_name: string;
  version: string;
  classes: string[];
  gemini_enabled: boolean;
  gemini_model: string | null;
  min_confidence: number;
  stable_frames: number;
  commit_cooldown_ms: number;
  duplicate_suppression_ms: number;
  release_frames: number;
  max_word_suggestions: number;
  max_sentence_suggestions: number;
  suggestion_debounce_ms: number;
  recommended_fps: number;
}

export interface SignToken {
  character: string;
  kind: "letter" | "space" | "digit" | "punctuation" | "unsupported";
  asset: string | null;
  label: string;
}

export interface SignWord {
  text: string;
  signs: SignToken[];
}

export interface SignTranslation {
  text: string;
  words: SignWord[];
  unsupported: string[];
  sign_count: number;
}

export interface SignCatalog {
  asset_base: string;
  signs: Record<string, string>;
}

/** Messages the backend pushes down the websocket. */
export type ServerMessage =
  | {
      type: "ready";
      session_id: string;
      classes: string[];
      gemini_enabled: boolean;
    }
  | {
      type: "recognition";
      session_id: string;
      hand_detected: boolean;
      status: RecognitionStatus;
      prediction: Prediction;
      progress: number;
      stable_count: number;
      required_frames: number;
      committed_letter: string | null;
      landmarks: [number, number][];
      text: TextState;
      latency_ms: number;
    }
  | {
      type: "suggestions";
      session_id: string;
      suggestions: Suggestions;
      text: TextState;
    }
  | { type: "text"; session_id: string; text: TextState }
  | { type: "error"; code: string; message: string; fatal: boolean }
  | { type: "pong" };

export type TextCommand =
  | "space"
  | "backspace"
  | "delete_word"
  | "clear"
  | "add_character"
  | "accept_word"
  | "accept_sentence"
  | "set_text";

export type ConnectionState =
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "failed";
