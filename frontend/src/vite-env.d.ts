/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin, e.g. http://127.0.0.1:8000. Empty means same-origin. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
