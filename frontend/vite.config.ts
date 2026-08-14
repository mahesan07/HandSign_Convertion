import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API base is injected at build time via VITE_API_BASE (see .env.example).
// In development we proxy instead, so the browser talks to one origin and
// there are no CORS or mixed-content surprises with the websocket.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/signs": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
