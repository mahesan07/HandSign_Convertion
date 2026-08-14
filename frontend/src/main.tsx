import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/global.css";

const container = document.getElementById("root");
if (!container) throw new Error("#root is missing from index.html");

// Deliberately not wrapped in <StrictMode>: its double-invoked effects would
// open the camera and the recognition websocket twice in development, which
// makes real latency and reconnection behaviour impossible to judge.
createRoot(container).render(<App />);
