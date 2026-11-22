import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import "./explorer.css";
import "./ribbon.css";
import "./sweeps.css";
import "./report.css";
import "./verify.css";
import "./study.css";
import "./learn.css";
import "./lab.css";
import "./playground.css";
import "./tour.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
