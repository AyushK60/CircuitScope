import { useEffect, useState } from "react";
import { api, type Health } from "./api/client";
import Landing from "./pages/Landing";
import Explorer from "./pages/Explorer";
import Playground from "./pages/Playground";
import Learn from "./pages/Learn";
import NeuralLab from "./pages/NeuralLab";
import Sweeps from "./pages/Sweeps";

type Route = "landing" | "explorer" | "playground" | "learn" | "lab" | "sweeps";

function currentRoute(): Route {
  const h = window.location.hash.replace("#/", "").split("?")[0];
  return (["explorer", "playground", "learn", "lab", "sweeps"].includes(h) ? h : "landing") as Route;
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute());
  const [health, setHealth] = useState<Health | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(currentRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refreshHealth = () => {
    setBackendError(null);
    api
      .health()
      .then(setHealth)
      .catch((e) => {
        setHealth(null);
        setBackendError(String(e.message ?? e));
      });
  };

  useEffect(() => {
    refreshHealth();
    const t = setInterval(refreshHealth, 10000);
    return () => clearInterval(t);
  }, []);

  const navigate = (r: Route) => {
    window.location.hash = r === "landing" ? "/" : `/${r}`;
    setRoute(r);
  };

  const modelState = backendError
    ? { cls: "err", label: "Backend offline" }
    : health?.model_loaded
      ? { cls: "ok", label: `GPT-2 ready · ${health.model?.device}` }
      : { cls: "busy", label: "Model loading…" };

  return (
    <>
      <header className="site-header">
        <a
          className="wordmark"
          href="#/"
          onClick={(e) => {
            e.preventDefault();
            navigate("landing");
          }}
        >
          <span className="scope">⊙</span> CircuitScope
        </a>
        <nav className="nav" aria-label="Main">
          {(["explorer", "lab", "playground", "learn", "sweeps"] as Route[]).map((r) => (
            <a
              key={r}
              href={`#/${r}`}
              className={route === r ? "active" : ""}
              onClick={(e) => {
                e.preventDefault();
                navigate(r);
              }}
            >
              {r === "explorer" ? "Model Explorer" : r === "playground" ? "Intervention Playground" : r === "lab" ? "NN Lab" : r === "sweeps" ? "Sweeps" : "Learn"}
            </a>
          ))}
        </nav>
        <button className="ghost" onClick={refreshHealth} title="Refresh backend status">
          ⟳
        </button>
        <div className="model-status-chip" role="status">
          <span className={`status-dot ${modelState.cls}`} aria-hidden />
          {modelState.label}
        </div>
      </header>

      {backendError && (
        <div className="page" style={{ paddingTop: "1rem" }}>
          <div className="state-banner error" role="alert">
            <strong>Backend unavailable.</strong> Start it with
            <code className="mono"> .venv\Scripts\python -m backend.app.main </code>
            from the project root (or run <code className="mono">start.bat</code>), then refresh. ({backendError})
          </div>
        </div>
      )}

      {route === "landing" && <Landing health={health} navigate={navigate} />}
      {route === "explorer" && <Explorer health={health} />}
      {route === "playground" && <Playground health={health} />}
      {route === "learn" && <Learn />}
      {route === "lab" && <NeuralLab />}
      {route === "sweeps" && <Sweeps health={health} />}

      <footer className="site-footer">
        CircuitScope — a local-first neural network observatory. GPT-2 Small runs entirely on this machine;
        every displayed value comes from a real inference or intervention.
      </footer>
    </>
  );
}
