import { useEffect, useLayoutEffect, useState } from "react";
import { api, type Health } from "./api/client";
import Landing from "./pages/Landing";
import Explorer from "./pages/Explorer";
import Playground from "./pages/Playground";
import Learn from "./pages/Learn";
import NeuralLab from "./pages/NeuralLab";
import Sweeps from "./pages/Sweeps";
import Study from "./pages/Study";
import Report from "./pages/Report";
import Verify from "./pages/Verify";
import Tour from "./pages/Tour";
import SiteNav from "./components/SiteNav";
import { ROUTE_NAMES, type Route } from "./nav";

const JOURNAL_ROUTES: Route[] = ["landing", "explorer", "sweeps", "report", "verify", "study", "learn", "lab", "playground", "tour"];

function currentRoute(): Route {
  const h = window.location.hash.replace("#/", "").split("?")[0];
  return (ROUTE_NAMES.includes(h as Route) ? h : "landing") as Route;
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

  /* redesigned pages use the light "journal" theme; the others stay dark until their turn */
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = JOURNAL_ROUTES.includes(route) ? "journal" : "dark";
  }, [route]);

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
      <SiteNav route={route} navigate={navigate} status={modelState} onRefresh={refreshHealth} />

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
      {route === "study" && <Study health={health} />}
      {route === "report" && <Report health={health} />}
      {route === "verify" && <Verify health={health} />}
      {route === "tour" && <Tour health={health} />}

      <footer className="site-footer">
        CircuitScope — a local-first neural network observatory. GPT-2 Small runs entirely on this machine;
        every displayed value comes from a real inference or intervention.
      </footer>
    </>
  );
}
