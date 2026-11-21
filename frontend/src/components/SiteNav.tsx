import { useEffect, useState } from "react";
import { NAV_GROUPS, NAV_LABELS, TAB_LABELS, TAB_ROUTES, type Route } from "../nav";

/** Site header: logo and status on top, every page below it grouped by what a
 *  visitor is doing. On phones the second row becomes a bottom tab bar with a
 *  "More" sheet. */
export default function SiteNav({
  route,
  navigate,
  status,
  onRefresh,
}: {
  route: Route;
  navigate: (r: Route) => void;
  status: { cls: string; label: string };
  onRefresh: () => void;
}) {
  const [sheet, setSheet] = useState(false);

  useEffect(() => {
    if (!sheet) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSheet(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sheet]);

  const go = (r: Route) => {
    setSheet(false);
    navigate(r);
  };
  const link = (r: Exclude<Route, "landing">, cls = "") => {
    const label = cls === "tab" ? (TAB_LABELS[r] ?? NAV_LABELS[r]) : NAV_LABELS[r];
    return (
    <a
      key={r}
      href={`#/${r}`}
      className={`${cls}${route === r ? " active" : ""}`.trim()}
      aria-current={route === r ? "page" : undefined}
      onClick={(e) => {
        e.preventDefault();
        go(r);
      }}
    >
      {label}
    </a>
    );
  };
  const onMoreRoute = route !== "landing" && !TAB_ROUTES.includes(route);

  return (
    <>
      <header className="site-header">
        <div className="hd-top">
          <a
            className="wordmark"
            href="#/"
            onClick={(e) => {
              e.preventDefault();
              go("landing");
            }}
          >
            <span className="scope">⊙</span> CircuitScope
          </a>
          <span className="hd-sp" />
          <button className="ghost hd-refresh" onClick={onRefresh} title="Refresh backend status" aria-label="Refresh backend status">
            ⟳
          </button>
          <div className="model-status-chip" role="status">
            <span className={`status-dot ${status.cls}`} aria-hidden />
            <span className="chip-label">{status.label}</span>
          </div>
        </div>
        <nav className="hd-tier" aria-label="Main">
          {NAV_GROUPS.map((g) => (
            <div className="hd-group" role="group" aria-label={g.name} key={g.name}>
              <small>{g.name}</small>
              {g.routes.map((r) => link(r))}
            </div>
          ))}
        </nav>
      </header>

      <nav className="tabbar" aria-label="Quick links">
        {TAB_ROUTES.map((r) => link(r, "tab"))}
        <button
          type="button"
          className={`tab more${onMoreRoute || sheet ? " active" : ""}`}
          aria-expanded={sheet}
          aria-controls="more-sheet"
          onClick={() => setSheet((s) => !s)}
        >
          More
        </button>
      </nav>
      {sheet && (
        <>
          <div className="sheet-dim" onClick={() => setSheet(false)} aria-hidden />
          <div className="sheet" id="more-sheet" role="dialog" aria-label="All pages">
            {NAV_GROUPS.map((g) => (
              <div key={g.name}>
                <h4>{g.name}</h4>
                {g.routes.map((r) => link(r))}
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}
