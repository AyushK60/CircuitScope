import type { SavedExperiment } from "../../api/client";

/** "Experiment history" panel: persisted completed experiments, selection for
 *  comparison, load, delete. */
export default function HistoryPanel({
  history,
  compareIds,
  toggleCompare,
  delConfirm,
  setDelConfirm,
  loadSaved,
  deleteSaved,
  compareData,
  runCompare,
}: {
  history: SavedExperiment[];
  compareIds: string[];
  toggleCompare: (id: string) => void;
  delConfirm: string | null;
  setDelConfirm: (id: string | null) => void;
  loadSaved: (id: string) => void;
  deleteSaved: (id: string) => void;
  compareData: { id: string; delta: number | null; recovery: number | null }[];
  runCompare: () => void;
}) {
  return (
    <div className="pg-card">
      <h2>Experiment history</h2>
      <p className="inspector-meta" style={{ marginBottom: "0.8rem" }}>
        Completed experiments persisted in the local database. Loading one restores its exact stored
        results (config, metrics, provenance).
      </p>
      {history.length === 0 ? (
        <p className="inspector-meta">No completed experiments yet — run one in the Patch tab.</p>
      ) : (
        <div className="table-scroll">
        <table className="kv-table">
          <tbody>
            {history.map((h) => (
              <tr key={h.id}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Select experiment ${h.id} for comparison`}
                    checked={compareIds.includes(h.id)}
                    onChange={() => toggleCompare(h.id)}
                  />
                </td>
                <td>{new Date(h.created_at * 1000).toLocaleTimeString()}</td>
                <td>
                  {(h.config as Record<string, unknown>).hook
                    ? `intervention · ${String((h.config as Record<string, unknown>).hook)}`
                    : "candidate comparison"}
                </td>
                <td>
                  pos {JSON.stringify((h.config as Record<string, unknown>).positions ?? [])}
                </td>
                <td>
                  <button onClick={() => loadSaved(h.id)}>Load</button>{" "}
                  {delConfirm === h.id ? (
                    <>
                      <button className="danger" onClick={() => deleteSaved(h.id)}>
                        Confirm delete
                      </button>
                      <button className="ghost" onClick={() => setDelConfirm(null)}>
                        ✕
                      </button>
                    </>
                  ) : (
                    <button className="danger" onClick={() => setDelConfirm(h.id)}>
                      Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
      {compareIds.length === 2 && (
        <div style={{ marginTop: "0.8rem", display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" }}>
          <button onClick={runCompare}>Compare selected (Δ / Recovery)</button>
          {compareData.map((c) => (
            <span key={c.id} className="inspector-meta mono">
              {c.id.slice(0, 6)}: Δ {c.delta?.toFixed(3) ?? "n/a"} · rec {c.recovery?.toFixed(3) ?? "n/a"}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
