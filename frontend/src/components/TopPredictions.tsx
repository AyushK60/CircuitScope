import type { Candidate } from "../api/client";

/** Horizontal bar chart of next-token candidates. Bars encode measured logits
 *  (signed scale) or probabilities (0–1), never invented values. */
export default function TopPredictions({
  candidates,
  metric = "logit",
  maxRows = 8,
  highlight = [],
}: {
  candidates: Candidate[] | null;
  metric?: "logit" | "prob";
  maxRows?: number;
  highlight?: string[];
}) {
  if (!candidates) return <div className="skeleton" style={{ height: "10rem" }} />;
  const rows = candidates.slice(0, maxRows);
  if (metric === "prob") {
    const max = Math.max(...rows.map((r) => r.prob), 1e-9);
    return (
      <div>
        {rows.map((c) => (
          <div className="bar-row" key={c.token_id}>
            <span className="label">{c.token}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ left: 0, width: `${(c.prob / max) * 100}%`, background: "var(--accent)" }}
              />
            </div>
            <span className="value">{(c.prob * 100).toFixed(2)}%</span>
          </div>
        ))}
      </div>
    );
  }
  // signed logit scale anchored at the row set's minimum
  const vals = rows.map((r) => r.logit);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = Math.max(max - min, 1e-9);
  const zeroPct = ((0 - min) / span) * 100;
  return (
    <div>
      {rows.map((c) => {
        const v = c.logit;
        const from = Math.min(zeroPct, ((v - min) / span) * 100);
        const width = (Math.abs(v) / span) * 100;
        return (
          <div className="bar-row" key={c.token_id}>
            <span className="label" style={highlight.includes(c.token) ? { color: "var(--accent)" } : undefined}>
              {c.token}
            </span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  left: `${from}%`,
                  width: `${width}%`,
                  background: v >= 0 ? "var(--accent)" : "var(--danger)",
                }}
              />
            </div>
            <span className="value">{v.toFixed(2)}</span>
          </div>
        );
      })}
      <div className="legend-scale">
        <span>logit scale</span>
        <div
          className="legend-bar"
          style={{ background: "linear-gradient(90deg, var(--danger) 0 33%, var(--bg-2) 33% 66%, var(--accent) 66% 100%)" }}
        />
        <span>negative ← 0 → positive</span>
      </div>
    </div>
  );
}
