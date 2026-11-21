/* Small graphs for the Report: each one draws values copied straight from the stored report data. */

export interface PushRow {
  label: string;
  value: number;
}

/** Two candidate probabilities as a pair of thin bars. */
export function MiniProb({ rows }: { rows: { label: string; value: number }[] }) {
  return (
    <div className="rp-mini rp-mini-prob" role="img" aria-label={rows.map((r) => `${r.label} ${(r.value * 100).toFixed(1)}%`).join(", ")}>
      {rows.map((r, i) => (
        <div key={r.label}>
          <span>{r.label}</span>
          <i>
            <b className={i === 0 ? "pos" : "neg"} style={{ width: `${Math.max(2, r.value * 100)}%` }} />
          </i>
          <em>{(r.value * 100).toFixed(1)}%</em>
        </div>
      ))}
    </div>
  );
}

/** One bar per block: the best single-site recovery found in that block. */
export function MiniBlocks({ values, highlight }: { values: number[]; highlight?: number }) {
  const max = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  return (
    <div
      className="rp-mini rp-mini-blocks"
      role="img"
      aria-label={`Best single-site recovery in each of ${values.length} blocks, highest ${max.toFixed(2)}`}
    >
      {values.map((v, i) => (
        <span key={i} title={`block ${i}: ${v.toFixed(2)}`}>
          <b className={i === highlight ? "hot" : ""} style={{ height: `${Math.max(6, (Math.max(0, v) / max) * 100)}%` }} />
        </span>
      ))}
    </div>
  );
}

/** A few components that push toward candidate A (right) or B (left). */
export function MiniPush({ rows }: { rows: PushRow[] }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  return (
    <div className="rp-mini rp-mini-push" role="img" aria-label={rows.map((r) => `${r.label} ${r.value.toFixed(2)}`).join(", ")}>
      {rows.map((r) => (
        <div key={r.label}>
          <span>{r.label}</span>
          <i>
            <b
              className={r.value >= 0 ? "pos" : "neg"}
              style={{ width: `${(Math.abs(r.value) / max) * 50}%`, [r.value >= 0 ? "left" : "right"]: "50%" }}
            />
          </i>
        </div>
      ))}
    </div>
  );
}

/** Full-size version of MiniPush for the figure under "Which parts of the model matter?". */
export function PushBars({ rows }: { rows: PushRow[] }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  return (
    <div className="rp-pushbars">
      {rows.map((r) => (
        <div key={r.label}>
          <span>head {r.label}</span>
          <i>
            <b
              className={r.value >= 0 ? "pos" : "neg"}
              style={{ width: `${(Math.abs(r.value) / max) * 50}%`, [r.value >= 0 ? "left" : "right"]: "50%" }}
            />
            <u />
          </i>
          <em>{r.value > 0 ? "+" : ""}{r.value.toFixed(2)}</em>
        </div>
      ))}
    </div>
  );
}
