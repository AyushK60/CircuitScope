import { useMemo, useState } from "react";

/**
 * Diverging heatmap for patching-sweep grids (Recovery centred at 0).
 * Values come verbatim from the sweep API response — no invented numbers.
 * Hover/focus a cell for the exact Recovery and D_patched; "Show numbers"
 * toggles an accessible <table> of the same values.
 */

const CELL = 34;
const LABEL_W = 70;
const LABEL_H = 46;

export interface SweepHeatmapProps {
  rowLabels: string[];
  colLabels: string[];
  recovery: number[][];
  dPatched: number[][];
  /** what the values mean, shown under the legend (e.g. the patch direction) */
  direction?: string;
}

// diverging scale from the theme's --div-neg / --div-zero / --div-pos (negative = corrupted, positive = restored)
function mix(side: "pos" | "neg" | "zero", t: number): string {
  if (side === "zero") return "var(--div-zero)";
  return `color-mix(in srgb, var(--div-${side}) ${Math.round(Math.max(0, Math.min(1, t)) * 100)}%, var(--div-zero))`;
}

function fmt(v: number, digits: number): string {
  return Number.isFinite(v) ? v.toFixed(digits) : "n/a";
}

export default function SweepHeatmap({
  rowLabels,
  colLabels,
  recovery,
  dPatched,
  direction,
}: SweepHeatmapProps) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const [showNumbers, setShowNumbers] = useState(false);

  const maxAbs = useMemo(() => {
    let m = 0;
    for (const row of recovery) for (const v of row) m = Math.max(m, Math.abs(Number.isFinite(v) ? v : 0));
    return m > 0 ? m : 1;
  }, [recovery]);

  const color = (v: number) =>
    Number.isFinite(v) ? mix(v >= 0 ? "pos" : "neg", Math.min(1, Math.abs(v) / maxAbs)) : "var(--bg-3)";

  const nRows = rowLabels.length;
  const nCols = colLabels.length;
  const width = LABEL_W + nCols * CELL + 8;
  const height = LABEL_H + nRows * CELL + 8;
  const hovered = hover ? { recovery: recovery[hover.r]?.[hover.c], d: dPatched[hover.r]?.[hover.c] } : null;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", flexWrap: "wrap", marginBottom: "0.4rem" }}>
        <button className="ghost" aria-pressed={showNumbers} onClick={() => setShowNumbers((s) => !s)}>
          {showNumbers ? "Show heatmap" : "Show numbers"}
        </button>
        <div className="legend-scale" role="img" aria-label={`Diverging Recovery scale centred at 0, from ${fmt(-maxAbs, 2)} to ${fmt(maxAbs, 2)}`}>
          <span>{fmt(-maxAbs, 2)}</span>
          <span
            className="legend-bar"
            style={{
              background: `linear-gradient(90deg, ${mix("neg", 1)}, ${mix("neg", 0.5)}, ${mix("zero", 0)}, ${mix("pos", 0.5)}, ${mix("pos", 1)})`,
            }}
          />
          <span>0</span>
          <span>{fmt(maxAbs, 2)}</span>
          <span className="inspector-meta">Recovery (0 = no effect, 1 = fully restores clean)</span>
        </div>
        {direction && <span className="inspector-meta">direction: {direction}</span>}
      </div>

      {showNumbers ? (
        <div className="table-scroll">
          <table className="kv-table sweep-numbers">
            <caption className="inspector-meta" style={{ textAlign: "left", marginBottom: "0.3rem" }}>
              Recovery per cell (rows = {rowLabels[0]}…{rowLabels[nRows - 1]}, columns = positions/heads).
              Same values as the heatmap.
            </caption>
            <thead>
              <tr>
                <th scope="col">block ↓ / col →</th>
                  {colLabels.map((_, ci) => (
                    <th scope="col" key={ci}>{colLabels[ci]}</th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {rowLabels.map((rl, ri) => (
                <tr key={ri}>
                  <td>{rl}</td>
                  {colLabels.map((_, ci) => {
                    const v = recovery[ri]?.[ci];
                    return (
                      <td
                        key={ci}
                        style={{
                          textAlign: "right",
                          color: (v ?? 0) >= 0.5 ? "var(--div-pos)" : (v ?? 0) <= -0.5 ? "var(--div-neg)" : undefined,
                          fontWeight: Math.abs((v ?? 0) - 1) < 5e-4 ? 700 : 400,
                        }}
                      >
                        {fmt(v ?? NaN, 3)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="heatmap-wrap">
          <svg
            className="arch-svg"
            viewBox={`0 0 ${width} ${height}`}
            width={width}
            height={height}
            style={{ maxWidth: "100%", height: "auto" }}
            role="img"
            aria-label={`Patching sweep heatmap, ${nRows} rows (blocks) by ${nCols} columns. Recovery scale centred at 0.`}
          >
            {colLabels.map((t, ci) => (
              <text
                key={`c${ci}`}
                x={LABEL_W + ci * CELL + CELL / 2}
                y={LABEL_H - 10}
                fill="var(--text-2)"
                fontSize={9}
                textAnchor="start"
                transform={`rotate(-55 ${LABEL_W + ci * CELL + CELL / 2} ${LABEL_H - 10})`}
              >
                {t.length > 10 ? `${t.slice(0, 9)}…` : t}
              </text>
            ))}
            {rowLabels.map((t, ri) => (
              <text
                key={`r${ri}`}
                x={LABEL_W - 8}
                y={LABEL_H + ri * CELL + CELL / 2 + 3}
                fill="var(--text-2)"
                fontSize={9}
                textAnchor="end"
              >
                {t}
              </text>
            ))}
            {rowLabels.map((rl, ri) =>
              colLabels.map((cl, ci) => {
                const v = recovery[ri]?.[ci];
                return (
                  <rect
                    key={`${ri}-${ci}`}
                    tabIndex={0}
                    role="gridcell"
                    aria-label={`${rl} / ${cl}: Recovery ${fmt(v ?? NaN, 4)}, D_patched ${fmt(dPatched[ri]?.[ci] ?? NaN, 4)}`}
                    x={LABEL_W + ci * CELL + 1}
                    y={LABEL_H + ri * CELL + 1}
                    width={CELL - 2}
                    height={CELL - 2}
                    rx={3}
                    fill={color(v)}
                    stroke={hover && hover.r === ri && hover.c === ci ? "var(--text-0)" : "var(--line-0)"}
                    strokeWidth={hover && hover.r === ri && hover.c === ci ? 1.5 : 0.5}
                    onMouseEnter={() => setHover({ r: ri, c: ci })}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover({ r: ri, c: ci })}
                    onBlur={() => setHover(null)}
                  />
                );
              }),
            )}
          </svg>
        </div>
      )}

      {hovered && hover && (
        <div className="state-banner info" style={{ marginTop: "0.5rem" }} role="status">
          <span className="inspector-meta mono">
            <strong>
              {rowLabels[hover.r]} / {colLabels[hover.c]}
            </strong>
            {" — "}
            Recovery {fmt(hovered.recovery ?? NaN, 4)} · D_patched {fmt(hovered.d ?? NaN, 4)}
          </span>
        </div>
      )}
      <p className="inspector-meta" style={{ marginTop: "0.4rem" }}>
        Hover or keyboard-focus a cell (Tab) for the exact measured values.
      </p>
    </div>
  );
}
