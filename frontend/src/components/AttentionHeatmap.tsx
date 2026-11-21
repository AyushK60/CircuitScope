import { useMemo, useState } from "react";
import type { AttentionPattern } from "../api/client";

const CELL = 26;

/** Token-to-token attention heatmap for one head, from a real captured
 *  hook_pattern tensor. Values shown on hover; rows sum to 1.
 *  Clicking a query token selects it: that row is highlighted, other rows dim,
 *  and the measured top attended tokens are listed (causal mask visible as the
 *  empty upper triangle). */
export default function AttentionHeatmap({
  att,
  head,
  compareHead,
  onCompareHead,
}: {
  att: AttentionPattern | null;
  head: number;
  /** optional second head for a measured |A−B| comparison */
  compareHead?: number | null;
  onCompareHead?: (h: number) => void;
}) {
  const [hover, setHover] = useState<{ q: number; k: number } | null>(null);
  const [selectedQuery, setSelectedQuery] = useState<number | null>(null);

  const tokens = att?.tokens ?? [];
  const matrix = att?.pattern?.[head];

  const diffMatrix = useMemo(() => {
    if (compareHead == null || compareHead === head || !att) return null;
    const a = att.pattern[head];
    const b = att.pattern[compareHead];
    return a.map((row, q) => row.map((v, k) => Math.abs(v - b[q][k])));
  }, [compareHead, head, att]);

  const topAttended = useMemo(() => {
    if (selectedQuery == null || !matrix) return null;
    const row = matrix[selectedQuery];
    return row
      .map((v, k) => ({ v, k }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 4);
  }, [selectedQuery, matrix]);

  // sequential scale between the theme's --heat-lo and --heat-hi; value in [0,1] (rows sum to 1)
  const heat = (v: number) =>
    `color-mix(in srgb, var(--heat-hi) ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, var(--heat-lo))`;

  if (!att || !matrix) return <div className="skeleton" style={{ height: "18rem" }} />;
  const n = tokens.length;
  const width = 90 + n * CELL + 10;
  const height = 40 + n * CELL + 10;

  return (
    <div className="heatmap-wrap">
      <svg
        className="arch-svg"
        width={width}
        height={height}
        role="img"
        aria-label={`Attention pattern for head ${head}, tensor ${att.hook}, shape ${att.shape.join("x")}. Click a query token to highlight its attention row.`}
      >
        {tokens.map((t, k) => (
          <text
            key={`k${k}`}
            x={90 + k * CELL + CELL / 2}
            y={30}
            fill="var(--text-2)"
            fontSize={9}
            textAnchor="start"
            transform={`rotate(-55 ${90 + k * CELL + CELL / 2} 30)`}
          >
            {t}
          </text>
        ))}
        {tokens.map((t, q) => (
          <text
            key={`q${q}`}
            x={84}
            y={40 + q * CELL + CELL / 2 + 4}
            fill={selectedQuery === q ? "var(--accent)" : "var(--text-2)"}
            fontSize={10}
            textAnchor="end"
            style={{ cursor: "pointer" }}
            onClick={() => setSelectedQuery((s) => (s === q ? null : q))}
          >
            {t}
          </text>
        ))}
        {(diffMatrix ?? matrix).map((row, q) =>
          row.map((v, k) => {
            const dimmed = selectedQuery != null && selectedQuery !== q;
            const masked = k > q; // causally masked: no attention possible
            return (
              <rect
                key={`${q}-${k}`}
                x={90 + k * CELL}
                y={40 + q * CELL}
                width={CELL - 2}
                height={CELL - 2}
                fill={masked ? "var(--bg-2)" : undefined}
                style={masked ? undefined : { fill: heat(v) }}
                opacity={masked ? 0.5 : dimmed ? 0.25 : 1}
                stroke={selectedQuery === q && v > 0.001 ? "var(--brass)" : "transparent"}
                strokeWidth={selectedQuery === q ? 1 : 0}
                onMouseEnter={() => setHover({ q, k })}
                onMouseLeave={() => setHover(null)}
              />
            );
          }),
        )}
        {selectedQuery != null && (
          <rect
            x={88}
            y={40 + selectedQuery * CELL - 1}
            width={n * CELL + 4}
            height={CELL}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={1.5}
            pointerEvents="none"
          />
        )}
        {hover && (
          <text x={width - 8} y={16} fill="var(--text-0)" fontSize={11} textAnchor="end" fontFamily="monospace">
            q{hover.q}→k{hover.k}:{" "}
            {diffMatrix
              ? `|Δ| ${(diffMatrix[hover.q][hover.k] as number).toFixed(4)}`
              : (matrix[hover.q][hover.k] as number).toFixed(4)}
          </text>
        )}
      </svg>
      {selectedQuery != null && topAttended && (
        <div className="state-banner info" role="status">
          <span>
            <strong>{tokens[selectedQuery]}</strong> (query position {selectedQuery}) attends most to:{" "}
            {topAttended.map(({ v, k }) => (
              <span key={k} className="mono">
                {tokens[k]} {v.toFixed(3)}
                {k !== topAttended[topAttended.length - 1].k ? " · " : ""}
              </span>
            ))}
            <span className="inspector-meta"> — measured, head {head}; the empty upper triangle is the causal mask</span>
          </span>
          <button className="ghost" onClick={() => setSelectedQuery(null)} aria-label="Clear token selection">
            ✕
          </button>
        </div>
      )}
      <div className="legend-scale">
        <span>0</span>
        <div className="legend-bar" style={{ background: "linear-gradient(90deg, var(--heat-lo), var(--heat-hi))" }} />
        <span>attention weight 1.0 · rows sum to 1 · click a row token to inspect it · dark gray = causally masked (key after query)</span>
        {onCompareHead && (
          <label className="inspector-meta" style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
            compare with
            <select
              value={compareHead ?? head}
              onChange={(e) => onCompareHead(Number(e.target.value))}
              style={{ width: "auto", padding: "0.1rem 0.3rem" }}
            >
              {att.pattern.map((_, h) => (
                <option key={h} value={h}>
                  head {h}{h === head ? " (diff off)" : ""}
                </option>
              ))}
            </select>
          </label>
        )}
        <span className="mono" style={{ marginLeft: "auto" }}>
          {att.hook} · head {head}
          {diffMatrix ? ` |Δ| vs head ${compareHead}` : ""} · shape {att.shape.join("×")}
        </span>
      </div>
    </div>
  );
}
