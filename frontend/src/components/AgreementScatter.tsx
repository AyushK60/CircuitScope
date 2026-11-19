import { useMemo, useState } from "react";

export interface ScatterPoint {
  label: string;
  kind: "head" | "mlp";
  layer: number;
  head: number | null;
  x: number;
  y: number;
}

/**
 * Accessible scatter plot: x = exact denoising recovery, y = a cheap method.
 * A dashed identity line marks perfect agreement. Colour is never the only
 * distinction: heads are circles, MLPs are squares, and the legend says so.
 * The exact numbers are always available in a toggled table.
 */
export default function AgreementScatter({
  points,
  title,
  xLabel,
  yLabel,
}: {
  points: ScatterPoint[];
  title: string;
  xLabel: string;
  yLabel: string;
}) {
  const [showTable, setShowTable] = useState(false);

  const bounds = useMemo(() => {
    const values = points.flatMap((p) => [p.x, p.y]).filter((v) => Number.isFinite(v));
    if (values.length === 0) return { min: 0, max: 1 };
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      min -= 0.5;
      max += 0.5;
    }
    const pad = (max - min) * 0.08;
    return { min: min - pad, max: max + pad };
  }, [points]);

  const W = 320;
  const H = 250;
  const PAD = 42;
  const sx = (v: number) => PAD + ((v - bounds.min) / (bounds.max - bounds.min)) * (W - PAD - 10);
  const sy = (v: number) => H - PAD - ((v - bounds.min) / (bounds.max - bounds.min)) * (H - PAD - 14);

  const ticks = useMemo(() => {
    const step = (bounds.max - bounds.min) / 4;
    return Array.from({ length: 5 }, (_, i) => bounds.min + i * step);
  }, [bounds]);

  if (points.length === 0) {
    return (
      <div className="state-banner info" role="status">
        No scatter data yet.
      </div>
    );
  }

  const heads = points.filter((p) => p.kind === "head");
  const mlps = points.filter((p) => p.kind === "mlp");

  return (
    <figure className="agreement-scatter">
      <figcaption>
        <strong>{title}</strong>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title}. ${heads.length} attention heads (circles) and ${mlps.length} MLPs (squares). Dashed line is perfect agreement.`}
      >
        <rect x={PAD} y={14} width={W - PAD - 10} height={H - PAD - 14} fill="var(--bg-1)" stroke="var(--line-1)" />
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PAD} x2={W - 10} y1={sy(t)} y2={sy(t)} stroke="var(--line-0)" />
            <line x1={sx(t)} x2={sx(t)} y1={14} y2={H - PAD} stroke="var(--line-0)" />
            <text x={PAD - 5} y={sy(t) + 3} fontSize={8} textAnchor="end" fill="var(--text-2)">
              {t.toFixed(2)}
            </text>
            <text x={sx(t)} y={H - PAD + 12} fontSize={8} textAnchor="middle" fill="var(--text-2)">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <line
          x1={sx(bounds.min)}
          y1={sy(bounds.min)}
          x2={sx(bounds.max)}
          y2={sy(bounds.max)}
          stroke="var(--brass)"
          strokeDasharray="5 4"
          strokeWidth={1.5}
        />
        {heads.map((p) => (
          <circle
            key={p.label}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={2.8}
            fill="var(--violet)"
            opacity={0.8}
            aria-hidden
          >
            <title>{`${p.label}: exact ${p.x.toFixed(4)}, method ${p.y.toFixed(4)}`}</title>
          </circle>
        ))}
        {mlps.map((p) => (
          <rect
            key={p.label}
            x={sx(p.x) - 2.6}
            y={sy(p.y) - 2.6}
            width={5.2}
            height={5.2}
            fill="var(--sage)"
            opacity={0.9}
            aria-hidden
          >
            <title>{`${p.label}: exact ${p.x.toFixed(4)}, method ${p.y.toFixed(4)}`}</title>
          </rect>
        ))}
        <text x={PAD + (W - PAD - 10) / 2} y={H - 4} fontSize={9} textAnchor="middle" fill="var(--text-1)">
          {xLabel}
        </text>
        <text
          x={10}
          y={14 + (H - PAD - 14) / 2}
          fontSize={9}
          textAnchor="middle"
          fill="var(--text-1)"
          transform={`rotate(-90 10 ${14 + (H - PAD - 14) / 2})`}
        >
          {yLabel}
        </text>
      </svg>
      <div className="scatter-legend inspector-meta">
        <span>○ attention head (144)</span>
        <span>□ MLP (12)</span>
        <span>--- perfect agreement</span>
      </div>
      <button className="ghost" onClick={() => setShowTable((v) => !v)} aria-expanded={showTable}>
        {showTable ? "Hide numbers table" : "Show numbers table"}
      </button>
      {showTable && (
        <div className="table-scroll">
          <table className="kv-table">
            <caption>Exact values and cheap-method values for every component.</caption>
            <thead>
              <tr>
                <th scope="col">Component</th>
                <th scope="col">Kind</th>
                <th scope="col">{xLabel}</th>
                <th scope="col">{yLabel}</th>
              </tr>
            </thead>
            <tbody>
              {points.map((p) => (
                <tr key={p.label}>
                  <td className="mono">{p.label}</td>
                  <td>{p.kind === "head" ? "head" : "MLP"}</td>
                  <td className="mono">{p.x.toFixed(4)}</td>
                  <td className="mono">{p.y.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  );
}
