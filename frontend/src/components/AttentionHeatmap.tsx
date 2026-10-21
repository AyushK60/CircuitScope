import { useMemo, useState } from "react";
import type { AttentionPattern } from "../api/client";

const CELL = 26;

/** Token-to-token attention heatmap for one head, from a real captured
 *  hook_pattern tensor. Values shown on hover; rows sum to 1. */
export default function AttentionHeatmap({ att, head }: { att: AttentionPattern | null; head: number }) {
  const [hover, setHover] = useState<{ q: number; k: number } | null>(null);

  const tokens = att?.tokens ?? [];
  const matrix = att?.pattern?.[head];

  const color = useMemo(() => {
    // sequential asagi→ruri scale; value in [0,1] (attention weights sum to 1 per row)
    return (v: number) => {
      const t = Math.max(0, Math.min(1, v));
      const r = Math.round(26 + t * 48);
      const g = Math.round(40 + t * 106);
      const b = Math.round(58 + t * 142);
      return `rgb(${r},${g},${b})`;
    };
  }, []);

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
        aria-label={`Attention pattern for head ${head}, layer pattern tensor ${att.hook}, shape ${att.shape.join("x")}`}
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
          <text key={`q${q}`} x={84} y={40 + q * CELL + CELL / 2 + 4} fill="var(--text-2)" fontSize={10} textAnchor="end">
            {t}
          </text>
        ))}
        {matrix.map((row, q) =>
          row.map((v, k) => (
            <rect
              key={`${q}-${k}`}
              x={90 + k * CELL}
              y={40 + q * CELL}
              width={CELL - 2}
              height={CELL - 2}
              fill={color(v)}
              stroke={hover && hover.q === q && hover.k === k ? "var(--text-0)" : "transparent"}
              onMouseEnter={() => setHover({ q, k })}
              onMouseLeave={() => setHover(null)}
            />
          )),
        )}
        {hover && (
          <text x={width - 8} y={16} fill="var(--text-0)" fontSize={11} textAnchor="end" fontFamily="monospace">
            q{hover.q}→k{hover.k}: {(matrix[hover.q][hover.k] as number).toFixed(4)}
          </text>
        )}
      </svg>
      <div className="legend-scale">
        <span>0</span>
        <div className="legend-bar" style={{ background: "linear-gradient(90deg, rgb(26,40,58), rgb(74,186,200))" }} />
        <span>attention weight 1.0 · rows sum to 1</span>
        <span className="mono" style={{ marginLeft: "auto" }}>
          {att.hook} · head {head} · shape {att.shape.join("×")}
        </span>
      </div>
    </div>
  );
}
