import type { FlowData } from "../../api/client";
import { hoverFromEvent, type HoverFn } from "./hover";

export type { HoverFn };

export function StageBand({ x, y, w, title, sub }: { x: number; y: number; w: number; title: string; sub: string }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={36} rx={7} fill="var(--bg-3)" stroke="var(--line-1)" />
      <text x={x + 12} y={y + 15} fill="var(--text-0)" fontSize={11.5}>
        {title}
      </text>
      <text x={x + 12} y={y + 28} fill="var(--text-2)" fontSize={9}>
        {sub}
      </text>
    </g>
  );
}

export function TokenStrip({ flow, x, y, markedPosition }: { flow: FlowData | null; x: number; y: number; markedPosition?: number }) {
  const tokens = flow?.tokens ?? [];
  if (tokens.length === 0) {
    return (
      <text x={x} y={y + 14} fill="var(--text-2)" fontSize={11}>
        run the model to see your text tokenized here
      </text>
    );
  }
  // measured tokens; wrap into rows of 12
  const perRow = 11;
  return (
    <g>
      {tokens.map((t, i) => {
        const row = Math.floor(i / perRow);
        const col = i % perRow;
        const marked = markedPosition === i;
        return (
          <g key={i}>
            <rect x={x + col * 50} y={y + row * 22} width={46} height={18} rx={3} fill={marked ? "#3a2c1c" : "var(--bg-2)"} stroke={marked ? "var(--brass)" : "var(--line-0)"} />
            <text x={x + col * 50 + 23} y={y + row * 22 + 13} fill="var(--text-1)" fontSize={8.5} textAnchor="middle">
              {t.length > 8 ? t.slice(0, 8) : t}
            </text>
          </g>
        );
      })}
      <text x={x + 11 * 50 + 8} y={y + 14} fill="var(--brass)" fontSize={8} fontFamily="var(--font-mono)">
        MEASURED
      </text>
    </g>
  );
}

export function LogitsBand({ x, y, w, flow }: { x: number; y: number; w: number; flow: FlowData | null }) {
  const top = flow?.top_predictions.slice(0, 5) ?? null;
  return (
    <g>
      <rect x={x} y={y} width={w} height={110} rx={9} fill="var(--bg-2)" stroke="var(--line-1)" />
      {top ? (
        <>
          <text x={x + 14} y={y + 20} fill="var(--text-2)" fontSize={10}>
            measured probabilities at the final position
          </text>
          {top.map((c, i) => {
            const bw = ((w - 160) * c.prob) / top[0].prob;
            return (
              <g key={c.token_id}>
                <text x={x + 14} y={y + 44 + i * 15} fill="var(--text-1)" fontSize={10} style={{ whiteSpace: "pre" }}>
                  {c.token.trim() || "␣"}
                </text>
                <rect x={x + 80} y={y + 36 + i * 15} width={Math.max(bw, 3)} height={9} rx={2} fill="var(--accent)" opacity={0.9} />
                <text x={x + w - 12} y={y + 44 + i * 15} fill="var(--text-2)" fontSize={9.5} textAnchor="end" fontFamily="var(--font-mono)">
                  {(c.prob * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
          <text x={x + w - 12} y={y + 104} fill="var(--brass)" fontSize={8} textAnchor="end" fontFamily="var(--font-mono)">
            MEASURED — softmax of real logits
          </text>
        </>
      ) : (
        <text x={x + 14} y={y + 30} fill="var(--text-2)" fontSize={11}>
          {`one score per vocabulary token (${(50257).toLocaleString()})`}
        </text>
      )}
    </g>
  );
}

export function AddNode({ x, y, setHover, clear, note, hook }: { x: number; y: number; setHover: HoverFn; clear: () => void; note: string; hook: string }) {
  return (
    <g
      className="arch-node"
      onMouseEnter={(e) => setHover(hoverFromEvent(e, "Residual addition", [note, `this is what the violet rail carries · ${hook}`]))}
      onMouseLeave={clear}
      style={{ cursor: "help" }}
    >
      <circle cx={x} cy={y} r={13} fill="var(--bg-3)" stroke="var(--violet)" />
      <text x={x} y={y + 5} fill="var(--violet)" fontSize={15} textAnchor="middle">
        ⊕
      </text>
    </g>
  );
}
