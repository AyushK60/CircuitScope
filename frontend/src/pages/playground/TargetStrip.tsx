/** Mini architecture strip: where in the real 12-block pipeline this patch sits. */
export default function TargetStrip({ hook, position }: { hook: string; position: number }) {
  const match = hook.match(/blocks\.(\d+)\./);
  const layer = match ? Number(match[1]) : null;
  const kind = hook.includes("mlp") ? "MLP" : hook.includes("attn") ? "attention" : "residual stream";
  return (
    <div className="pg-strip">
      <svg viewBox="0 0 720 78" width="100%" role="img" aria-label={`Patch target: ${hook} at token position ${position}`}>
        <path d="M 16 22 L 704 22" stroke="var(--measure)" strokeDasharray="4 4" strokeWidth={2} fill="none" />
        <text x={16} y={13} fill="var(--measure)" fontSize={11}>
          residual stream
        </text>
        {Array.from({ length: 12 }, (_, l) => {
          const x = 20 + l * 57;
          const isTarget = layer === l;
          return (
            <g key={l}>
              <path d={`M ${x + 24} 22 L ${x + 24} 36`} stroke="var(--measure)" strokeDasharray="3 3" fill="none" opacity={0.6} />
              <rect
                x={x}
                y={36}
                width={48}
                height={28}
                rx={6}
                fill={isTarget ? "var(--bg-1)" : "var(--bg-3)"}
                stroke={isTarget ? "var(--accent)" : "var(--line-1)"}
                strokeWidth={isTarget ? 2 : 1}
              />
              <text x={x + 24} y={54} fill={isTarget ? "var(--accent)" : "var(--text-2)"} fontSize={11} textAnchor="middle">
                B{l}
              </text>
              {isTarget && (
                <text x={x + 24} y={76} fill="var(--accent)" fontSize={11} textAnchor="middle">
                  {kind} · patch here
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
