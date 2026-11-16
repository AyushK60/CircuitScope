/** Mini architecture strip: where in the real 12-block pipeline this patch sits. */
export default function TargetStrip({ hook, position, tokens }: { hook: string; position: number; tokens: string[] }) {
  const match = hook.match(/blocks\.(\d+)\./);
  const layer = match ? Number(match[1]) : null;
  const kind = hook.includes("mlp") ? "MLP" : hook.includes("attn") ? "attention" : "residual stream";
  return (
    <div className="target-strip">
      <svg viewBox="0 0 720 74" width="100%" role="img" aria-label={`Patch target: ${hook} at token position ${position}`}>
        {/* residual rail */}
        <path d="M 16 20 L 704 20" stroke="var(--violet)" strokeDasharray="4 4" strokeWidth={2} fill="none" />
        <text x={16} y={12} fill="var(--violet)" fontSize={9}>
          residual stream
        </text>
        {Array.from({ length: 12 }, (_, l) => {
          const x = 20 + l * 57;
          const isTarget = layer === l;
          return (
            <g key={l}>
              <path d={`M ${x + 24} 20 L ${x + 24} 34`} stroke="var(--violet)" strokeDasharray="3 3" fill="none" opacity={0.6} />
              <rect
                x={x}
                y={34}
                width={48}
                height={28}
                rx={5}
                fill={isTarget ? "var(--bg-1)" : "var(--bg-3)"}
                stroke={isTarget ? "var(--accent)" : "var(--line-1)"}
                strokeWidth={isTarget ? 2 : 1}
              />
              <text x={x + 24} y={52} fill={isTarget ? "var(--accent)" : "var(--text-2)"} fontSize={10} textAnchor="middle">
                B{l}
              </text>
              {isTarget && (
                <text x={x + 24} y={72} fill="var(--accent)" fontSize={8.5} textAnchor="middle">
                  {kind} · patch here
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {tokens.length > 0 && (
        <div className="token-row" aria-label="Patch position on the clean tokens">
          {tokens.map((t, i) => (
            <span key={i} className={`token${i === position ? " hl" : ""}`} title={`position ${i}`}>
              {t}
              <span className="idx">{i}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
