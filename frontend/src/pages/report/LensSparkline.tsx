/** Lens sparkline (Stage 4 Task 6): the score gap D across lens stages,
 * drawn in SVG. Values come verbatim from the stored lens data; missing
 * values create gaps, never invented points. */
export interface SparkStage {
  stage: string;
  D: number | null;
}

export default function LensSparkline({ stages }: { stages: SparkStage[] }) {
  const W = 320;
  const H = 90;
  const PAD = 8;
  const valid = stages.filter((s) => s.D !== null) as { stage: string; D: number }[];
  if (valid.length < 2) {
    return (
      <p className="inspector-meta" role="status">
        Not enough lens stages with a score gap to draw a trace.
      </p>
    );
  }
  const xs = valid.map((_, i) => PAD + (i * (W - 2 * PAD)) / (valid.length - 1));
  const dMin = Math.min(...valid.map((v) => v.D));
  const dMax = Math.max(...valid.map((v) => v.D));
  const span = dMax - dMin || 1;
  const ys = valid.map((v) => H - PAD - ((v.D - dMin) / span) * (H - 2 * PAD));
  const points = xs.map((x, i) => `${x},${ys[i]}`).join(" ");
  const zeroY = H - PAD - ((0 - dMin) / span) * (H - 2 * PAD);
  return (
    <figure className="lens-sparkline">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label={`Score gap D across ${valid.length} lens stages, from ${dMin.toFixed(2)} to ${dMax.toFixed(2)}`}
      >
        {dMin < 0 && dMax > 0 && (
          <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY} stroke="var(--line-1)" strokeDasharray="3 3" />
        )}
        <polyline points={points} fill="none" stroke="var(--violet)" strokeWidth={2} />
        {xs.map((x, i) => (
          <circle key={i} cx={x} cy={ys[i]} r={2.5} fill="var(--violet)" />
        ))}
        <text x={PAD} y={H - 1} fill="var(--text-2)" fontSize={8}>
          embeddings
        </text>
        <text x={W - PAD} y={H - 1} fill="var(--text-2)" fontSize={8} textAnchor="end">
          block 11
        </text>
        <text x={PAD} y={10} fill="var(--text-2)" fontSize={8}>
          D: {dMin.toFixed(2)} … {dMax.toFixed(2)}
        </text>
      </svg>
      <figcaption className="inspector-meta">
        Logit-lens trace of the score gap (heuristic view; values from the stored report data).
      </figcaption>
    </figure>
  );
}
