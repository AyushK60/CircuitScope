import type { FlowData, ModelMeta } from "../../api/client";
import { pointedness } from "../../lib/attention";

const STEPS = [
  "LN1 output",
  "Q, K, V projections",
  "scores: Q·Kᵀ / √d",
  "causal mask",
  "softmax",
  "weighted sum of V",
  "join heads, project",
  "added to the ribbon",
];

export default function AttentionLevel({
  meta,
  flow,
  layer,
  onHead,
  onBack,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  layer: number;
  onHead: (h: number) => void;
  onBack: () => void;
}) {
  const patterns = flow?.blocks[layer]?.attention ?? null;
  return (
    <div className="at">
      <header className="bk-head">
        <div>
          <p className="ex-kick">Inside block {layer}</p>
          <h2>Attention anatomy</h2>
          <p className="bk-sub">The {meta.n_heads} heads run side by side in one computation, not one after another.</p>
        </div>
        <div className="bk-nav">
          <button type="button" onClick={onBack}>
            ← Block {layer}
          </button>
        </div>
      </header>

      <ol className="at-pipe" aria-label="What every head does">
        {STEPS.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>

      <h3 className="at-h">Pick a head</h3>
      <div className="at-heads">
        {Array.from({ length: meta.n_heads }, (_, h) => {
          const p = patterns ? pointedness(patterns[h]) : null;
          return (
            <button
              key={h}
              type="button"
              className="at-head"
              aria-label={`Inspect head ${h}`}
              onClick={() => onHead(h)}
            >
              <b>Head {h}</b>
              <span className="at-meter">
                <span style={{ width: `${(p ?? 0) * 100}%` }} />
              </span>
              <small>{p != null ? `focus ${p.toFixed(2)}` : `${meta.d_head} dims`}</small>
            </button>
          );
        })}
      </div>
      <p className="bk-sub">
        {patterns
          ? "Focus is measured: how strongly each token's attention lands on one other token, averaged over the sentence. It describes the pattern, not what the head causes."
          : "Run the model to measure each head."}
      </p>
    </div>
  );
}
