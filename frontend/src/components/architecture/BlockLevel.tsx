import type { FlowData, ModelMeta } from "../../api/client";

/** Inside one block, in the order GPT-2 runs it (pre-LN): the ribbon is read by LN1 and
 *  LN2 but never replaced; attention and the MLP each add their result back to it. */
export default function BlockLevel({
  meta,
  flow,
  layer,
  onOpenAttention,
  onLayer,
  onBack,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  layer: number;
  onOpenAttention: () => void;
  onLayer: (l: number) => void;
  onBack: () => void;
}) {
  const norms = flow?.blocks[layer]?.resid_norm_per_token ?? null;
  const max = norms ? Math.max(...norms, 1e-9) : 1;
  const last = layer >= meta.n_layers - 1;

  return (
    <div className="bk">
      <header className="bk-head">
        <div>
          <p className="ex-kick">Inside one block</p>
          <h2>Block {layer}</h2>
          <p className="bk-sub">GPT-2 order, checked against the running model. Hooks: blocks.{layer}.*</p>
        </div>
        <div className="bk-nav">
          <button type="button" onClick={() => onLayer(layer - 1)} disabled={layer <= 0}>
            ← Block {layer - 1}
          </button>
          <button type="button" onClick={onBack}>
            All blocks
          </button>
          <button type="button" onClick={() => onLayer(layer + 1)} disabled={last}>
            Block {layer + 1} →
          </button>
        </div>
      </header>

      <ol className="bk-flow">
        <li className="bk-step">
          <div className="bk-card">
            <h3>Ribbon in</h3>
            <p>{meta.d_model} numbers for every token.</p>
            {norms && (
              <div className="bk-norms" role="img" aria-label="Measured size of each token's vector after this block">
                {norms.map((v, i) => (
                  <span key={i} style={{ height: `${Math.max(8, Math.sqrt(v / max) * 100)}%` }} title={`${flow?.tokens[i] ?? i}: ${v}`} />
                ))}
              </div>
            )}
            {norms && <small>Measured size of each token's vector after this block (square-root scale; the first token is usually far larger than the rest).</small>}
          </div>
        </li>
        <li className="bk-step tap">
          <div className="bk-card slim">
            <h3>LayerNorm 1</h3>
            <p>A normalized copy goes to attention. The ribbon itself is untouched.</p>
          </div>
        </li>
        <li className="bk-step">
          <button type="button" className="bk-card main" aria-label="Open attention anatomy" onClick={onOpenAttention}>
            <h3>Attention, {meta.n_heads} heads</h3>
            <p>
              Each head decides which earlier tokens to read ({meta.d_head} numbers each) and writes what it found.
            </p>
            <span className="go">Open attention anatomy →</span>
          </button>
        </li>
        <li className="bk-step add">
          <span className="bk-add" aria-hidden>+</span>
          <p>Attention's result is added to the ribbon.</p>
        </li>
        <li className="bk-step tap">
          <div className="bk-card slim">
            <h3>LayerNorm 2</h3>
            <p>A normalized copy goes to the MLP.</p>
          </div>
        </li>
        <li className="bk-step">
          <div className="bk-card main plain">
            <h3>
              MLP, {meta.d_model} → {meta.d_mlp} → {meta.d_model}
            </h3>
            <p>Each token is processed on its own, with a GELU in the middle. No looking at other tokens.</p>
          </div>
        </li>
        <li className="bk-step add">
          <span className="bk-add" aria-hidden>+</span>
          <p>The MLP's result is added to the ribbon.</p>
        </li>
        <li className="bk-step end">
          <p>{last ? "The ribbon continues to the final LayerNorm and the output." : `The ribbon continues into block ${layer + 1}.`}</p>
        </li>
      </ol>
    </div>
  );
}
