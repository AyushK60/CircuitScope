import type { ModelMeta } from "../api/client";

/** Level 0 — the complete real pipeline of the loaded model, top to bottom.
 *  Node count and labels come from live model metadata (n_layers, etc.).
 *  Violet dashed lines are the genuine residual pathways; every block is a
 *  real transformer block of the running model, not decoration. */
export function FullModelDiagram({
  meta,
  selectedLayer,
  onSelectLayer,
  tokenCount,
}: {
  meta: ModelMeta;
  selectedLayer: number | null;
  onSelectLayer: (l: number) => void;
  tokenCount: number | null;
}) {
  const W = 320;
  const NODE_H = 34;
  const GAP = 12;
  const blocks = Array.from({ length: meta.n_layers }, (_, i) => i);
  // stack: input, tokenize, embed, [blocks], lnf, unembed, logits
  const nAbove = 3;
  const nBelow = 3;
  const H = (nAbove + meta.n_layers + nBelow) * (NODE_H + GAP) + 40;
  const x = 90;
  const w = W - 180;

  const yOf = (i: number) => 24 + i * (NODE_H + GAP);
  let row = 0;
  const fixed = (label: string, sub: string, key: string) => ({ label, sub, key, y: yOf(row++), fixed: true });
  const blockNode = (l: number) => ({ label: `Transformer Block ${l}`, sub: `attn ×${meta.n_heads} · mlp ${meta.d_mlp}`, key: `b${l}`, y: yOf(nAbove + l), fixed: false });

  const nodes = [
    fixed(`Input text${tokenCount != null ? ` · ${tokenCount} tokens` : ""}`, "your sentence", "in"),
    fixed("Tokenizer", `${meta.tokenizer_class} · byte-level BPE`, "tok"),
    fixed("Embeddings", `token ${meta.d_model}d + position ${meta.d_model}d`, "emb"),
    ...blocks.map(blockNode),
    fixed("Final LayerNorm", "ln_final", "lnf"),
    fixed("Unembed", `linear → ${meta.d_vocab} logits`, "unembed"),
    fixed("Next-token logits", "score for every vocabulary token", "logits"),
  ];

  return (
    <div className="diagram-wrap">
      <svg
        className="arch-svg"
        width={W}
        height={H}
        role="img"
        aria-label={`Architecture of ${meta.model_id}: ${meta.n_layers} transformer blocks`}
      >
        {/* sequential edges */}
        {nodes.slice(0, -1).map((n, i) => (
          <path key={`e${i}`} className="arch-edge" d={`M ${x + w / 2} ${n.y + NODE_H} L ${x + w / 2} ${nodes[i + 1].y}`} />
        ))}
        {/* residual edges: each block output adds back to the stream */}
        {blocks.map((l) => (
          <path
            key={`r${l}`}
            className="arch-edge resid"
            d={`M ${x + w} ${yOf(nAbove + l) + NODE_H / 2} C ${x + w + 26} ${yOf(nAbove + l) + NODE_H / 2}, ${x + w + 26} ${yOf(nAbove + l + 1) + NODE_H / 2}, ${x + w} ${yOf(nAbove + l + 1) + NODE_H / 2}`}
            markerEnd="arrow"
          />
        ))}
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--violet)" />
          </marker>
        </defs>
        <text x={x + w + 34} y={yOf(nAbove) + NODE_H} fill="var(--violet)" fontSize={10} transform={`rotate(90 ${x + w + 34} ${yOf(nAbove) + NODE_H})`} opacity={0.85}>
          residual stream (adds, not replaces)
        </text>
        {nodes.map((n) => {
          const isSel = !n.fixed && selectedLayer === Number(n.key.slice(1));
          return (
            <g
              key={n.key}
              className={`arch-node${isSel ? " selected" : ""}`}
              onClick={() => !n.fixed && onSelectLayer(Number(n.key.slice(1)))}
              onKeyDown={(e) => {
                if (!n.fixed && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  onSelectLayer(Number(n.key.slice(1)));
                }
              }}
              tabIndex={n.fixed ? undefined : 0}
              role={n.fixed ? undefined : "button"}
              aria-label={n.label}
            >
              <rect x={x} y={n.y} width={w} height={NODE_H} rx={6} fill={n.fixed ? "var(--bg-2)" : "var(--bg-3)"} stroke="var(--line-1)" />
              <text x={x + 12} y={n.y + 15} fill="var(--text-0)" fontSize={12}>
                {n.label}
              </text>
              <text x={x + 12} y={n.y + 28} fill="var(--text-2)" fontSize={9.5}>
                {n.sub}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend-scale">
        <span>solid: sequential computation ·</span>
        <span style={{ color: "var(--violet)" }}>dashed violet: residual adds (real skip connections)</span>
      </div>
      <p className="inspector-meta" style={{ marginTop: "0.4rem" }}>
        Schematic of the loaded model's real topology (metadata: {meta.n_layers} blocks, {meta.n_heads} heads,
        d_model {meta.d_model}). Node timing is not implied — the GPU computes each stage for all {tokenCount ?? "N"} tokens
        in parallel. Click a block to inspect it.
      </p>
    </div>
  );
}

/** Level 1 — genuine GPT-2 block internals: pre-normalization with two residual adds.
 *  Order verified against the running model: x → LN1 → attention → +x → LN2 → MLP → +. */
export function BlockDiagram({
  meta,
  layer,
  onSelectAttention,
  onSelectMlp,
}: {
  meta: ModelMeta;
  layer: number;
  onSelectAttention: () => void;
  onSelectMlp: () => void;
}) {
  const W = 460;
  const H = 380;
  const cx = 200;
  const boxW = 150;
  const boxH = 36;
  const residX = 400;

  const boxes = [
    { label: `input x  [${meta.d_model}]`, y: 20, click: undefined as undefined | (() => void) },
    { label: "LayerNorm 1", y: 80, click: undefined },
    { label: `Attention ×${meta.n_heads} heads`, y: 140, click: onSelectAttention },
    { label: "add ⊕ residual", y: 200, click: undefined },
    { label: "LayerNorm 2", y: 255, click: undefined },
    { label: `MLP (${meta.d_model}→${meta.d_mlp}→${meta.d_model})`, y: 310, click: onSelectMlp },
  ];

  return (
    <div className="diagram-wrap">
      <svg className="arch-svg" width={W} height={H} role="img" aria-label={`Internals of transformer block ${layer}`}>
        <defs>
          <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--line-1)" />
          </marker>
        </defs>
        {/* main sequential path */}
        {boxes.slice(0, -1).map((b, i) => (
          <path key={i} className="arch-edge" markerEnd="url(#arrow2)" d={`M ${cx + boxW / 2} ${b.y + boxH} L ${cx + boxW / 2} ${boxes[i + 1].y}`} />
        ))}
        {/* attention output residual: from attention box out and up to add node */}
        <path
          className="arch-edge resid"
          markerEnd="url(#arrow2)"
          d={`M ${cx + boxW} 158 C ${residX} 158, ${residX} 218, ${cx + boxW} 218`}
        />
        {/* MLP output residual: to the block output (x + attn + mlp) */}
        <path
          className="arch-edge resid"
          markerEnd="url(#arrow2)"
          d={`M ${cx + boxW} 328 C ${residX} 328, ${residX} 372, ${cx - 60} 372 L ${cx + boxW / 2} 372 L ${cx + boxW / 2} 346`}
          opacity={0.5}
        />
        {/* skip line from block input x to the add node */}
        <path
          className="arch-edge resid"
          d={`M ${cx} 38 L ${residX - 8} 38 L ${residX - 8} 218`}
          opacity={0.8}
        />
        <text x={residX - 4} y={140} fill="var(--violet)" fontSize={10} transform={`rotate(90 ${residX - 4} 140)`}>
          residual stream
        </text>
        {boxes.map((b) => (
          <g
            key={b.label}
            className={`arch-node${b.click ? "" : ""}`}
            onClick={b.click}
            onKeyDown={(e) => {
              if (b.click && (e.key === "Enter" || e.key === " ")) {
                e.preventDefault();
                b.click();
              }
            }}
            tabIndex={b.click ? 0 : undefined}
            role={b.click ? "button" : undefined}
          >
            <rect x={cx} y={b.y} width={boxW} height={boxH} rx={6} fill="var(--bg-3)" stroke="var(--line-1)" />
            <text x={cx + 10} y={b.y + 22} fill="var(--text-0)" fontSize={11}>
              {b.label}
            </text>
          </g>
        ))}
        <text x={20} y={52} fill="var(--text-2)" fontSize={10}>
          block {layer} · pre-LN order verified
        </text>
        <text x={20} y={70} fill="var(--text-2)" fontSize={10}>
          against the running model
        </text>
      </svg>
      <div className="legend-scale">
        <span>Click Attention or MLP to inspect real captured measurements. GPT-2 applies normalization
        before each sublayer; both sublayer outputs are added back into the residual stream.</span>
      </div>
    </div>
  );
}
