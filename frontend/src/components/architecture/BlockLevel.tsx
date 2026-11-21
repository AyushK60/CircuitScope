import type { FlowData, ModelMeta } from "../../api/client";
import { hoverFromEvent, type HoverFn } from "./hover";
import { AddNode, StageBand } from "./parts";

export default function BlockLevel({
  meta,
  flow,
  layer,
  onOpenAttention,
  setHover,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  layer: number;
  onOpenAttention: () => void;
  setHover: HoverFn;
}) {
  const cx = 250;
  const bw = 300;
  const rx = 640;
  const clear = () => setHover(null);
  return (
    <g>
      <text x={30} y={40} fill="var(--text-0)" fontSize={20} fontFamily="var(--font-display)">
        Inside transformer block {layer}
      </text>
      <text x={30} y={60} fill="var(--text-2)" fontSize={11}>
        GPT-2 pre-LN order - verified against the running model · hooks: blocks.{layer}.*
      </text>

      {/* residual rails */}
      <path className="arch-edge resid" strokeWidth={2.2} d={`M ${rx} 96 L ${rx} 216`} markerEnd="url(#arrowResid)" />
      <text x={rx + 10} y={160} fill="var(--violet)" fontSize={10} transform={`rotate(90 ${rx + 10} 160)`}>
        pass-through (added, not consumed)
      </text>
      <path className="arch-edge resid" strokeWidth={2.2} d={`M ${rx} 336 L ${rx} 520`} markerEnd="url(#arrowResid)" />

      {/* input */}
      <StageBand x={cx} y={84} w={bw} title="input x" sub={`residual stream · ${meta.d_model} dims`} />
      {flow && (
        <g>
          {flow.blocks[layer].resid_norm_per_token.slice(0, 14).map((n, i) => (
            <rect key={i} x={cx + i * 20} y={126} width={12} height={Math.max(2, (n / 500) * 24)} fill="var(--brass)" opacity={0.8} />
          ))}
          <text x={cx + bw} y={140} fill="var(--brass)" fontSize={8} fontFamily="var(--font-mono)">
            MEASURED norm/token
          </text>
        </g>
      )}
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${cx + bw / 2} 124 L ${cx + bw / 2} 148`} />

      <StageBand x={cx} y={152} w={bw} title="LayerNorm 1" sub={`blocks.${layer}.ln1 · feeds attention only`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${cx + bw / 2} 192 L ${cx + bw / 2} 216`} />

      {/* attention module - opens level 3 */}
      <g
        className="arch-node"
        onClick={onOpenAttention}
        role="button"
        tabIndex={0}
        aria-label="Open attention anatomy"
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            onOpenAttention();
          }
        }}
        onMouseEnter={(e) =>
          setHover(hoverFromEvent(e, "Attention sublayer", [`12 heads × d_head ${meta.d_head}`, "click to open attention anatomy (scale 3)"]))
        }
        onMouseLeave={clear}
      >
        <rect x={cx} y={220} width={bw} height={96} rx={9} fill="var(--bg-3)" stroke="var(--accent)" strokeDasharray="none" />
        <text x={cx + 14} y={244} fill="var(--text-0)" fontSize={13}>
          Attention × {meta.n_heads} heads
        </text>
        <text x={cx + 14} y={262} fill="var(--text-2)" fontSize={10}>
          Q · K · V projections → scores → softmax →
        </text>
        <text x={cx + 14} y={276} fill="var(--text-2)" fontSize={10}>
          weighted values → output projection
        </text>
        <text x={cx + bw - 12} y={300} fill="var(--accent)" fontSize={10} textAnchor="end">
          open anatomy →
        </text>
      </g>
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${cx + bw / 2} 316 L ${cx + bw / 2} 340`} />

      {/* first residual add */}
      <AddNode x={cx + bw / 2} y={352} setHover={setHover} clear={clear} note="⊕ attention output added to stream" hook={`blocks.${layer}.hook_out (mid)`} />
      <path className="arch-edge resid" d={`M ${rx} 216 C ${rx} 352, ${cx + bw / 2 + 40} 352, ${cx + bw / 2 + 26} 352`} />

      <StageBand x={cx} y={384} w={bw} title="LayerNorm 2" sub={`blocks.${layer}.ln2 · feeds MLP only`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${cx + bw / 2} 424 L ${cx + bw / 2} 448`} />

      <StageBand x={cx} y={452} w={bw} title={`MLP · ${meta.d_model} → ${meta.d_mlp} → ${meta.d_model}`} sub={`GELU · blocks.${layer}.mlp.*`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${cx + bw / 2} 492 L ${cx + bw / 2} 516`} />

      <AddNode x={cx + bw / 2} y={528} setHover={setHover} clear={clear} note="⊕ MLP output added → next block" hook={`blocks.${layer}.hook_out`} />
      <path className="arch-edge resid" d={`M ${rx} 336 C ${rx} 528, ${cx + bw / 2 + 40} 528, ${cx + bw / 2 + 26} 528`} />

      <StageBand x={cx} y={560} w={bw} title="output → block " sub={`stream continues to block ${layer + 1 < meta.n_layers ? layer + 1 : "final LN"}`} />
    </g>
  );
}
