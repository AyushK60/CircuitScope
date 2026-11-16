import type { FlowData, ModelMeta } from "../../api/client";
import { hoverFromEvent, type HoverFn } from "./hover";
import { StageBand } from "./parts";

export default function AttentionLevel({
  meta,
  flow,
  layer,
  onHead,
  setHover,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  layer: number;
  onHead: (h: number) => void;
  setHover: HoverFn;
}) {
  const clear = () => setHover(null);
  const lanes = Array.from({ length: meta.n_heads }, (_, h) => h);
  return (
    <g>
      <text x={30} y={40} fill="var(--text-0)" fontSize={20} fontFamily="var(--font-display)">
        Attention anatomy — block {layer}
      </text>
      <text x={30} y={60} fill="var(--text-2)" fontSize={11}>
        parallel heads: the 12 lanes run together in one computation, not sequentially
      </text>

      <StageBand x={40} y={84} w={160} title="from LayerNorm 1" sub={`blocks.${layer}.ln1.hook_normalized`} />
      {/* Q K V projections */}
      {(["Q", "K", "V"] as const).map((k, i) => (
        <g key={k}>
          <path className="arch-edge" d={`M ${200} ${102} C ${240} ${102}, ${250} ${140 + i * 60}, ${290} ${140 + i * 60}`} />
          <StageBand x={290} y={122 + i * 60} w={190} title={`${k} projection`} sub={`W_${k.toLowerCase()} · ${meta.d_model}→${meta.n_heads}×${meta.d_head}`} />
        </g>
      ))}

      {/* per-head lanes */}
      <text x={520} y={110} fill="var(--text-2)" fontSize={10}>
        one lane per head (click to inspect)
      </text>
      {lanes.map((h, i) => (
        <g
          key={h}
          className="arch-node"
          role="button"
          tabIndex={0}
          aria-label={`Inspect head ${h}`}
          onClick={() => {
            setHover(null);
            onHead(h);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              onHead(h);
            }
          }}
          onMouseEnter={(e) =>
            setHover(
              hoverFromEvent(e, `Head ${h}`, [
                `blocks.${layer}.attn.hook_q/k/v [1, seq, 12, 64]`,
                flow ? `measured pattern: blocks.${layer}.attn.hook_pattern` : "measured pattern appears after a run",
                "click to see its real token-level attention",
              ]),
            )
          }
          onMouseLeave={clear}
        >
          <rect x={510} y={118 + i * 26} width={150} height={20} rx={4} fill="var(--bg-3)" stroke="var(--line-0)" />
          <text x={520} y={132 + i * 26} fill="var(--text-1)" fontSize={10}>
            head {h}
          </text>
          <text x={648} y={132 + i * 26} fill="var(--text-2)" fontSize={8.5} textAnchor="end">
            64 dims
          </text>
        </g>
      ))}
      {/* converge */}
      <path className="arch-edge" d={`M ${480} ${132} C ${500} ${132}, ${498} ${232}, ${510} 232`} opacity={0.4} />
      <path className="arch-edge" d={`M ${480} ${252} C ${500} ${252}, ${498} ${252}, ${510} 252`} opacity={0.4} />
      <path className="arch-edge" d={`M ${480} ${372} C ${500} ${372}, ${498} ${272}, ${510} 272`} opacity={0.4} />

      {/* per-head pipeline */}
      <StageBand x={510} y={452} w={150} title="QKᵀ / √64" sub="scores [heads, q, k]" />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${585} ${440} L ${585} ${448}`} />
      <StageBand x={510} y={496} w={150} title="causal mask" sub="future positions hidden" />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${585} ${488} L ${585} ${492}`} />
      <StageBand x={510} y={540} w={150} title="softmax" sub="rows sum to 1" />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${585} ${532} L ${585} ${536}`} />
      <StageBand x={510} y={584} w={150} title="weighted Σ V" sub="A·V per head" />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${585} ${576} L ${585} ${580}`} />

      {/* combined output */}
      <StageBand x={290} y={620} w={370} title="concat 12×64 → 768 → output projection W_O" sub={`blocks.${layer}.attn.hook_out · then ⊕ residual`} />

      {/* measured pattern preview */}
      {flow && (
        <g>
          <text x={40} y={660} fill="var(--brass)" fontSize={10} fontFamily="var(--font-mono)">
            MEASURED — attention patterns for this block are captured in the current run (inspect any head above)
          </text>
        </g>
      )}
    </g>
  );
}
