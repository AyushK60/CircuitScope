import type { FlowData, ModelMeta } from "../../api/client";
import { BLOCK_H, BLOCK_PITCH, BLOCK_Y0 } from "./constants";
import { hoverFromEvent, type HoverFn } from "./hover";
import { LogitsBand, StageBand, TokenStrip } from "./parts";

export default function ModelLevel({
  meta,
  flow,
  layer,
  maxNorm,
  onEnterBlock,
  setHover,
  setHoverClear,
  intervention,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  layer: number;
  maxNorm: number;
  onEnterBlock: (l: number) => void;
  setHover: HoverFn;
  setHoverClear: () => void;
  intervention?: { hook: string; position: number } | null;
}) {
  const targetLayer = intervention ? Number((intervention.hook.match(/blocks\.(\d+)\./) ?? [])[1] ?? -1) : -1;
  const busX = 108;
  const modX = 210;
  const modW = 430;
  const stageNum = (y: number, n: string, label: string) => (
    <g key={`stage${n}${y}`}>
      <text x={26} y={y} fill="var(--brass)" fontSize={26} fontFamily="var(--font-display)">
        {n}
      </text>
      <text x={26} y={y + 18} fill="var(--text-2)" fontSize={9} style={{ textTransform: "uppercase", letterSpacing: "0.08em" }}>
        {label}
      </text>
    </g>
  );

  return (
    <g>
      {/* stage 1: input */}
      {stageNum(46, "1", "input")}
      <text x={modX} y={40} fill="var(--text-2)" fontSize={11}>
        {flow ? `your sentence - ${flow.tokens.length} tokens received by the model` : "your sentence"}
      </text>
      <TokenStrip flow={flow} y={54} x={modX} markedPosition={intervention?.position} />

      {/* stage 2: tokenizer */}
      {stageNum(146, "2", "tokenize")}
      <StageBand y={132} w={modW} x={modX} title="Tokenizer" sub={`${meta.tokenizer_class} · byte-level BPE · vocab ${meta.d_vocab.toLocaleString()}`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${modX + modW / 2} 168 L ${modX + modW / 2} 196`} />

      {/* stage 3: embeddings - two real tables converging into the stream */}
      {stageNum(236, "3", "embed")}
      <StageBand y={200} w={modW / 2 - 10} x={modX} title="Token embeddings" sub={`W_E · ${meta.d_vocab}→${meta.d_model}`} />
      <StageBand y={200} w={modW / 2 - 10} x={modX + modW / 2 + 10} title="Position embeddings" sub={`W_pos · ${meta.n_ctx}×${meta.d_model}`} />
      {/* converge into the residual stream bus */}
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${modX + modW / 4} 234 C ${modX + modW / 4} 268, ${busX} 268, ${busX} 296`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${modX + (3 * modW) / 4} 234 C ${modX + (3 * modW) / 4} 268, ${busX} 268, ${busX} 296`} />
      <text x={modX + modW / 2} y={262} fill="var(--text-2)" fontSize={10} textAnchor="middle">
        added element-wise → residual stream x₀
      </text>

      {/* residual bus */}
      <path
        className="arch-edge resid"
        d={`M ${busX} 296 L ${busX} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 30}`}
        strokeWidth={2.4}
      />
      <text x={busX - 14} y={BLOCK_Y0 + (meta.n_layers * BLOCK_PITCH) / 2} fill="var(--violet)" fontSize={11} transform={`rotate(-90 ${busX - 14} ${BLOCK_Y0 + (meta.n_layers * BLOCK_PITCH) / 2})`} textAnchor="middle">
        residual stream
      </text>

      {/* stage 4: the 12 real blocks, in execution order */}
      {stageNum(BLOCK_Y0 + 20, "4", `${meta.n_layers} transformer blocks - in order`)}
      {Array.from({ length: meta.n_layers }, (_, l) => {
        const by = BLOCK_Y0 + l * BLOCK_PITCH;
        const norm = flow?.blocks[l]?.resid_norm_mean;
        const selected = layer === l;
        return (
          <g
            key={`b${l}`}
            className={`arch-node${selected ? " selected" : ""}`}
            onClick={() => onEnterBlock(l)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onEnterBlock(l);
              }
            }}
            tabIndex={0}
            role="button"
            aria-label={`Open transformer block ${l}`}
            onMouseEnter={(e) =>
              setHover(
                hoverFromEvent(
                  e,
                  `Transformer block ${l}`,
                  [
                    `hooks: blocks.${l}.attn.* · blocks.${l}.mlp.*`,
                    `pre-LN: LN1 → attention → ⊕ → LN2 → MLP → ⊕`,
                    norm != null ? `measured residual norm after block: ${norm}` : "measured overlay appears after a run",
                  ],
                ),
              )
            }
            onMouseLeave={setHoverClear}
          >
            <rect x={modX} y={by} width={modW} height={BLOCK_H} rx={9} fill="var(--bg-2)" stroke={selected ? "var(--lavender)" : "var(--line-1)"} />
            <text x={modX + 14} y={by + 20} fill="var(--text-0)" fontSize={13} fontFamily="var(--font-display)">
              Block {l}
            </text>
            {/* internal pre-LN strip (miniature, hints at level 2) */}
            {[
              { x: 0.13, w: 0.12, label: "LN1" },
              { x: 0.27, w: 0.24, label: "attention ×12" },
              { x: 0.53, w: 0.12, label: "LN2" },
              { x: 0.67, w: 0.24, label: "MLP 3072" },
            ].map((s) => (
              <g key={s.label}>
                <rect x={modX + s.x * modW} y={by + 32} width={s.w * modW} height={22} rx={4} fill="var(--bg-3)" stroke="var(--line-0)" />
                <text x={modX + s.x * modW + s.w * modW * 0.5} y={by + 47} fill="var(--text-1)" fontSize={9} textAnchor="middle">
                  {s.label}
                </text>
              </g>
            ))}
            <text x={modX + modW - 12} y={by + 22} fill="var(--text-2)" fontSize={10} textAnchor="end">
              ⊕ ⊕
            </text>
            {/* synchronized intervention overlay from the Playground */}
            {intervention && l === targetLayer && (
              <g>
                <rect x={modX - 3} y={by - 3} width={modW + 6} height={BLOCK_H + 6} rx={11} fill="none" stroke="var(--brass)" strokeWidth={2} strokeDasharray="6 4" />
                <text x={modX + modW / 2} y={by - 9} fill="var(--brass)" fontSize={9.5} textAnchor="middle" fontFamily="var(--font-mono)">
                  INTERVENTION TARGET · {intervention.hook} @ pos {intervention.position}
                </text>
              </g>
            )}
            {/* measured norm overlay */}
            {norm != null && (
              <g>
                <rect x={modX + modW + 12} y={by + 14} width={8} height={BLOCK_H - 28} fill="var(--bg-3)" />
                <rect
                  x={modX + modW + 12}
                  y={by + 14 + (BLOCK_H - 28) * (1 - norm / maxNorm)}
                  width={8}
                  height={(BLOCK_H - 28) * (norm / maxNorm)}
                  fill="var(--brass)"
                  rx={2}
                />
              </g>
            )}
          </g>
        );
      })}
      {/* bus taps: read + write for each block */}
      {Array.from({ length: meta.n_layers }, (_, l) => {
        const by = BLOCK_Y0 + l * BLOCK_PITCH;
        return (
          <g key={`tap${l}`}>
            <path className="arch-edge resid" markerEnd="url(#arrowResid)" d={`M ${busX} ${by + 20} C ${modX - 24} ${by + 20}, ${modX - 24} ${by + 26}, ${modX} ${by + 26}`} />
            <path className="arch-edge resid" markerEnd="url(#arrowResid)" d={`M ${modX + modW} ${by + 56} C ${modX + modW + 26} ${by + 56}, ${busX + 26} ${by + 56}, ${busX} ${by + 62}`} />
          </g>
        );
      })}

      {/* stage 5: final LN - strictly after all blocks */}
      {stageNum(BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 52, "5", "final normalization")}
      <StageBand y={BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 66} w={modW} x={modX} title="Final LayerNorm" sub={`ln_final · normalizes the stream once, after every block`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${busX} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 30} C ${busX} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 90}, ${modX - 30} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 90}, ${modX + modW / 2} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 66}`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${modX + modW / 2} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 102} L ${modX + modW / 2} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 126}`} />

      {/* stage 6: unembed */}
      {stageNum(BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 140, "6", "unembed")}
      <StageBand y={BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 154} w={modW} x={modX} title="Output projection" sub={`W_U · linear ${meta.d_model}→${meta.d_vocab} logits`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${modX + modW / 2} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 190} L ${modX + modW / 2} ${BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 214}`} />

      {/* stage 7: logits */}
      {stageNum(BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 228, "7", "next-token scores")}
      <LogitsBand y={BLOCK_Y0 + meta.n_layers * BLOCK_PITCH + 242} x={modX} w={modW} flow={flow} />
    </g>
  );
}
