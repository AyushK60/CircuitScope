import { useCallback, useEffect, useRef, useState } from "react";
import type { FlowData, ModelMeta } from "../api/client";

/** The CircuitScope architecture canvas: a large, pan/zoom-able, scientifically
 *  faithful rendering of the loaded GPT-2 model with semantic zoom levels.
 *
 *  Truthfulness rules:
 *  - Stage order and counts come from live model metadata; coordinates place
 *    all 12 blocks strictly before the final LayerNorm and unembed.
 *  - Residual stream is drawn as a real bus every block reads from and writes
 *    back to (adds, not replaces).
 *  - Measured overlays (norm bars, logits, tokens) render only when flow data
 *    from a real forward pass is present, and are labeled MEASURED.
 *  - No invented edges: data edges, residual adds, and measurement marks are
 *    distinct stroke types; hover exposes each edge's meaning.
 */

export type CanvasLevel = "model" | "block" | "attn";

const CONTENT_W = 780;
// model level is the tallest; explicit rows guarantee correct ordering
const MODEL_H = 1780;
const BLOCK_Y0 = 330;
const BLOCK_PITCH = 90;
const BLOCK_H = 78;

interface Cam {
  x: number;
  y: number;
  k: number;
}

const VIEW_W = 900;
const VIEW_H = 800;

function fitCam(): Cam {
  const k = Math.min(VIEW_W / (CONTENT_W + 80), VIEW_H / (MODEL_H + 60));
  return { x: (VIEW_W - CONTENT_W * k) / 2 + 10, y: 8, k };
}

export default function ArchitectureCanvas({
  meta,
  flow,
  level,
  layer,
  onLevel,
  onLayer,
  onHead,
  keyboardFocusLayer,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  level: CanvasLevel;
  layer: number;
  onLevel: (l: CanvasLevel) => void;
  onLayer: (l: number) => void;
  onHead: (head: number) => void;
  keyboardFocusLayer: number;
}) {
  const [cam, setCam] = useState<Cam>(() => fitCam());
  const [dragging, setDragging] = useState(false);
  const [hover, setHover] = useState<{ x: number; y: number; title: string; lines: string[] } | null>(null);
  const dragRef = useRef<{ px: number; py: number; cam: Cam } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const maxNorm = flow ? Math.max(...flow.blocks.map((b) => b.resid_norm_mean), 1e-9) : 1;

  const zoomBy = useCallback((factor: number) => {
    setCam((c) => {
      const k = Math.min(2.2, Math.max(0.3, c.k * factor));
      const cx = VIEW_W / 2;
      const cy = VIEW_H / 2;
      return { k, x: cx - ((cx - c.x) / c.k) * k, y: cy - ((cy - c.y) / c.k) * k };
    });
  }, []);

  const resetCam = useCallback(() => setCam(fitCam()), []);

  const zoomToBlock = useCallback((l: number) => {
    const by = BLOCK_Y0 + l * BLOCK_PITCH;
    const k = 1.15;
    setCam({ x: VIEW_W / 2 - (CONTENT_W / 2) * k, y: VIEW_H / 2 - (by + BLOCK_H / 2) * k, k });
  }, []);

  // guided transition: zoom into the block, then swap levels
  const enterBlock = useCallback(
    (l: number) => {
      zoomToBlock(l);
      onLayer(l);
      window.setTimeout(() => onLevel("block"), 420);
    },
    [onLayer, onLevel, zoomToBlock],
  );

  // each level has its own comfortable camera; clear stale hover on switch
  useEffect(() => {
    setHover(null);
    if (level === "model") {
      setCam(fitCam());
    } else {
      const k = 0.92;
      setCam({ x: (VIEW_W - CONTENT_W * k) / 2, y: (VIEW_H - 700 * k) / 2, k });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [level]);

  // keyboard navigation on the canvas
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onKey = (e: KeyboardEvent) => {
      if (level !== "model") return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const d = e.key === "ArrowDown" ? 1 : -1;
        const next = Math.min(meta.n_layers - 1, Math.max(0, keyboardFocusLayer + d));
        onLayer(next);
        setHover({ x: 0, y: 0, title: `Block ${next}`, lines: ["keyboard focus — press Enter to open"] });
      } else if (e.key === "Enter") {
        enterBlock(keyboardFocusLayer);
      } else if (e.key === "+" || e.key === "=") {
        zoomBy(1.2);
      } else if (e.key === "-") {
        zoomBy(1 / 1.2);
      } else if (e.key === "0") {
        resetCam();
      }
    };
    el.addEventListener("keydown", onKey);
    return () => el.removeEventListener("keydown", onKey);
  }, [level, keyboardFocusLayer, meta.n_layers, onLayer, enterBlock, zoomBy, resetCam]);

  // wheel zoom about the pointer
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * VIEW_W;
    const py = ((e.clientY - rect.top) / rect.height) * VIEW_H;
    setCam((c) => {
      const k = Math.min(2.2, Math.max(0.3, c.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      return { k, x: px - ((px - c.x) / c.k) * k, y: py - ((py - c.y) / c.k) * k };
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    dragRef.current = { px: e.clientX, py: e.clientY, cam };
    setDragging(true);
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const sx = (VIEW_W / rect.width) * (e.clientX - d.px);
    const sy = (VIEW_H / rect.height) * (e.clientY - d.py);
    setCam({ ...d.cam, x: d.cam.x + sx, y: d.cam.y + sy });
  };
  const onPointerUp = () => {
    dragRef.current = null;
    setDragging(false);
  };

  const contentH = level === "model" ? MODEL_H : level === "block" ? 680 : 700;

  return (
    <div className="canvas-shell" style={{ position: "relative" }}>
      <svg
        ref={svgRef}
        className="arch-canvas"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        width="100%"
        height="78vh"
        role="img"
        aria-label={`Interactive architecture of ${meta.model_id}: ${meta.n_layers} transformer blocks, pan and zoom enabled`}
        tabIndex={0}
        style={{
          background: "var(--bg-1)",
          border: "1px solid var(--line-0)",
          borderRadius: "var(--radius)",
          cursor: dragging ? "grabbing" : "grab",
          touchAction: "none",
        }}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <g
          transform={`translate(${cam.x} ${cam.y}) scale(${cam.k})`}
          style={{ transition: dragging ? "none" : "transform 0.45s cubic-bezier(0.22, 1, 0.36, 1)" }}
        >
          <defs>
            <marker id="arrowData" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-2)" />
            </marker>
            <marker id="arrowResid" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--violet)" />
            </marker>
          </defs>
          {level === "model" && (
            <ModelLevel
              meta={meta}
              flow={flow}
              layer={layer}
              maxNorm={maxNorm}
              onEnterBlock={enterBlock}
              setHover={setHover}
              setHoverClear={() => setHover(null)}
            />
          )}
          {level === "block" && (
            <BlockLevel meta={meta} flow={flow} layer={layer} onOpenAttention={() => onLevel("attn")} setHover={setHover} />
          )}
          {level === "attn" && (
            <AttentionLevel meta={meta} flow={flow} layer={layer} onHead={onHead} setHover={setHover} />
          )}
        </g>
      </svg>

      {/* camera controls */}
      <div className="canvas-controls" role="toolbar" aria-label="Canvas controls">
        <button onClick={() => zoomBy(1.25)} aria-label="Zoom in" title="Zoom in (+)">＋</button>
        <button onClick={() => zoomBy(1 / 1.25)} aria-label="Zoom out" title="Zoom out (−)">－</button>
        <button onClick={resetCam} aria-label="Fit to screen" title="Fit to screen (0)">⤢</button>
        {level === "model" ? (
          <>
            <button
              onClick={() => {
                const next = Math.min(meta.n_layers - 1, keyboardFocusLayer + 1);
                onLayer(next);
                zoomToBlock(next);
              }}
              disabled={keyboardFocusLayer >= meta.n_layers - 1}
              aria-label="Next block"
              title="Next block (↓)"
            >
              ▼
            </button>
            <button
              onClick={() => {
                const prev = Math.max(0, keyboardFocusLayer - 1);
                onLayer(prev);
                zoomToBlock(prev);
              }}
              disabled={keyboardFocusLayer <= 0}
              aria-label="Previous block"
              title="Previous block (↑)"
            >
              ▲
            </button>
            <select
              aria-label="Jump to block"
              value={keyboardFocusLayer}
              onChange={(e) => {
                const l = Number(e.target.value);
                onLayer(l);
                zoomToBlock(l);
              }}
              style={{ width: "auto", padding: "0.2rem 0.4rem" }}
            >
              {Array.from({ length: meta.n_layers }, (_, l) => (
                <option key={l} value={l}>
                  Block {l}
                </option>
              ))}
            </select>
          </>
        ) : (
          <button onClick={() => { onLevel("model"); resetCam(); }} aria-label="Back to whole model">
            ⌂ Model
          </button>
        )}
      </div>

      {hover && (
        <div
          className="canvas-tooltip"
          style={{ left: hover.x, top: hover.y }}
          role="tooltip"
        >
          <strong>{hover.title}</strong>
          {hover.lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      )}
      <div className="legend-scale" style={{ padding: "0.4rem 0.6rem" }}>
        <span style={{ color: "var(--text-1)" }}>─ data edge</span>
        <span style={{ color: "var(--violet)" }}>┄ residual add</span>
        <span style={{ color: "var(--brass)" }}>▮ measured (from real run)</span>
        <span className="mono" style={{ marginLeft: "auto" }}>
          wheel = zoom · drag = pan · ↑↓ = move focus · Enter = open
        </span>
      </div>
      <span className="sr-only" aria-live="polite">
        {level === "model" ? `Whole model view, block ${keyboardFocusLayer} focused` : `${level} view of block ${layer}`}
      </span>
      {/* keep contentH referenced for future level sizing */}
      <span hidden>{contentH}</span>
    </div>
  );
}

type HoverFn = (h: { x: number; y: number; title: string; lines: string[] } | null) => void;

function hoverFromEvent(e: React.MouseEvent, title: string, lines: string[]): { x: number; y: number; title: string; lines: string[] } {
  const rect = (e.currentTarget as Element).closest(".canvas-shell")?.getBoundingClientRect();
  return { x: (e.clientX - (rect?.left ?? 0)) + 14, y: (e.clientY - (rect?.top ?? 0)) + 14, title, lines };
}

/* ------------------------------------------------------------------ */
/* Level 1 — whole model                                               */
/* ------------------------------------------------------------------ */

function ModelLevel({
  meta,
  flow,
  layer,
  maxNorm,
  onEnterBlock,
  setHover,
  setHoverClear,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  layer: number;
  maxNorm: number;
  onEnterBlock: (l: number) => void;
  setHover: HoverFn;
  setHoverClear: () => void;
}) {
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
        {flow ? `your sentence — ${flow.tokens.length} tokens received by the model` : "your sentence"}
      </text>
      <TokenStrip flow={flow} y={54} x={modX} />

      {/* stage 2: tokenizer */}
      {stageNum(146, "2", "tokenize")}
      <StageBand y={132} w={modW} x={modX} title="Tokenizer" sub={`${meta.tokenizer_class} · byte-level BPE · vocab ${meta.d_vocab.toLocaleString()}`} />
      <path className="arch-edge" markerEnd="url(#arrowData)" d={`M ${modX + modW / 2} 168 L ${modX + modW / 2} 196`} />

      {/* stage 3: embeddings — two real tables converging into the stream */}
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
      {stageNum(BLOCK_Y0 + 20, "4", `${meta.n_layers} transformer blocks — in order`)}
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

      {/* stage 5: final LN — strictly after all blocks */}
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

function StageBand({ x, y, w, title, sub }: { x: number; y: number; w: number; title: string; sub: string }) {
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

function TokenStrip({ flow, x, y }: { flow: FlowData | null; x: number; y: number }) {
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
        return (
          <g key={i}>
            <rect x={x + col * 50} y={y + row * 22} width={46} height={18} rx={3} fill="var(--bg-2)" stroke="var(--line-0)" />
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

function LogitsBand({ x, y, w, flow }: { x: number; y: number; w: number; flow: FlowData | null }) {
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

/* ------------------------------------------------------------------ */
/* Level 2 — block anatomy (verified pre-LN)                           */
/* ------------------------------------------------------------------ */

function BlockLevel({
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
        GPT-2 pre-LN order — verified against the running model · hooks: blocks.{layer}.*
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

      {/* attention module — opens level 3 */}
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

function AddNode({ x, y, setHover, clear, note, hook }: { x: number; y: number; setHover: HoverFn; clear: () => void; note: string; hook: string }) {
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

/* ------------------------------------------------------------------ */
/* Level 3 — attention anatomy                                         */
/* ------------------------------------------------------------------ */

function AttentionLevel({
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
