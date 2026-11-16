import { useCallback, useEffect, useRef, useState } from "react";
import type { FlowData, ModelMeta } from "../api/client";
import { BLOCK_H, BLOCK_PITCH, BLOCK_Y0, CONTENT_W, fitCam, MODEL_H, VIEW_H, VIEW_W } from "./architecture/constants";
import ModelLevel from "./architecture/ModelLevel";
import BlockLevel from "./architecture/BlockLevel";
import AttentionLevel from "./architecture/AttentionLevel";

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

interface Cam {
  x: number;
  y: number;
  k: number;
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
  intervention,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  level: CanvasLevel;
  layer: number;
  onLevel: (l: CanvasLevel) => void;
  onLayer: (l: number) => void;
  onHead: (head: number) => void;
  keyboardFocusLayer: number;
  intervention?: { hook: string; position: number } | null;
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
              intervention={intervention}
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
