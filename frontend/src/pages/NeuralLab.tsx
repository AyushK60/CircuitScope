import { useMemo, useRef, useState } from "react";
import { createLabNetwork, edgeCount, forwardLab, type LabNetwork } from "../lib/feedforward";

/** Neural Network Lab — an educational, fully-connected feedforward network.
 *
 *  4 inputs → 6 → 6 → 5 → 3 outputs, ReLU hidden activations, stable softmax
 *  outputs, deterministic documented initial weights. Every displayed value is
 *  computed from the current inputs and the actual weight matrices.
 *
 *  This is NOT GPT-2 and NOT a measured transformer view: it is an educational
 *  simulation of a classic dense network, contrasted with the transformer in
 *  the Model Explorer. */

const INPUT_LABELS = ["in·0", "in·1", "in·2", "in·3"];
const OUT_LABELS = ["class A", "class B", "class C"];

const W = 940;
const H = 560;
const COL_X = [70, 280, 490, 700, 880];

function colY(l: number, i: number, sizes: number[]): number {
  const gap = Math.min(74, (H - 110) / sizes[l]);
  const y0 = H / 2 - ((sizes[l] - 1) * gap) / 2;
  return y0 + i * gap;
}

export default function NeuralLab() {
  const [seed, setSeed] = useState(1234);
  const net: LabNetwork = useMemo(() => createLabNetwork(seed), [seed]);
  const [inputs, setInputs] = useState<number[]>([0.5, -1, 1.2, 0]);
  const [sel, setSel] = useState<{ l: number; i: number } | null>(null);
  const [focusedOnly, setFocusedOnly] = useState(false);
  void setFocusedOnly;
  const [playLayer, setPlayLayer] = useState<number | null>(null);
  const [cam, setCam] = useState({ x: 0, y: 0, k: 1 });
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef<{ px: number; py: number; cam: typeof cam } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const result = useMemo(() => forwardLab(net, inputs), [net, inputs]);

  const reset = () => {
    setSeed(1234);
    setInputs([0.5, -1, 1.2, 0]);
    setSel(null);
    setPlayLayer(null);
    setCam({ x: 0, y: 0, k: 1 });
  };

  const zoomBy = (f: number) =>
    setCam((c) => {
      const k = Math.min(2.2, Math.max(0.4, c.k * f));
      return { k, x: 470 - ((470 - c.x) / c.k) * k, y: 280 - ((280 - c.y) / c.k) * k };
    });

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const py = ((e.clientY - rect.top) / rect.height) * H;
    setCam((c) => {
      const k = Math.min(2.2, Math.max(0.4, c.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      return { k, x: px - ((px - c.x) / c.k) * k, y: py - ((py - c.y) / c.k) * k };
    });
  };

  const selInfo = sel
    ? {
        layer: sel.l,
        index: sel.i,
        bias: sel.l > 0 ? net.b[sel.l - 1][sel.i] : null,
        z: sel.l > 0 ? result.z[sel.l - 1][sel.i] : null,
        a: result.a[sel.l][sel.i],
        incoming: sel.l > 0 ? net.W[sel.l - 1].map((row, j) => ({ from: j, w: row[sel.i] })) : null,
      }
    : null;

  const edgeOpacity = (w: number) => 0.12 + Math.min(0.75, Math.abs(w) / 1.4);
  const edgeColor = (w: number) => (w >= 0 ? "var(--brass)" : "var(--violet)");

  return (
    <main className="page" style={{ maxWidth: "1400px" }}>
      <h1>Neural Network Lab</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "80ch" }}>
        A classic fully-connected feedforward network: <strong>4 inputs → 6 → 6 → 5 → 3 outputs</strong>{" "}
        with ReLU hidden activations and a stable softmax. Adjust the inputs and every neuron value,
        weight path, and output probability is computed live from the actual weight matrices.
      </p>
      <div className="state-banner warn" style={{ marginBottom: "1rem" }}>
        <strong>Educational Feedforward Network — separate from GPT-2 Small.</strong> This toy network's
        outputs are abstract classes, not real-world predictions. The transformer explorer remains the
        authoritative view of the running language model.
      </div>

      <div className="panel">
        <h3>Inputs</h3>
        <div className="lab-input-grid">
          {inputs.map((v, i) => (
            <label key={i} htmlFor={`nlin${i}`} className="mono lab-input">
              {INPUT_LABELS[i]}
              <input
                id={`nlin${i}`}
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={v}
                onChange={(e) =>
                  setInputs((arr) => arr.map((x, j) => (j === i ? Number(e.target.value) : x)))
                }
              />
              <span className="lab-input-value">{v.toFixed(1)}</span>
            </label>
          ))}
          <button onClick={reset}>Reset</button>
          <label className="inspector-meta" style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
            <input type="checkbox" checked={focusedOnly} onChange={(e) => setFocusedOnly(e.target.checked)} />
            focus selected neuron's edges only
          </label>
        </div>
      </div>

      <div className="panel" style={{ padding: "0.6rem", position: "relative" }}>
        <div className="canvas-controls" style={{ position: "absolute", top: "0.9rem", right: "0.9rem", zIndex: 5 }}>
          <button onClick={() => zoomBy(1.25)} aria-label="Zoom in">＋</button>
          <button onClick={() => zoomBy(1 / 1.25)} aria-label="Zoom out">－</button>
          <button onClick={() => setCam({ x: 0, y: 0, k: 1 })} aria-label="Fit network">⤢</button>
        </div>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          height="560"
          role="img"
          aria-label={`Feedforward network with ${net.sizes.join(" → ")} neurons and ${edgeCount(net)} connections`}
          tabIndex={0}
          style={{ cursor: dragging ? "grabbing" : "grab", touchAction: "none" }}
          onWheel={onWheel}
          onPointerDown={(e) => {
            dragRef.current = { px: e.clientX, py: e.clientY, cam };
            setDragging(true);
          }}
          onPointerMove={(e) => {
            const d = dragRef.current;
            if (!d) return;
            const rect = svgRef.current!.getBoundingClientRect();
            setCam({
              ...d.cam,
              x: d.cam.x + (W / rect.width) * (e.clientX - d.px),
              y: d.cam.y + (H / rect.height) * (e.clientY - d.py),
            });
          }}
          onPointerUp={() => {
            dragRef.current = null;
            setDragging(false);
          }}
          onPointerLeave={() => {
            dragRef.current = null;
            setDragging(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") setSel(null);
            if (e.key === "ArrowRight") setPlayLayer((p) => Math.min(net.sizes.length - 1, (p ?? -1) + 1));
            if (e.key === "ArrowLeft") setPlayLayer((p) => Math.max(0, (p ?? 2) - 1));
          }}
        >
          <g transform={`translate(${cam.x} ${cam.y}) scale(${cam.k})`} style={{ transition: dragging ? "none" : "transform 0.4s cubic-bezier(0.22,1,0.36,1)" }}>
            <defs>
              <marker id="labArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--line-1)" />
              </marker>
            </defs>
            {/* edges */}
            {net.W.map((mat, l) =>
              mat.map((row, j) =>
                row.map((w, i) => {
                  const dim =
                    playLayer != null && l >= playLayer
                      ? 0.06
                      : focusedOnly && sel
                        ? !(
                            (sel.l === l + 1 && sel.i === j) ||
                            (sel.l === l && sel.i === i)
                          )
                        : false;
                  return (
                    <path
                      key={`e${l}-${j}-${i}`}
                      d={`M ${COL_X[l] + 10} ${colY(l, i, net.sizes)} C ${(COL_X[l] + COL_X[l + 1]) / 2} ${colY(l, i, net.sizes)}, ${(COL_X[l] + COL_X[l + 1]) / 2} ${colY(l + 1, j, net.sizes)}, ${COL_X[l + 1] - 10} ${colY(l + 1, j, net.sizes)}`}
                      stroke={edgeColor(w)}
                      strokeWidth={0.6 + Math.min(1.8, Math.abs(w))}
                      opacity={dim ? 0.05 : edgeOpacity(w)}
                      fill="none"
                      style={{ cursor: "pointer" }}
                      onClick={() => setSel({ l: l + 1, i: j })}
                    />
                  );
                }),
              ),
            )}
            {/* neurons */}
            {net.sizes.map((n, l) =>
              Array.from({ length: n }, (_, i) => {
                const a = result.a[l][i];
                const t = l === 0 ? (a + 2) / 4 : l === net.sizes.length - 1 ? a : Math.max(0, Math.min(1, a));
                const isSel = sel?.l === l && sel?.i === i;
                const dim = playLayer != null && l > playLayer;
                return (
                  <g
                    key={`n${l}-${i}`}
                    className="arch-node"
                    onClick={() => setSel(isSel ? null : { l, i })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSel(isSel ? null : { l, i });
                      }
                    }}
                    tabIndex={0}
                    role="button"
                    aria-label={`Neuron layer ${l} index ${i}: activation ${a.toFixed(3)}`}
                    opacity={dim ? 0.15 : 1}
                  >
                    <circle
                      cx={COL_X[l]}
                      cy={colY(l, i, net.sizes)}
                      r={16}
                      fill={`rgba(223,121,90,${0.08 + t * 0.8})`}
                      stroke={isSel ? "var(--lavender)" : "var(--line-1)"}
                      strokeWidth={isSel ? 2.4 : 1.2}
                    />
                    <text x={COL_X[l]} y={colY(l, i, net.sizes) + 3.5} fill="var(--text-0)" fontSize={9} textAnchor="middle" fontFamily="monospace">
                      {a.toFixed(2)}
                    </text>
                    {l === 0 && (
                      <text x={COL_X[l] - 26} y={colY(l, i, net.sizes) + 3} fill="var(--text-2)" fontSize={10} textAnchor="end">
                        {INPUT_LABELS[i]}
                      </text>
                    )}
                    {l === net.sizes.length - 1 && (
                      <text x={COL_X[l] + 26} y={colY(l, i, net.sizes) + 3} fill="var(--text-2)" fontSize={10}>
                        {OUT_LABELS[i]} {(result.probs[i] * 100).toFixed(1)}%
                      </text>
                    )}
                  </g>
                );
              }),
            )}
            {["input", "hidden 1 (ReLU)", "hidden 2 (ReLU)", "hidden 3 (ReLU)", "output (softmax)"].map((s, l) => (
              <text key={s} x={COL_X[l]} y={26} fill={playLayer === l ? "var(--accent)" : "var(--text-2)"} fontSize={11} textAnchor="middle">
                {s}
              </text>
            ))}
          </g>
        </svg>
        <div className="legend-scale" style={{ padding: "0.2rem 0.6rem" }}>
          <span>connection color: </span>
          <span style={{ color: "var(--brass)" }}>positive weight</span>
          <span> / </span>
          <span style={{ color: "var(--violet)" }}>negative weight</span>
          <span> · thickness & opacity ∝ |weight| · click a neuron or edge to inspect</span>
        </div>
      </div>

      <div className="grid-2" style={{ alignItems: "start", marginTop: "1rem" }}>
        <div className="panel">
          <h3>Selected neuron</h3>
          {!selInfo && <p className="inspector-meta">Click any neuron in the diagram to inspect its bias, pre-activation, and incoming weights.</p>}
          {selInfo && (
            <>
              <p className="inspector-meta mono">
                layer {selInfo.layer} · neuron {selInfo.index}
              </p>
              {selInfo.z != null ? (
                <table className="kv-table">
                  <tbody>
                    <tr><td>pre-activation z</td><td>{selInfo.z.toFixed(4)}</td></tr>
                    <tr><td>bias</td><td>{selInfo.bias!.toFixed(4)}</td></tr>
                    <tr><td>activation</td><td>{selInfo.a.toFixed(4)} {selInfo.layer < net.sizes.length - 1 ? "(ReLU: max(0, z))" : "(logit)"}</td></tr>
                    <tr><td>incoming weights</td><td>{selInfo.incoming!.map((w) => `w[${w.from}]=${w.w.toFixed(3)}`).join(" · ")}</td></tr>
                  </tbody>
                </table>
              ) : (
                <p className="inspector-meta mono">input neuron: value {selInfo.a.toFixed(2)} (set by the sliders above)</p>
              )}
            </>
          )}
          <h3 style={{ marginTop: "1rem" }}>Forward-pass replay</h3>
          <p className="inspector-meta" style={{ marginBottom: "0.6rem" }}>
            Schematic replay of the computed forward pass — each stage highlights the layer whose values
            were just computed. Presentation pacing only.
          </p>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            <button
              className="primary"
              onClick={() => {
                setPlayLayer(0);
                let l = 0;
                const t = setInterval(() => {
                  l += 1;
                  setPlayLayer(l);
                  if (l >= net.sizes.length - 1) clearInterval(t);
                }, 700);
              }}
            >
              ▶ Replay forward pass
            </button>
            <button onClick={() => setPlayLayer(null)}>Show all layers</button>
          </div>
          <h3 style={{ marginTop: "1rem" }}>Output probabilities</h3>
          {result.probs.map((p, i) => (
            <div className="bar-row" key={i}>
              <span className="label">{OUT_LABELS[i]}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ left: 0, width: `${p * 100}%`, background: "var(--accent)" }} />
              </div>
              <span className="value">{(p * 100).toFixed(1)}%</span>
            </div>
          ))}
          <p className="inspector-meta mono">logits: [{result.logits.map((v) => v.toFixed(2)).join(", ")}]</p>
        </div>
        <div className="panel">
          <h3>Compare with GPT-2</h3>
          <p style={{ color: "var(--text-1)" }}>
            This dense network pushes each input through fixed layers, one direction, no memory. GPT-2
            instead tokenizes text and lets every token <em>attend</em> to earlier tokens, writing each
            result into a shared residual stream that all twelve blocks read and write.
          </p>
          <ul className="fact-list">
            <li><strong>{edgeCount(net)} connections</strong> here — all weights are visible and inspectable</li>
            <li><strong>no attention</strong>: every neuron sees only the previous layer</li>
            <li><strong>no residual stream</strong>: values are replaced, not added to</li>
          </ul>
          <a href="#/explorer">
            <button className="primary">Open the GPT-2 Model Explorer →</button>
          </a>
          <h3 style={{ marginTop: "1.2rem" }}>Accessible activation table</h3>
          <div className="table-scroll">
            <table className="kv-table">
              <caption className="inspector-meta" style={{ captionSide: "top", textAlign: "left" }}>
                every neuron's current activation, computed from your inputs
              </caption>
              <tbody>
                {net.sizes.map((_n, l) => (
                  <tr key={l}>
                    <td>layer {l}</td>
                    <td>{result.a[l].map((v) => v.toFixed(2)).join(" · ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </main>
  );
}
