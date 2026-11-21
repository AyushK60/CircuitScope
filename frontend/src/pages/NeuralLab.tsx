import { useEffect, useMemo, useRef, useState } from "react";
import { createLabNetwork, edgeCount, forwardLab, type LabNetwork } from "../lib/feedforward";

/** Neural Network Lab: an educational, fully-connected feedforward network.
 *
 *  4 inputs → 6 → 6 → 5 → 3 outputs, ReLU hidden activations, stable softmax outputs, deterministic
 *  documented initial weights. Every displayed value is computed from the current inputs and the actual
 *  weight matrices. Each connection is drawn by how much signal it carries right now (weight × the
 *  activation of the neuron it leaves), so lines swell, thin and vanish as the inputs move.
 *
 *  This is NOT GPT-2 and NOT a measured transformer view: it is an educational simulation of a classic
 *  dense network, contrasted with the transformer in the Model Explorer. */

const INPUT_LABELS = ["in·0", "in·1", "in·2", "in·3"];
const OUT_LABELS = ["class A", "class B", "class C"];
const LAYER_TITLES = ["input", "hidden 1 · ReLU", "hidden 2 · ReLU", "hidden 3 · ReLU", "output · softmax"];

const W = 1100;
const H = 500;
const COL_X = [200, 375, 545, 715, 845];
const RADIUS = 13;
const CIRC = 2 * Math.PI * RADIUS;

function nodeY(l: number, i: number, sizes: number[]): number {
  const gap = Math.min(72, (H - 120) / sizes[l]);
  return H / 2 + (i - (sizes[l] - 1) / 2) * gap + 8;
}

const pctX = (x: number) => `${(x / W) * 100}%`;
const pctY = (y: number) => `${(y / H) * 100}%`;

/** The value card shown while a neuron is hovered or focused: what it holds and where it came from. */
function NodeTip({
  net,
  result,
  l,
  i,
  last,
}: {
  net: LabNetwork;
  result: ReturnType<typeof forwardLab>;
  l: number;
  i: number;
  last: number;
}) {
  const a = result.a[l][i];
  const x = COL_X[l];
  const y = nodeY(l, i, net.sizes);
  const kind = l === 0 ? "input" : l === last ? "output" : "hidden";
  const lines = [
    `${kind} neuron ${i}`,
    l === last ? `logit ${a.toFixed(3)} · ${(result.probs[i] * 100).toFixed(1)}%` : `activation ${a.toFixed(3)}`,
    ...(l > 0 ? [`z ${result.z[l - 1][i].toFixed(3)} · bias ${net.b[l - 1][i].toFixed(3)}`] : []),
    "click to inspect",
  ];
  const w = Math.max(...lines.map((s) => s.length)) * 6.6 + 22;
  const h = lines.length * 16 + 14;
  const tx = Math.min(W - w - 4, Math.max(4, x - w / 2));
  const ty = y - RADIUS - 12 - h < 38 ? y + RADIUS + 12 : y - RADIUS - 12 - h;
  return (
    <g pointerEvents="none" className="nl-tip">
      <rect x={tx} y={ty} width={w} height={h} rx={9} fill="var(--text-0)" opacity={0.96} />
      {lines.map((s, k) => (
        <text
          key={k}
          x={tx + 11}
          y={ty + 19 + k * 16}
          fontSize={11.5}
          fill={k === 0 ? "var(--bg-1)" : k === lines.length - 1 ? "var(--bg-3)" : "var(--bg-1)"}
          fontWeight={k === 0 ? 700 : 400}
          fontFamily="var(--font-mono)"
        >
          {s}
        </text>
      ))}
    </g>
  );
}

export default function NeuralLab() {
  const [seed] = useState(1234);
  const net: LabNetwork = useMemo(() => createLabNetwork(seed), [seed]);
  const [inputs, setInputs] = useState<number[]>([0.5, -1, 1.2, 0]);
  const [sel, setSel] = useState<{ l: number; i: number } | null>(null);
  // hovering (or keyboard-focusing) a neuron previews what a click does: its connections light up
  const [hover, setHover] = useState<{ l: number; i: number } | null>(null);
  const focus = hover ?? sel;
  const [playLayer, setPlayLayer] = useState<number | null>(null);
  const result = useMemo(() => forwardLab(net, inputs), [net, inputs]);
  const timer = useRef<number | null>(null);
  useEffect(() => () => {
    if (timer.current) clearInterval(timer.current);
  }, []);

  const top = result.probs.indexOf(Math.max(...result.probs));
  const last = net.sizes.length - 1;

  const reset = () => {
    setInputs([0.5, -1, 1.2, 0]);
    setSel(null);
    setPlayLayer(null);
  };

  const replay = () => {
    if (timer.current) clearInterval(timer.current);
    setPlayLayer(0);
    let l = 0;
    timer.current = window.setInterval(() => {
      l += 1;
      setPlayLayer(l);
      if (l >= last) {
        if (timer.current) clearInterval(timer.current);
        window.setTimeout(() => setPlayLayer(null), 900);
      }
    }, 650);
  };

  // how much signal each connection carries: weight × the activation of the neuron it leaves
  const flows = useMemo(
    () =>
      net.W.map((mat, l) => {
        const c = mat.map((row) => row.map((w, i) => w * result.a[l][i]));
        const max = Math.max(...c.flat().map((v) => Math.abs(v)), 1e-9);
        return { c, max };
      }),
    [net, result],
  );

  const selInfo = sel
    ? {
        layer: sel.l,
        index: sel.i,
        bias: sel.l > 0 ? net.b[sel.l - 1][sel.i] : null,
        z: sel.l > 0 ? result.z[sel.l - 1][sel.i] : null,
        a: result.a[sel.l][sel.i],
        incoming: sel.l > 0 ? net.W[sel.l - 1][sel.i].map((w, from) => ({ from, w })) : null,
      }
    : null;

  const tOf = (l: number, i: number) => {
    const a = result.a[l][i];
    if (l === 0) return (a + 2) / 4;
    if (l === last) return result.probs[i];
    return Math.max(0, Math.min(1, a));
  };

  return (
    <main className="page nl-page">
      <p className="ex-kick">A classic network, for contrast</p>
      <h1 className="ex-title">Neural Network Lab</h1>
      <p className="ex-sub">
        A small fully-connected network: <strong>4 inputs → 6 → 6 → 5 → 3 outputs</strong>, with ReLU hidden units and a
        stable softmax. Drag a slider beside an input and watch the whole network answer: every neuron, connection and
        output is computed live from the actual weight matrices.
      </p>
      <div className="nl-note" role="note">
        <strong>Separate from GPT-2.</strong> This toy network&apos;s outputs are abstract classes, not predictions about
        language. The Model Explorer is the authoritative view of the running language model.
      </div>

      <div className="nl-stage">
        <div className="nl-sliders">
          {inputs.map((v, i) => (
            <label
              key={i}
              htmlFor={`nlin${i}`}
              className="mono nl-slid"
              style={{ ["--x" as string]: pctX(8), ["--y" as string]: pctY(nodeY(0, i, net.sizes)) }}
            >
              <span>{INPUT_LABELS[i]}</span>
              <input
                id={`nlin${i}`}
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={v}
                onChange={(e) => setInputs((arr) => arr.map((x, j) => (j === i ? Number(e.target.value) : x)))}
              />
              <em>{v.toFixed(1)}</em>
            </label>
          ))}
        </div>

        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Feedforward network with ${net.sizes.join(" → ")} neurons and ${edgeCount(net)} connections`}
          onKeyDown={(e) => {
            if (e.key === "Escape") setSel(null);
          }}
        >
          {net.W.map((mat, l) =>
            mat.map((row, j) =>
              row.map((_w, i) => {
                const c = flows[l].c[j][i];
                const r = Math.abs(c) / flows[l].max;
                if (r < 0.04) return null;
                const touches = !focus || (focus.l === l && focus.i === i) || (focus.l === l + 1 && focus.i === j);
                const lit = !!focus && touches;
                const dim = playLayer !== null && l >= playLayer;
                const x1 = COL_X[l] + RADIUS - 2;
                const x2 = COL_X[l + 1] - RADIUS + 2;
                const mx = (x1 + x2) / 2;
                return (
                  <path
                    key={`e${l}-${j}-${i}`}
                    d={`M ${x1} ${nodeY(l, i, net.sizes)} C ${mx} ${nodeY(l, i, net.sizes)}, ${mx} ${nodeY(l + 1, j, net.sizes)}, ${x2} ${nodeY(l + 1, j, net.sizes)}`}
                    fill="none"
                    stroke={c >= 0 ? "var(--div-pos)" : "var(--div-neg)"}
                    strokeWidth={0.4 + r * 5.5}
                    strokeLinecap="round"
                    opacity={dim ? 0.04 : lit ? Math.min(1, 0.35 + r * 0.7) : (0.12 + r * 0.6) * (touches ? 1 : 0.25)}
                    style={{ cursor: "pointer", transition: "opacity 0.3s" }}
                    onClick={() => setSel({ l: l + 1, i: j })}
                  />
                );
              }),
            ),
          )}

          {net.sizes.map((n, l) =>
            Array.from({ length: n }, (_, i) => {
              const a = result.a[l][i];
              const t = tOf(l, i);
              const isSel = sel?.l === l && sel?.i === i;
              const isHover = hover?.l === l && hover?.i === i;
              const dim = playLayer !== null && l > playLayer;
              const x = COL_X[l];
              const y = nodeY(l, i, net.sizes);
              return (
                <g
                  key={`n${l}-${i}`}
                  className="nl-node"
                  onMouseEnter={() => setHover({ l, i })}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover({ l, i })}
                  onBlur={() => setHover(null)}
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
                  <circle cx={x} cy={y} r={RADIUS + 5} fill="transparent" />
                  <circle cx={x} cy={y} r={RADIUS} fill="var(--bg-1)" stroke="var(--bg-3)" strokeWidth={isHover ? 6 : 4.5} style={{ transition: "stroke-width 0.15s" }} />
                  <circle
                    cx={x}
                    cy={y}
                    r={RADIUS}
                    fill="none"
                    stroke={isSel || isHover ? "var(--accent)" : "var(--measure)"}
                    strokeWidth={isHover ? 6 : 4.5}
                    strokeLinecap="round"
                    strokeDasharray={`${Math.max(0.01, CIRC * t)} ${CIRC}`}
                    transform={`rotate(-90 ${x} ${y})`}
                    style={{ transition: "stroke-width 0.15s" }}
                  />
                  {(l === 0 || isSel) && !isHover && (
                    <text x={x} y={y - RADIUS - 9} fontSize={11} textAnchor="middle" fill="var(--text-2)" fontFamily="var(--font-mono)">
                      {a.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            }),
          )}

          {hover && <NodeTip net={net} result={result} l={hover.l} i={hover.i} last={last} />}

          {LAYER_TITLES.map((s, l) => (
            <text key={s} x={COL_X[l]} y={26} fontSize={12} textAnchor="middle" fill={playLayer === l ? "var(--accent)" : "var(--text-2)"}>
              {s}
            </text>
          ))}
        </svg>

        <div className="nl-outs">
          {result.probs.map((p, i) => (
            <div
              key={i}
              className={`nl-obar${i === top ? " top1" : ""}`}
              style={{ ["--x" as string]: pctX(878), ["--y" as string]: pctY(nodeY(last, i, net.sizes)) }}
            >
              <span>{OUT_LABELS[i]}</span>
              <div>
                <i style={{ width: `${p * 100}%` }} />
              </div>
              <b>{(p * 100).toFixed(1)}%</b>
            </div>
          ))}
        </div>

        <div className="nl-foot">
          <p className="nl-legend">
            <span>
              <i className="pos" />
              positive signal
            </span>
            <span>
              <i className="neg" />
              negative signal
            </span>
            <span>Line weight = how much signal passes (weight × the neuron before it). A switched-off neuron sends nothing. Ring = how active a neuron is. Click a neuron or line to inspect it.</span>
          </p>
          <div className="nl-actions">
            <button className="primary" onClick={replay}>
              Replay forward pass
            </button>
            <button onClick={reset}>Reset</button>
          </div>
        </div>
      </div>

      <div className="grid-2 nl-below">
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
                    <tr>
                      <td>pre-activation z</td>
                      <td>{selInfo.z.toFixed(4)}</td>
                    </tr>
                    <tr>
                      <td>bias</td>
                      <td>{selInfo.bias!.toFixed(4)}</td>
                    </tr>
                    <tr>
                      <td>activation</td>
                      <td>
                        {selInfo.a.toFixed(4)} {selInfo.layer < last ? "(ReLU: max(0, z))" : "(logit)"}
                      </td>
                    </tr>
                    <tr>
                      <td>incoming weights</td>
                      <td>{selInfo.incoming!.map((w) => `w[${w.from}]=${w.w.toFixed(3)}`).join(" · ")}</td>
                    </tr>
                  </tbody>
                </table>
              ) : (
                <p className="inspector-meta mono">input neuron: value {selInfo.a.toFixed(2)} (set by the slider beside it)</p>
              )}
            </>
          )}
          <p className="inspector-meta mono" style={{ marginTop: "0.8rem" }}>
            logits: [{result.logits.map((v) => v.toFixed(2)).join(", ")}]
          </p>
        </div>

        <div className="panel">
          <h3>Compared with GPT-2</h3>
          <p style={{ color: "var(--text-1)" }}>
            This dense network pushes each input through fixed layers, one direction, no memory. GPT-2 instead tokenizes
            text and lets every token <em>attend</em> to earlier tokens, writing each result into a shared residual stream
            that all twelve blocks read and write.
          </p>
          <ul className="nl-facts">
            <li>
              <strong>{edgeCount(net)} connections</strong> here: all weights are visible and inspectable
            </li>
            <li>
              <strong>no attention</strong>: every neuron sees only the previous layer
            </li>
            <li>
              <strong>no residual stream</strong>: values are replaced, not added to
            </li>
          </ul>
          <a href="#/explorer">
            <button className="primary">Open the GPT-2 Model Explorer →</button>
          </a>
          <details className="sw-details">
            <summary>Accessible activation table</summary>
            <div className="table-scroll">
              <table className="kv-table">
                <caption className="inspector-meta" style={{ captionSide: "top", textAlign: "left" }}>
                  every neuron&apos;s current activation, computed from your inputs
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
          </details>
        </div>
      </div>
    </main>
  );
}
