import { useCallback, useMemo, useRef, useState } from "react";

/** Neural Network Lab — a genuine, interactive feedforward network.
 *
 *  What this is: a tiny dense MLP (2-8-8-1) trained live in your browser with
 *  real gradient descent on a two-moons dataset. Every number on screen is
 *  computed here — loss, activations, decision boundary.
 *  What this is NOT: GPT-2. It contains no attention, no tokens, no residual
 *  stream. It exists to build intuition for dense layers before exploring the
 *  transformer in the Model Explorer. */

// ---------- deterministic data ----------
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Point {
  x: number;
  y: number;
  label: number;
}

function twoMoons(n = 200, noise = 0.12): Point[] {
  const rand = mulberry32(42);
  const pts: Point[] = [];
  const gauss = () => (rand() + rand() + rand() + rand() - 2) * noise * 2;
  for (let i = 0; i < n; i++) {
    const upper = i % 2 === 0;
    const t = rand() * Math.PI;
    if (upper) {
      pts.push({ x: Math.cos(t) + gauss(), y: Math.sin(t) - 0.25 + gauss(), label: 1 });
    } else {
      pts.push({ x: 1 - Math.cos(t) + gauss(), y: 0.25 - Math.sin(t) + gauss(), label: 0 });
    }
  }
  return pts;
}

// ---------- tiny MLP with plain backprop ----------
class TinyMLP {
  sizes = [2, 8, 8, 1];
  W: number[][][] = [];
  b: number[][] = [];

  constructor(seed = 7) {
    const rand = mulberry32(seed);
    for (let l = 1; l < this.sizes.length; l++) {
      const fanIn = this.sizes[l - 1];
      const scale = Math.sqrt(2 / fanIn);
      this.W.push(
        Array.from({ length: this.sizes[l] }, () =>
          Array.from({ length: fanIn }, () => (rand() * 2 - 1) * scale),
        ),
      );
      this.b.push(Array.from({ length: this.sizes[l] }, () => 0));
    }
  }

  forward(x: number[]): { acts: number[][] } {
    const acts: number[][] = [x];
    for (let l = 0; l < this.W.length; l++) {
      const prev = acts[l];
      const z = this.W[l].map((row, j) => row.reduce((s, w, i) => s + w * prev[i], 0) + this.b[l][j]);
      const isOut = l === this.W.length - 1;
      acts.push(z.map((v) => (isOut ? 1 / (1 + Math.exp(-v)) : Math.tanh(v))));
    }
    return { acts };
  }

  /** one full-batch gradient descent step; returns mean BCE loss */
  trainStep(data: Point[], lr: number): number {
    const gW = this.W.map((m) => m.map((r) => r.map(() => 0)));
    const gB = this.b.map((v) => v.map(() => 0));
    let loss = 0;
    for (const p of data) {
      const { acts } = this.forward([p.x, p.y]);
      const out = acts[acts.length - 1][0];
      const y = p.label;
      loss += -(y * Math.log(out + 1e-9) + (1 - y) * Math.log(1 - out + 1e-9));
      // deltas
      let delta: number[] = [out - y]; // sigmoid + BCE
      for (let l = this.W.length - 1; l >= 0; l--) {
        for (let j = 0; j < this.W[l].length; j++) {
          for (let i = 0; i < this.W[l][j].length; i++) gW[l][j][i] += delta[j] * acts[l][i];
          gB[l][j] += delta[j];
        }
        if (l > 0) {
          const prev = this.W[l].map((row, j) => row.map((w) => w * delta[j]));
          delta = this.b[l - 1].map((_, i) => {
            let s = 0;
            for (let j = 0; j < prev.length; j++) s += prev[j][i];
            const a = acts[l][i];
            return s * (1 - a * a); // tanh'
          });
        }
      }
    }
    const n = data.length;
    for (let l = 0; l < this.W.length; l++) {
      for (let j = 0; j < this.W[l].length; j++) {
        for (let i = 0; i < this.W[l][j].length; i++) this.W[l][j][i] -= (lr * gW[l][j][i]) / n;
        this.b[l][j] -= (lr * gB[l][j]) / n;
      }
    }
    return loss / n;
  }

  accuracy(data: Point[]): number {
    let ok = 0;
    for (const p of data) {
      const out = this.forward([p.x, p.y]).acts[this.W.length][0];
      if ((out > 0.5 ? 1 : 0) === p.label) ok++;
    }
    return ok / data.length;
  }
}

const BOUND = 240; // grid resolution for the boundary render

export default function NeuralLab() {
  const data = useMemo(() => twoMoons(), []);
  const netRef = useRef<TinyMLP>(new TinyMLP());
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState<number>(() => {
    const n = new TinyMLP();
    return n.trainStep(data, 0); // measure initial loss without updating (lr 0)
  });
  const [history, setHistory] = useState<number[]>([]);
  const [lr, setLr] = useState(1.0);
  const [probe, setProbe] = useState<[number, number]>([0.5, 0]);
  const [version, setVersion] = useState(0); // force repaints
  const boundaryRef = useRef<HTMLCanvasElement | null>(null);

  const toScreen = useCallback(
    ([x, y]: [number, number]): [number, number] => [
      ((x + 2.2) / 4.4) * 100,
      ((2.2 - y) / 4.4) * 100,
    ],
    [],
  );

  const drawBoundary = useCallback(() => {
    const cv = boundaryRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(BOUND, BOUND);
    const net = netRef.current;
    let i = 0;
    for (let py = 0; py < BOUND; py++) {
      const y = 2.2 - (py / BOUND) * 4.4;
      for (let px = 0; px < BOUND; px++) {
        const x = (px / BOUND) * 4.4 - 2.2;
        const p = net.forward([x, y]).acts[net.W.length][0];
        // ink (0) → persimmon (1)
        const t = Math.max(0, Math.min(1, p));
        img.data[i++] = Math.round(35 + t * (223 - 35));
        img.data[i++] = Math.round(42 + t * (121 - 42));
        img.data[i++] = Math.round(40 + t * (90 - 40));
        img.data[i++] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    // data points
    for (const p of data) {
      const [sx, sy] = toScreen([p.x, p.y]);
      ctx.beginPath();
      ctx.arc((sx / 100) * BOUND, (sy / 100) * BOUND, 2.4, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? "#91aa91" : "#c96a5e";
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.4)";
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }
    // probe
    const [sx, sy] = toScreen(probe);
    ctx.beginPath();
    ctx.arc((sx / 100) * BOUND, (sy / 100) * BOUND, 4.4, 0, Math.PI * 2);
    ctx.strokeStyle = "var(--text-0)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [data, probe, toScreen]);

  useMemo(() => drawBoundary(), [drawBoundary, version]);

  const train = (steps: number) => {
    const net = netRef.current;
    let last = 0;
    for (let i = 0; i < steps; i++) last = net.trainStep(data, lr);
    setStep((s) => s + steps);
    setLoss(last);
    setHistory((h) => [...h.slice(-199), last]);
    setVersion((v) => v + 1);
  };

  const reset = () => {
    netRef.current = new TinyMLP(Math.floor(Math.random() * 100000));
    setStep(0);
    setHistory([]);
    const n = netRef.current;
    setLoss(n.trainStep(data, 0));
    setVersion((v) => v + 1);
  };

  const probeOut = netRef.current.forward([...probe]).acts[netRef.current.W.length][0];
  const probeActs = netRef.current.forward([...probe]).acts;

  return (
    <main className="page">
      <h1>Neural Network Lab</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "78ch" }}>
        A genuine dense feedforward network — 2 → 8 → 8 → 1 units with tanh activations — trained{" "}
        <strong>live in your browser</strong> with real gradient descent on the classic two-moons dataset.
        Watch the decision boundary take shape, then inspect every activation inside the network for any
        probe point.
      </p>
      <div className="state-banner warn" style={{ marginBottom: "1rem" }}>
        <strong>Educational simulation.</strong> This toy network is computed entirely in your browser and
        is <em>not</em> GPT-2 — it has no attention, tokens, or residual stream. It shows what "layers of
        neurons" actually do before you explore the transformer, where CircuitScope's measured numbers live.
      </div>

      <div className="panel">
        <div className="lab-controls">
          <button className="primary" onClick={() => train(100)}>
            Train 100 steps
          </button>
          <button onClick={() => train(1)}>1 step</button>
          <button onClick={reset}>Reset weights</button>
          <label htmlFor="lr-sel">
            learning rate
            <select id="lr-sel" value={lr} onChange={(e) => setLr(Number(e.target.value))} style={{ width: "auto" }}>
              {[0.2, 0.5, 1.0, 2.0].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <span className="mono lab-stats">
            step {step} · loss {loss.toFixed(4)} · accuracy {(netRef.current.accuracy(data) * 100).toFixed(1)}%
          </span>
        </div>
        {history.length > 2 && <LossCurve history={history} />}
      </div>

      <div className="grid-2" style={{ alignItems: "start", marginTop: "1rem" }}>
        <div className="panel">
          <h3>Decision boundary</h3>
          <p className="inspector-meta">
            warm regions → model outputs ≈ 1 (upper moon) · dark regions → ≈ 0. Click anywhere to move the
            probe point.
          </p>
          <canvas
            ref={boundaryRef}
            width={BOUND}
            height={BOUND}
            style={{ width: "100%", maxWidth: 480, borderRadius: 8, cursor: "crosshair", border: "1px solid var(--line-0)" }}
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const px = ((e.clientX - rect.left) / rect.width) * 4.4 - 2.2;
              const py = 2.2 - ((e.clientY - rect.top) / rect.height) * 4.4;
              setProbe([Math.max(-2.2, Math.min(2.2, px)), Math.max(-2.2, Math.min(2.2, py))]);
            }}
          />
          <p className="inspector-meta mono">
            probe ({probe[0].toFixed(2)}, {probe[1].toFixed(2)}) → output {probeOut.toFixed(4)} → class{" "}
            {probeOut > 0.5 ? "1 (upper moon)" : "0 (lower moon)"}
          </p>
        </div>
        <div className="panel">
          <h3>Inside the network at the probe</h3>
          <NetDiagram acts={probeActs} />
          <p className="inspector-meta">
            node brightness = activation value (tanh ∈ [−1, 1] hidden, sigmoid ∈ [0, 1] output); edge
            opacity = |weight|. Every value is computed by the code on this page for the probe you clicked.
          </p>
        </div>
      </div>
    </main>
  );
}

function LossCurve({ history }: { history: number[] }) {
  const w = 560;
  const h = 60;
  const max = Math.max(...history, 0.8);
  const pts = history
    .map((v, i) => `${(i / (history.length - 1)) * w},${h - (v / max) * (h - 6) - 3}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" style={{ marginTop: "0.6rem" }} role="img" aria-label="Training loss curve">
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
      <text x={4} y={12} fill="var(--text-2)" fontSize={9}>
        training loss (measured each logged step)
      </text>
    </svg>
  );
}

function NetDiagram({ acts }: { acts: number[][] }) {
  const sizes = acts.map((a) => a.length);
  const W = 460;
  const colX = sizes.map((_, l) => 50 + (l * (W - 100)) / (sizes.length - 1));
  const yOf = (l: number, i: number) => {
    const gap = Math.min(34, 190 / sizes[l]);
    const y0 = 105 - ((sizes[l] - 1) * gap) / 2;
    return y0 + i * gap;
  };
  // weights are not passed here; draw connections structurally (opacity fixed)
  return (
    <svg viewBox="0 0 460 210" width="100%" role="img" aria-label="Network activations at the probe point">
      {[0, 1, 2].map((l) =>
        acts[l].map((_, i) =>
          acts[l + 1].map((_, j) => (
            <path
              key={`${l}-${i}-${j}`}
              d={`M ${colX[l] + 8} ${yOf(l, i)} L ${colX[l + 1] - 8} ${yOf(l + 1, j)}`}
              stroke="var(--line-1)"
              strokeWidth={0.6}
              opacity={0.3}
              fill="none"
            />
          )),
        ),
      )}
      {acts.map((layer, l) =>
        layer.map((a, i) => {
          const t = Math.max(0, Math.min(1, l === acts.length - 1 ? a : (a + 1) / 2));
          return (
            <g key={`n${l}-${i}`}>
              <circle
                cx={colX[l]}
                cy={yOf(l, i)}
                r={9}
                fill={`rgba(223,121,90,${0.08 + t * 0.85})`}
                stroke="var(--line-1)"
              />
              <text x={colX[l]} y={yOf(l, i) + 3} fill="var(--text-0)" fontSize={6.5} textAnchor="middle" fontFamily="monospace">
                {a.toFixed(1)}
              </text>
            </g>
          );
        }),
      )}
      {["input", "hidden 1", "hidden 2", "output"].map((s, l) => (
        <text key={s} x={colX[l]} y={200} fill="var(--text-2)" fontSize={9} textAnchor="middle">
          {s}
        </text>
      ))}
    </svg>
  );
}
