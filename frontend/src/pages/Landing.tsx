import { useEffect, useState } from "react";
import { api, type Health, type Prediction } from "../api/client";
import TopPredictions from "../components/TopPredictions";
import TokenRow from "../components/TokenRow";

const DEMO_TEXT = "John and Mary went to the shop. John gave a book to";

/* Hero architecture: a curated, clearly-labeled rendering of the real GPT-2
 * topology. Structure is accurate (12 blocks before final LN); the traveling
 * pulse is a schematic animation, never presented as measured activity. */
function HeroModel() {
  const blocks = Array.from({ length: 12 });
  const busX = 96;
  return (
    <svg
      className="arch-svg hero-model"
      viewBox="0 0 520 700"
      role="img"
      aria-label="Schematic of GPT-2: tokens, embeddings, twelve transformer blocks on a residual stream, final normalization, output scores"
    >
      <defs>
        <linearGradient id="hBlock" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#343a36" />
          <stop offset="100%" stopColor="#2b302d" />
        </linearGradient>
        <linearGradient id="hBus" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#df795a" />
          <stop offset="100%" stopColor="#a08cb8" />
        </linearGradient>
      </defs>

      {/* input tokens */}
      <rect x={150} y={10} width={310} height={30} rx={7} fill="var(--bg-2)" stroke="var(--line-1)" />
      <text x={165} y={30} fill="var(--text-1)" fontSize={12} fontFamily="var(--font-mono)">
        "John and Mary went to the shop …"
      </text>

      {/* tokenizer */}
      <rect x={150} y={62} width={310} height={26} rx={6} fill="var(--bg-3)" stroke="var(--line-1)" />
      <text x={165} y={79} fill="var(--text-2)" fontSize={10.5}>
        tokenizer · byte-level BPE · 50,257 vocabulary
      </text>
      <path d="M 305 88 L 305 104" stroke="var(--line-1)" fill="none" />

      {/* two embedding tables converging */}
      <rect x={150} y={106} width={148} height={26} rx={6} fill="var(--bg-3)" stroke="var(--line-1)" />
      <text x={160} y={123} fill="var(--text-2)" fontSize={10}>
        token embeddings
      </text>
      <rect x={312} y={106} width={148} height={26} rx={6} fill="var(--bg-3)" stroke="var(--line-1)" />
      <text x={322} y={123} fill="var(--text-2)" fontSize={10}>
        position embeddings
      </text>
      <path d="M 224 132 C 224 152, 96 152, 96 166" stroke="var(--line-1)" fill="none" />
      <path d="M 386 132 C 386 152, 96 152, 96 166" stroke="var(--line-1)" fill="none" />
      <text x={255} y={152} fill="var(--text-2)" fontSize={9} textAnchor="middle">
        added → residual stream
      </text>

      {/* residual bus */}
      <path d={`M ${busX} 166 L ${busX} 626`} stroke="url(#hBus)" strokeWidth={2.6} fill="none" />

      {/* blocks */}
      {blocks.map((_, i) => {
        const y = 172 + i * 38;
        return (
          <g key={i}>
            <path d={`M ${busX} ${y + 12} C 128 ${y + 12}, 130 ${y + 16}, 148 ${y + 16}`} stroke="var(--violet)" strokeDasharray="3 3" fill="none" opacity={0.75} />
            <rect x={148} y={y} width={312} height={32} rx={7} fill="url(#hBlock)" stroke="var(--line-1)" />
            <text x={160} y={y + 15} fill="var(--text-0)" fontSize={11} fontFamily="var(--font-display)">
              Block {i}
            </text>
            <rect x={228} y={y + 9} width={44} height={14} rx={3} fill="var(--bg-1)" />
            <text x={250} y={y + 19} fill="var(--text-2)" fontSize={7.5} textAnchor="middle">LN1</text>
            <rect x={276} y={y + 9} width={76} height={14} rx={3} fill="var(--bg-1)" />
            <text x={314} y={y + 19} fill="var(--text-2)" fontSize={7.5} textAnchor="middle">attention ×12</text>
            <rect x={356} y={y + 9} width={44} height={14} rx={3} fill="var(--bg-1)" />
            <text x={378} y={y + 19} fill="var(--text-2)" fontSize={7.5} textAnchor="middle">LN2</text>
            <rect x={404} y={y + 9} width={44} height={14} rx={3} fill="var(--bg-1)" />
            <text x={426} y={y + 19} fill="var(--text-2)" fontSize={7.5} textAnchor="middle">MLP</text>
            <path d={`M 460 ${y + 16} C 486 ${y + 16}, 118 ${y + 16}, ${busX} ${y + 24}`} stroke="var(--violet)" strokeDasharray="3 3" fill="none" opacity={0.55} />
          </g>
        );
      })}

      {/* schematic pulse — labeled, decorative, motion-safe */}
      <circle r={3.4} fill="var(--accent)" opacity={0.85}>
        <animate attributeName="cy" values="166;626" dur="6s" repeatCount="indefinite" />
        <animate attributeName="cx" values="96;96" dur="6s" repeatCount="indefinite" />
      </circle>

      {/* final LN + unembed + logits */}
      <rect x={148} y={636} width={312} height={26} rx={6} fill="var(--bg-3)" stroke="var(--line-1)" />
      <text x={160} y={653} fill="var(--text-2)" fontSize={10}>
        final layer norm → unembed → 50,257 logits
      </text>
      <path d={`M ${busX} 626 C ${busX} 664, 130 649, 148 649`} stroke="var(--line-1)" fill="none" />
      <rect x={228} y={672} width={152} height={20} rx={5} fill="#2b2620" stroke="var(--brass)" />
      <text x={304} y={686} fill="var(--brass)" fontSize={10.5} textAnchor="middle" fontFamily="var(--font-mono)">
        next-token scores
      </text>
    </svg>
  );
}

/* Real verified numbers from scripts/reference_patch.py on this machine —
 * clearly labeled as a reference result, not a live run. */
const REFERENCE = { dClean: 2.5128, dCorr: -4.1768, dPatch: 1.9426, recovery: 0.9148 };

export default function Landing({
  health,
  navigate,
}: {
  health: Health | null;
  navigate: (r: "explorer" | "playground" | "learn" | "lab") => void;
}) {
  const [demo, setDemo] = useState<Prediction | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [text, setText] = useState(DEMO_TEXT);
  const [hoverHead, setHoverHead] = useState<[number, number] | null>(null);

  const runDemo = (t: string) => {
    setLoading(true);
    setDemoError(null);
    setDemo(null);
    api
      .predict(t, 6)
      .then((r) => setDemo(r.prediction))
      .catch((e) => setDemoError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (health?.model_loaded && !demo && !demoError && !loading) runDemo(DEMO_TEXT);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [health?.model_loaded]);

  const live = health?.model_loaded === true;

  return (
    <main>
      {/* ================= HERO ================= */}
      <section className="hero">
        <div className="hero-grid">
          <div>
            <div className="kicker">NEURAL NETWORK OBSERVATORY · LOCAL-FIRST</div>
            <h1>
              Inspect the computation. <em>Change an activation.</em> Observe the prediction.
            </h1>
            <p className="lede">
              CircuitScope runs a real GPT-2 model on this computer. Follow your sentence through twelve
              transformer blocks, read the measurements inside, then replace one internal signal and watch
              the prediction change.
            </p>
            <div className="hero-actions">
              <button className="primary" onClick={() => navigate("explorer")}>
                Explore a real model →
              </button>
              <button onClick={() => navigate("playground")}>Predict &amp; Continue</button>
              <button onClick={() => navigate("lab")}>Neural Network Lab</button>
            </div>
            <p className="schematic-note">
              <span className={live ? "live-badge" : "static-badge"}>
                {live ? "LIVE MODEL READY" : "MODEL LOADING"}
              </span>
              {live && health.model
                ? ` GPT-2 Small · ${health.model.n_layers} blocks · ${health.model.n_heads} heads · ${health.model.n_params_approx}M parameters · ${health.model.device}`
                : " the visualization below is accurate structure; measured values appear when the model is ready"}
            </p>
          </div>
          <div className="hero-viz">
            <HeroModel />
            <div className="schematic-note">
              <span className="static-badge">SCHEMATIC</span> real topology · the moving pulse is an
              animation of stages, not GPU telemetry
            </div>
          </div>
        </div>
      </section>

      {/* ============ SECTION 1: SEE THE MODEL ============ */}
      <section className="landing-section">
        <div className="section-head">
          <span className="section-num">01</span>
          <h2>See the model</h2>
        </div>
        <div className="grid-2" style={{ alignItems: "start" }}>
          <div>
            <p style={{ color: "var(--text-1)", maxWidth: "52ch" }}>
              GPT-2 Small is a decoder-only transformer: tokens enter, twelve blocks each read and write a
              shared residual stream, and a final projection scores every possible next token. The explorer
              renders this faithfully — every count, hook name, and shape comes from the running model.
            </p>
            <ul className="fact-list">
              <li><strong>12 blocks</strong> in strictly sequential order</li>
              <li><strong>12 attention heads</strong> per block, running in parallel</li>
              <li><strong>1 residual stream</strong> — blocks add, they never replace</li>
              <li><strong>50,257 logits</strong> at the output projection</li>
            </ul>
            <button className="primary" onClick={() => navigate("explorer")}>
              Open the Model Explorer
            </button>
          </div>
          <div className="panel">
            <h3>Try a prediction right now</h3>
            <label htmlFor="demo-input" style={{ fontSize: "0.85rem", color: "var(--text-2)" }}>
              Input text (tokenized live)
            </label>
            <input id="demo-input" type="text" value={text} onChange={(e) => setText(e.target.value)} />
            <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.7rem" }}>
              <button className="primary" onClick={() => runDemo(text)} disabled={!live || loading}>
                {loading ? "Running…" : "Run prediction"}
              </button>
              <button onClick={() => { setText(DEMO_TEXT); runDemo(DEMO_TEXT); }}>Use example</button>
            </div>
            {demoError && <div className="state-banner error" style={{ marginTop: "0.8rem" }}>{demoError}</div>}
            {demo && (
              <>
                <h3 style={{ marginTop: "1rem" }}>Tokens the model receives</h3>
                <TokenRow tok={{ tokens: demo.tokens, token_ids: demo.token_ids, n_tokens: demo.tokens.length }} />
                <h3 style={{ marginTop: "0.8rem" }}>Top candidates — measured</h3>
                <TopPredictions candidates={demo.top_predictions} metric="prob" maxRows={5} />
              </>
            )}
            {!live && !loading && (
              <div className="state-banner info" style={{ marginTop: "0.8rem" }}>
                Waiting for the model — this demo runs the real weights, so it needs them loaded.
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ============ SECTION 2: FOLLOW A PREDICTION ============ */}
      <section className="landing-section alt">
        <div className="section-head">
          <span className="section-num">02</span>
          <h2>Follow a prediction</h2>
        </div>
        <p style={{ color: "var(--text-1)", maxWidth: "70ch" }}>
          After a real forward pass, CircuitScope can replay the recorded stages — tokenization,
          embeddings, every block's measured attention, final normalization, and the output scores — as a
          guided, step-controllable playback.
        </p>
        <div className="flow-strip">
          {["tokens", "embed", "block 0–11", "final LN", "logits"].map((s, i) => (
            <div key={s} style={{ display: "contents" }}>
              <div className="flow-step">
                <div className="flow-step-num">{i + 1}</div>
                <div>{s}</div>
              </div>
              {i < 4 && <div className="flow-arrow" aria-hidden>→</div>}
            </div>
          ))}
        </div>
        <p className="schematic-note">
          The replay is built from one recorded execution; stage pacing is presentation, never GPU timing.
        </p>
        <button onClick={() => { window.location.hash = "#/explorer?mode=flow"; }}>
          Watch a real computation flow →
        </button>
      </section>

      {/* ============ SECTION 3: CHANGE AN ACTIVATION ============ */}
      <section className="landing-section">
        <div className="section-head">
          <span className="section-num">03</span>
          <h2>Change an activation</h2>
        </div>
        <div className="grid-2" style={{ alignItems: "start" }}>
          <div>
            <p style={{ color: "var(--text-1)", maxWidth: "52ch" }}>
              Copy one internal activation from a clean run into a corrupted one, re-run the model, and
              measure what changes. Below is the verified reference result for the prepared example —
              produced by this machine's test suite, reproducible from the Playground in one click.
            </p>
            <div className="reference-result">
              <div className="ref-row">
                <span>clean run prefers</span>
                <strong> " Mary" (D = +{REFERENCE.dClean.toFixed(2)})</strong>
              </div>
              <div className="ref-row">
                <span>corrupted run prefers</span>
                <strong> " John" (D = {REFERENCE.dCorr.toFixed(2)})</strong>
              </div>
              <div className="ref-row">
                <span>after patching block 5's residual @ " John"</span>
                <strong> " Mary" again (D = +{REFERENCE.dPatch.toFixed(2)})</strong>
              </div>
              <div className="ref-row">
                <span>recovery of the clean behaviour</span>
                <strong> {(REFERENCE.recovery * 100).toFixed(1)}%</strong>
              </div>
            </div>
            <p className="schematic-note">
              <span className="static-badge">REFERENCE RESULT</span> measured by{" "}
              <code className="mono">scripts/reference_patch.py</code> — run it live below.
            </p>
            <button className="primary" onClick={() => navigate("playground")}>
              Run this intervention live →
            </button>
          </div>
          <div className="panel patch-diagram">
            <h3>What a patch is</h3>
            <svg viewBox="0 0 420 150" width="100%" role="img" aria-label="Diagram of activation patching: clean activation copied into corrupted run">
              <rect x={10} y={20} width={150} height={34} rx={6} fill="var(--bg-3)" stroke="var(--ok)" />
              <text x={85} y={41} fill="var(--text-1)" fontSize={11} textAnchor="middle">clean run activation</text>
              <rect x={10} y={94} width={150} height={34} rx={6} fill="var(--bg-3)" stroke="var(--danger)" />
              <text x={85} y={115} fill="var(--text-1)" fontSize={11} textAnchor="middle">corrupted run slot</text>
              <path d="M 160 37 C 210 37, 210 111, 164 111" stroke="var(--brass)" strokeWidth={2} strokeDasharray="5 4" fill="none" markerEnd="url(#refArrow)" />
              <defs>
                <marker id="refArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth={7} markerHeight={7} orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--brass)" />
                </marker>
              </defs>
              <rect x={280} y={55} width={130} height={40} rx={6} fill="var(--bg-2)" stroke="var(--line-1)" />
              <text x={345} y={72} fill="var(--text-0)" fontSize={11} textAnchor="middle">re-run corrupted</text>
              <text x={345} y={87} fill="var(--text-2)" fontSize={10} textAnchor="middle">with clean signal</text>
              <path d="M 164 111 C 230 111, 240 75, 278 75" stroke="var(--line-1)" fill="none" markerEnd="url(#refArrow)" />
            </svg>
            <p className="inspector-meta">
              The backend clones the clean tensor, splices it at the exact hook and token position, and
              re-executes — the baselines are never mutated (verified by tests).
            </p>
          </div>
        </div>
      </section>

      {/* ============ SECTION 4: DISCOVER THE COMPONENTS ============ */}
      <section className="landing-section alt">
        <div className="section-head">
          <span className="section-num">04</span>
          <h2>Discover the components</h2>
        </div>
        <p style={{ color: "var(--text-1)", maxWidth: "70ch" }}>
          144 attention heads, each a small specialist. Hover to see where a head lives; the explorer shows
          its real measured attention pattern for your own text.
        </p>
        <div className="head-matrix" role="group" aria-label="All 144 attention heads by block">
          {Array.from({ length: 12 }, (_, l) => (
            <div key={l} className="head-row">
              <span className="head-row-label">block {l}</span>
              <div className="head-row-cells">
                {Array.from({ length: 12 }, (_, h) => (
                  <button
                    key={h}
                    className={`head-dot${hoverHead && hoverHead[0] === l && hoverHead[1] === h ? " hot" : ""}`}
                    onMouseEnter={() => setHoverHead([l, h])}
                    onMouseLeave={() => setHoverHead(null)}
                    onFocus={() => setHoverHead([l, h])}
                    onBlur={() => setHoverHead(null)}
                    aria-label={`Block ${l} head ${h}`}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
        <p className="inspector-meta" style={{ minHeight: "1.4rem" }}>
          {hoverHead
            ? `block ${hoverHead[0]} · head ${hoverHead[1]} — blocks.${hoverHead[0]}.attn.hook_pattern`
            : "hover a cell — every one of these heads has a measurable pattern"}
        </p>
        <button onClick={() => navigate("explorer")}>Inspect a real head →</button>
      </section>

      {/* ============ SECTION 5: ENTER THE LABORATORY ============ */}
      <section className="landing-cta">
        <span className="section-num" style={{ display: "block" }}>05</span>
        <h2>Enter the laboratory</h2>
        <p style={{ color: "var(--text-1)", maxWidth: "60ch", margin: "0.6rem auto 1.4rem" }}>
          The Intervention Playground runs three real model passes — clean, corrupted, patched — and shows
          you the measured difference, including honest no-effect results.
        </p>
        <button className="primary" onClick={() => navigate("playground")}>
          Open the Intervention Playground →
        </button>
        <p className="schematic-note" style={{ justifyContent: "center" }}>
          <button className="ghost" onClick={() => navigate("learn")} style={{ border: "none", color: "var(--text-2)", textDecoration: "underline" }}>
            New here? Read what a transformer is →
          </button>
        </p>
      </section>
    </main>
  );
}
