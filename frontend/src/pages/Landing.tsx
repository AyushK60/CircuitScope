import { useEffect, useState } from "react";
import { api, type Health, type Prediction } from "../api/client";
import TopPredictions from "../components/TopPredictions";
import TokenRow from "../components/TokenRow";

const DEMO_TEXT = "John and Mary went to the shop. John gave a book to";

/** Static architectural illustration for the hero: labeled transformer block
 *  stack. Deliberately marked as illustration — it renders no invented values. */
function HeroSchematic() {
  const blocks = Array.from({ length: 12 });
  return (
    <svg className="arch-svg" width={220} height={360} role="img" aria-label="Illustration of a 12-block transformer">
      <defs>
        <linearGradient id="gBlock" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#1c2638" />
          <stop offset="100%" stopColor="#15202f" />
        </linearGradient>
      </defs>
      <rect x={80} y={4} width={60} height={20} rx={4} fill="#173038" stroke="#2b7f78" />
      <text x={110} y={18} fill="#5ad1c8" fontSize={9} textAnchor="middle">text</text>
      {blocks.map((_, i) => (
        <g key={i}>
          <rect x={52} y={34 + i * 24} width={116} height={18} rx={4} fill="url(#gBlock)" stroke="#2e3b56" />
          <rect x={60} y={39 + i * 24} width={28} height={8} rx={2} fill="#24405c" />
          <rect x={132} y={39 + i * 24} width={28} height={8} rx={2} fill="#3c3358" />
        </g>
      ))}
      <rect x={80} y={332} width={60} height={20} rx={4} fill="#241c10" stroke="#7a5a2e" />
      <text x={110} y={346} fill="#e8a05a" fontSize={9} textAnchor="middle">logits</text>
    </svg>
  );
}

export default function Landing({
  health,
  navigate,
}: {
  health: Health | null;
  navigate: (r: "explorer" | "playground" | "learn") => void;
}) {
  const [demo, setDemo] = useState<Prediction | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [text, setText] = useState(DEMO_TEXT);

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

  return (
    <main className="page">
      <section className="hero">
        <div className="kicker">NEURAL NETWORK OBSERVATORY · LOCAL-FIRST</div>
        <div className="grid-2" style={{ alignItems: "center" }}>
          <div>
            <h1>Inspect the computation. Change an activation. Observe the prediction.</h1>
            <p className="lede">
              CircuitScope runs a real GPT-2 language model on this computer. Watch your sentence become
              tokens, travel through 12 transformer blocks, and turn into next-word scores — then change
              something inside the model and see the measured effect.
            </p>
            <div className="hero-actions">
              <button className="primary" onClick={() => navigate("explorer")}>
                Explore a real model →
              </button>
              <button onClick={() => navigate("playground")}>Open the Intervention Playground</button>
              <button className="ghost" onClick={() => navigate("learn")}>
                What is a transformer?
              </button>
            </div>
            <p className="schematic-note">
              <span className={health?.model_loaded ? "live-badge" : "static-badge"}>
                {health?.model_loaded ? "LIVE MODEL READY" : "MODEL LOADING"}
              </span>
              {health?.model_loaded && health.model
                ? ` GPT-2 Small on ${health.model.device} · ${health.model.n_layers} blocks · ${health.model.n_params_approx}M parameters`
                : " weights load on first use; this page updates when ready"}
            </p>
          </div>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <div>
              <HeroSchematic />
              <div className="schematic-note">
                <span className="static-badge">ARCHITECTURAL ILLUSTRATION</span> structure only — no measured
                values shown here
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="pillars">
        <div className="pillar">
          <div className="num">01 · PREDICT</div>
          <h3>See a genuine prediction</h3>
          <p>
            Enter any sentence. CircuitScope shows the exact tokens the model receives and its real
            next-token scores, computed by the pretrained model — nothing hardcoded.
          </p>
          <p>
            <button className="ghost" onClick={() => runDemo(text)} disabled={!health?.model_loaded || loading}>
              ▶ Run a live prediction
            </button>
          </p>
        </div>
        <div className="pillar">
          <div className="num">02 · INSPECT</div>
          <h3>Open the architecture</h3>
          <p>
            Walk from the full model down to a single attention head. Heatmaps and shapes come from
            recorded activations of your own input, labeled with the hook that produced them.
          </p>
          <p>
            <button className="ghost" onClick={() => navigate("explorer")}>
              Open the Model Explorer
            </button>
          </p>
        </div>
        <div className="pillar">
          <div className="num">03 · INTERVENE</div>
          <h3>Change what happens inside</h3>
          <p>
            Replace one internal activation with its value from a different run and compare predictions
            before and after. Effects are measured in logits — including honest no-effect results.
          </p>
          <p>
            <button className="ghost" onClick={() => navigate("playground")}>
              Run a verified intervention
            </button>
          </p>
        </div>
      </section>

      <section className="panel">
        <h2>Try it right now</h2>
        <p style={{ color: "var(--text-1)" }}>
          The prepared example asks the model to continue a sentence about two people. Who receives the book?
        </p>
        <div className="grid-2" style={{ alignItems: "start" }}>
          <div>
            <label htmlFor="demo-input" style={{ fontSize: "0.85rem", color: "var(--text-2)", display: "block", marginBottom: "0.3rem" }}>
              Input text (tokenized live)
            </label>
            <input id="demo-input" type="text" value={text} onChange={(e) => setText(e.target.value)} />
            <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.7rem" }}>
              <button className="primary" onClick={() => runDemo(text)} disabled={!health?.model_loaded || loading}>
                {loading ? "Running…" : "Run prediction"}
              </button>
              <button onClick={() => { setText(DEMO_TEXT); runDemo(DEMO_TEXT); }}>Use example</button>
            </div>
            {demoError && (
              <div className="state-banner error" style={{ marginTop: "0.8rem" }}>
                {demoError}
              </div>
            )}
            <h3 style={{ marginTop: "1.2rem" }}>Tokens the model actually receives</h3>
            <TokenRow tok={demo ? { tokens: demo.tokens, token_ids: demo.token_ids, n_tokens: demo.tokens.length } : null} />
          </div>
          <div>
            <h3>Top next-token candidates (measured logits)</h3>
            {!health?.model_loaded && !loading && (
              <div className="state-banner info">
                Waiting for the model to finish loading — values appear when it is ready.
              </div>
            )}
            <TopPredictions candidates={demo?.top_predictions ?? null} metric="prob" />
            {demo && (
              <p className="inspector-meta">
                Computed in {demo.elapsed_seconds?.toFixed?.(2) ?? "—"} s on {demo.device}. These are softmax
                probabilities of the model's real logits at the final position.
              </p>
            )}
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Why change an activation?</h2>
        <p style={{ color: "var(--text-1)", maxWidth: "70ch" }}>
          A prediction is the end of a long chain of computations. By copying one internal signal from a
          different sentence and re-running the model, you can test whether that signal carries the
          information the output depends on. This is how interpretability researchers form hypotheses about
          what individual components do — carefully, with controls, and without claiming the model's whole
          "reasoning" is revealed.
        </p>
        <p className="schematic-note">
          CircuitScope never fabricates results: a weak or zero effect is reported as measured, and every
          number in the interface can be reproduced by a Python test.
        </p>
      </section>
    </main>
  );
}
