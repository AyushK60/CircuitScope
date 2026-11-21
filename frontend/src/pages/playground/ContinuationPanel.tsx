import type { ContinueResult } from "../../api/client";

/** "Custom prediction & continuation" panel: real greedy/sampling generation
 *  with step-by-step measured tokens. */
export default function ContinuationPanel({
  ready,
  contMode,
  setContMode,
  contTemp,
  setContTemp,
  contTopK,
  setContTopK,
  contSeed,
  setContSeed,
  contResult,
  setContResult,
  contBusy,
  runContinuation,
}: {
  ready: boolean;
  contMode: "greedy" | "sample";
  setContMode: (m: "greedy" | "sample") => void;
  contTemp: number;
  setContTemp: (v: number) => void;
  contTopK: number;
  setContTopK: (v: number) => void;
  contSeed: number;
  setContSeed: (v: number) => void;
  contResult: ContinueResult | null;
  setContResult: (r: ContinueResult | null) => void;
  contBusy: boolean;
  runContinuation: (nTokens: number) => void;
}) {
  return (
    <div className="pg-card">
      <h2>Custom prediction & continuation</h2>
      <p className="inspector-meta" style={{ marginBottom: "0.8rem" }}>
        Predict & Continue: the real model appends tokens to the prompt, re-running on the growing
        sequence. Greedy is deterministic; sampling uses temperature, top-k and a reproducible seed.
        Generate one token at a time to step through the run.
      </p>
      <div className="state-banner warn" style={{ marginBottom: "0.8rem" }}>
        GPT-2 Small predicts text continuations. Its generations are not guaranteed to be factually
        accurate or instruction-following.
      </div>
      <div style={{ display: "flex", gap: "0.8rem", alignItems: "center", flexWrap: "wrap" }}>
        <label className="inspector-meta" style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          mode
          <select value={contMode} onChange={(e) => { setContMode(e.target.value as "greedy" | "sample"); setContResult(null); }} style={{ width: "auto" }}>
            <option value="greedy">greedy</option>
            <option value="sample">sample</option>
          </select>
        </label>
        {contMode === "sample" && (
          <>
            <label className="inspector-meta" style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              temp {contTemp.toFixed(2)}
              <input type="range" min={0.05} max={2} step={0.05} value={contTemp} onChange={(e) => setContTemp(Number(e.target.value))} />
            </label>
            <label className="inspector-meta" style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              top-k {contTopK || "off"}
              <input type="range" min={0} max={50} step={1} value={contTopK} onChange={(e) => setContTopK(Number(e.target.value))} />
            </label>
            <label className="inspector-meta" style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              seed
              <input type="number" value={contSeed} onChange={(e) => setContSeed(Number(e.target.value))} style={{ width: "6rem" }} />
            </label>
          </>
        )}
      </div>
      <div style={{ display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap", marginTop: "0.8rem" }}>
        <button className="primary" onClick={() => runContinuation(5)} disabled={!ready || contBusy}>
          {contBusy ? "Generating…" : "Generate 5 tokens"}
        </button>
        <button onClick={() => runContinuation(1)} disabled={!ready || contBusy}>
          + 1 token
        </button>
        <button onClick={() => setContResult(null)} disabled={!contResult}>
          Reset to prompt
        </button>
        {contResult && (
          <span className="inspector-meta mono">
            {contResult.decoding} · {contResult.steps.length} tokens total
          </span>
        )}
      </div>
      {contResult && (
        <div style={{ marginTop: "0.8rem" }}>
          <p className="mono" style={{ fontSize: "0.95rem", background: "var(--bg-2)", padding: "0.6rem 0.8rem", borderRadius: 6 }}>
            {contResult.text_start}
            {contResult.steps.map((s) => (
              <a
                key={s.step}
                href={`#/explorer?text=${encodeURIComponent(s.input_so_far)}`}
                title={`step ${s.step}: p=${s.chosen_prob} — click to inspect this exact prefix`}
                style={{ color: "var(--accent)", textDecoration: "none" }}
              >
                {s.chosen}
              </a>
            ))}
          </p>
          <p className="inspector-meta">
            {contResult.steps.map((s) => (
              <span key={s.step} style={{ marginRight: "0.9rem" }}>
                step {s.step}: {JSON.stringify(s.chosen)} p={s.chosen_prob.toFixed(3)}
              </span>
            ))}
            <br />
            Click any generated token above to inspect that exact prefix in the Model Explorer.
          </p>
        </div>
      )}
    </div>
  );
}
