import { useEffect, useRef, useState } from "react";
import {
  api,
  type ExperimentRecord,
  type Health,
  type Prediction,
  type Tokenization,
} from "../api/client";
import TokenRow from "../components/TokenRow";
import TopPredictions from "../components/TopPredictions";

const PRESET = {
  id: "ioi_john_mary",
  name: "IOI · John and Mary",
  clean: "John and Mary went to the shop. John gave a book to",
  corrupted: "John and Mary went to the shop. Mary gave a book to",
  candidate_a: " Mary",
  candidate_b: " John",
  s2_position: 9,
};

/** Patchable hooks offered, grouped for clarity. The allowlist is enforced by
 *  the backend; these are the verified-supported options. */
const HOOK_OPTIONS = [
  { value: "blocks.5.hook_out", label: "Block 5 · residual stream (recommended)" },
  { value: "blocks.6.hook_out", label: "Block 6 · residual stream" },
  { value: "blocks.7.hook_out", label: "Block 7 · residual stream" },
  { value: "blocks.8.hook_out", label: "Block 8 · residual stream" },
  { value: "blocks.9.hook_out", label: "Block 9 · residual stream" },
  { value: "blocks.5.attn.hook_out", label: "Block 5 · attention output" },
  { value: "blocks.5.attn.hook_z", label: "Block 5 · attention z (per-head)" },
  { value: "blocks.9.attn.hook_out", label: "Block 9 · attention output" },
  { value: "blocks.5.mlp.hook_out", label: "Block 5 · MLP output" },
];

type RunState = "idle" | "running" | "complete" | "failed" | "cancelled";

export default function Playground({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [clean, setClean] = useState(PRESET.clean);
  const [corrupted, setCorrupted] = useState(PRESET.corrupted);
  const [hook, setHook] = useState("blocks.5.hook_out");
  const [position, setPosition] = useState(PRESET.s2_position);
  const [tokenization, setTokenization] = useState<Tokenization | null>(null);
  const [baseline, setBaseline] = useState<Prediction | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [record, setRecord] = useState<ExperimentRecord | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  // live tokenization + baseline for the clean prompt
  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(() => {
      api
        .predict(clean, 6)
        .then((r) => {
          setTokenization(r.tokenization);
          setBaseline(r.prediction);
        })
        .catch(() => {});
    }, 350);
    return () => clearTimeout(t);
  }, [clean, ready]);

  // poll experiment status while active
  useEffect(() => {
    if (!jobId || (runState !== "running")) return;
    pollRef.current = window.setInterval(() => {
      api
        .experiment(jobId)
        .then((r) => {
          setRecord(r);
          if (r.status === "complete") setRunState("complete");
          if (r.status === "failed" || r.status === "cancelled") setRunState(r.status);
        })
        .catch((e) => {
          setError(String(e.message ?? e));
          setRunState("failed");
        });
    }, 400);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId, runState]);

  const runExperiment = () => {
    setError(null);
    setRunState("running");
    setRecord(null);
    api
      .startExperiment({
        clean_text: clean,
        corrupted_text: corrupted,
        hook,
        positions: [position],
        heads: hook.endsWith("hook_z") ? null : null,
        candidate_a: PRESET.candidate_a,
        candidate_b: PRESET.candidate_b,
      })
      .then((r) => setJobId(r.job_id))
      .catch((e) => {
        setError(String(e.message ?? e));
        setRunState("failed");
      });
  };

  const res = record?.results ?? null;
  const m = res?.metrics ?? null;

  return (
    <main className="page">
      <h1>Intervention Playground</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "75ch" }}>
        This workbench runs three <em>real</em> model passes: the clean prompt, a corrupted variant, and a
        patched run that copies one internal activation from the clean run into the corrupted run. Every
        number below is measured from those passes.
      </p>

      {!ready && (
        <div className="state-banner info" style={{ marginBottom: "1rem" }}>
          <span className="spinner" aria-hidden /> Waiting for the model. Experiments need the loaded weights.
        </div>
      )}

      <div className="panel">
        <h2>1 · Choose the experiment</h2>
        <div className="grid-2">
          <div>
            <label htmlFor="clean-text" style={{ fontSize: "0.85rem", color: "var(--text-2)" }}>
              Clean prompt
            </label>
            <textarea id="clean-text" rows={2} value={clean} onChange={(e) => setClean(e.target.value)} />
            <label
              htmlFor="corr-text"
              style={{ fontSize: "0.85rem", color: "var(--text-2)", display: "block", marginTop: "0.6rem" }}
            >
              Corrupted prompt (same length, subject swapped)
            </label>
            <textarea
              id="corr-text"
              rows={2}
              value={corrupted}
              onChange={(e) => setCorrupted(e.target.value)}
            />
            <p className="inspector-meta">
              Candidates: <span className="mono">{PRESET.candidate_a}</span> (A) vs{" "}
              <span className="mono">{PRESET.candidate_b}</span> (B) — validated as single tokens by the
              backend before every run.
            </p>
          </div>
          <div>
            <label htmlFor="hook-sel" style={{ fontSize: "0.85rem", color: "var(--text-2)" }}>
              Activation to replace (clean → corrupted)
            </label>
            <select id="hook-sel" value={hook} onChange={(e) => setHook(e.target.value)}>
              {HOOK_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
            <label
              htmlFor="pos-sel"
              style={{ fontSize: "0.85rem", color: "var(--text-2)", display: "block", marginTop: "0.6rem" }}
            >
              Token position to patch
            </label>
            <select
              id="pos-sel"
              value={position}
              onChange={(e) => setPosition(Number(e.target.value))}
            >
              {(tokenization?.tokens ?? []).map((t, i) => (
                <option key={i} value={i}>
                  {i}: {JSON.stringify(t)}
                </option>
              ))}
            </select>
            <h3 style={{ marginTop: "0.9rem" }}>Clean tokens (positions shown)</h3>
            <TokenRow tok={tokenization} highlightPositions={[position]} />
            <div style={{ marginTop: "0.9rem", display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
              <button className="primary" onClick={runExperiment} disabled={!ready || runState === "running"}>
                {runState === "running" ? "Running 3 model passes…" : "Run experiment (clean + corrupted + patched)"}
              </button>
              {runState === "running" && jobId && (
                <button onClick={() => api.cancelExperiment(jobId)}>Cancel</button>
              )}
              <button
                onClick={() => {
                  setClean(PRESET.clean);
                  setCorrupted(PRESET.corrupted);
                  setHook("blocks.5.hook_out");
                  setPosition(PRESET.s2_position);
                  setRunState("idle");
                  setRecord(null);
                  setJobId(null);
                }}
              >
                Restore preset
              </button>
            </div>
          </div>
        </div>
      </div>

      {runState === "running" && (
        <div className="panel">
          <div className="state-banner info">
            <span className="spinner" aria-hidden /> Executing the real model three times (clean capture,
            corrupted, patched). This takes a few seconds on CPU.
          </div>
        </div>
      )}

      {runState === "failed" && (
        <div className="panel">
          <div className="state-banner error" role="alert">
            <strong>Experiment failed:</strong> {error ?? record?.error} — no results are shown because none
            were produced. Check that the two prompts tokenize to the same length and the position is valid.
          </div>
        </div>
      )}

      {runState === "complete" && res && m && (
        <>
          <div className="panel">
            <h2>2 · Measured effect</h2>
            <div className="metric-cards">
              <MetricCard
                name="D · logit(A) − logit(B)"
                value={m.D_patched?.toFixed(3) ?? "—"}
                sub={`clean ${m.D_clean?.toFixed(3)} · corrupted ${m.D_corrupted?.toFixed(3)}`}
              />
              <MetricCard
                name="Δ effect of patch"
                value={m.Delta?.toFixed(3) ?? "—"}
                sub="D_patched − D_corrupted"
              />
              <MetricCard
                name="Recovery"
                value={m.Recovery != null ? m.Recovery.toFixed(3) : "n/a"}
                sub={m.Recovery_note ?? "(D_p − D_corr)/(D_cl − D_corr), unclamped"}
              />
            </div>
            <p className="inspector-meta">
              {m.Recovery != null && Math.abs(m.Recovery) < 0.05
                ? "The intervention had a negligible measured effect on this example — reported as measured."
                : m.Recovery != null && m.Recovery > 0.5
                  ? "The patched activation carried most of the clean-vs-corrupted information for this metric."
                  : "Sign convention: positive D means candidate A scores above candidate B at the final position."}
            </p>
          </div>

          <div className="panel">
            <h2>3 · Compare all three runs</h2>
            <div className="grid-3">
              <div className="run-column clean">
                <h4>Clean run · {res.config.clean_text.length > 30 ? "…" : ""}{res.config.clean_text.slice(-28)}</h4>
                <TopPredictions candidates={res.runs.clean} metric="prob" maxRows={5} highlight={[res.config.candidate_a, res.config.candidate_b]} />
              </div>
              <div className="run-column corrupted">
                <h4>Corrupted run · …{res.config.corrupted_text.slice(-28)}</h4>
                <TopPredictions candidates={res.runs.corrupted} metric="prob" maxRows={5} highlight={[res.config.candidate_a, res.config.candidate_b]} />
              </div>
              <div className="run-column patched">
                <h4>Patched run · {res.patch.hook} @ pos {res.patch.positions.join(",")}</h4>
                <TopPredictions candidates={res.runs.patched} metric="prob" maxRows={5} highlight={[res.config.candidate_a, res.config.candidate_b]} />
              </div>
            </div>
            <div className="grid-2" style={{ marginTop: "1rem", alignItems: "start" }}>
              <div>
                <h3>Corrupted tokens (patched position highlighted)</h3>
                <TokenRow
                  tok={{ tokens: res.tokens_corrupted, token_ids: [], n_tokens: res.tokens_corrupted.length }}
                  highlightPositions={res.patch.positions}
                />
              </div>
              <div>
                <h3>Experiment record</h3>
                <table className="kv-table">
                  <tbody>
                    <tr><td>Experiment ID</td><td>{record?.id}</td></tr>
                    <tr><td>Model</td><td>{res.provenance.model_id} · {res.provenance.device} · torch {res.provenance.torch}</td></tr>
                    <tr><td>Instrumentation</td><td>{res.provenance.library}</td></tr>
                    <tr><td>Patch source</td><td>{res.patch.source}</td></tr>
                    <tr><td>Elapsed</td><td>{res.provenance.elapsed_seconds.toFixed(2)} s (3 runs)</td></tr>
                    <tr><td>Candidates</td><td>A {JSON.stringify(res.config.candidate_a)} · B {JSON.stringify(res.config.candidate_b)}</td></tr>
                  </tbody>
                </table>
                <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.8rem" }}>
                  <a href={api.exportUrl(record!.id, "json")} target="_blank" rel="noreferrer">
                    <button>Export JSON</button>
                  </a>
                  <a href={api.exportUrl(record!.id, "csv")} target="_blank" rel="noreferrer">
                    <button>Export CSV</button>
                  </a>
                  <button
                    onClick={() => {
                      setRunState("idle");
                      setRecord(null);
                      setJobId(null);
                    }}
                  >
                    Restore baseline
                  </button>
                </div>
              </div>
            </div>
            <p className="inspector-meta" style={{ marginTop: "0.8rem" }}>
              <strong>What this establishes (and what it does not):</strong> replacing this activation at
              these positions changed the model's measured preference under this specific intervention.
              It is evidence about this example and hook — not a proof that one component "understands"
              anything, and not a statistical claim over other inputs.
            </p>
          </div>
        </>
      )}

      {runState === "cancelled" && (
        <div className="panel">
          <div className="state-banner warn">Experiment cancelled before completion. Nothing was saved.</div>
        </div>
      )}

      <div className="panel">
        <h2>Baseline (live, clean prompt)</h2>
        <p className="inspector-meta">Independent live prediction of the clean prompt for reference.</p>
        <TopPredictions candidates={baseline?.top_predictions ?? null} metric="prob" maxRows={6} />
      </div>
    </main>
  );
}

function MetricCard({ name, value, sub }: { name: string; value: string; sub: string }) {
  return (
    <div className="metric-card">
      <div className="name">{name}</div>
      <div className="value">{value}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}
