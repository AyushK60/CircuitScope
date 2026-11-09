import { useEffect, useRef, useState } from "react";
import {
  api,
  type ContinueResult,
  type ExperimentRecord,
  type Health,
  type Prediction,
  type QuestionResult,
  type SavedExperiment,
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

/** Patchable hooks offered — all verified in the backend allowlist
 *  (PATCHABLE_HOOKS). Generated so the UI can never desync from state. */
const HOOK_OPTIONS = [
  ...Array.from({ length: 12 }, (_, l) => ({
    value: `blocks.${l}.hook_out`,
    label: `Block ${l} · residual stream${l === 5 ? " (recommended)" : ""}`,
  })),
  ...Array.from({ length: 12 }, (_, l) => ({
    value: `blocks.${l}.attn.hook_out`,
    label: `Block ${l} · attention output`,
  })),
  { value: "blocks.5.attn.hook_z", label: "Block 5 · attention z (per-head)" },
  ...[5, 9].map((l) => ({
    value: `blocks.${l}.mlp.hook_out`,
    label: `Block ${l} · MLP output`,
  })),
];

type RunState = "idle" | "running" | "complete" | "failed" | "cancelled";

/** Reads #/playground?hook=...&position=... so the Explorer can carry a
 *  selected component into the laboratory. */
function readParams(): { hook?: string; position?: number } {
  const q = window.location.hash.split("?")[1];
  if (!q) return {};
  const p = new URLSearchParams(q);
  const position = p.get("position");
  return {
    hook: p.get("hook") ?? undefined,
    position: position != null && !Number.isNaN(Number(position)) ? Number(position) : undefined,
  };
}

/** Mini architecture strip: where in the real 12-block pipeline this patch sits. */
function TargetStrip({ hook, position, tokens }: { hook: string; position: number; tokens: string[] }) {
  const match = hook.match(/blocks\.(\d+)\./);
  const layer = match ? Number(match[1]) : null;
  const kind = hook.includes("mlp") ? "MLP" : hook.includes("attn") ? "attention" : "residual stream";
  return (
    <div className="target-strip">
      <svg viewBox="0 0 720 74" width="100%" role="img" aria-label={`Patch target: ${hook} at token position ${position}`}>
        {/* residual rail */}
        <path d="M 16 20 L 704 20" stroke="var(--violet)" strokeDasharray="4 4" strokeWidth={2} fill="none" />
        <text x={16} y={12} fill="var(--violet)" fontSize={9}>
          residual stream
        </text>
        {Array.from({ length: 12 }, (_, l) => {
          const x = 20 + l * 57;
          const isTarget = layer === l;
          return (
            <g key={l}>
              <path d={`M ${x + 24} 20 L ${x + 24} 34`} stroke="var(--violet)" strokeDasharray="3 3" fill="none" opacity={0.6} />
              <rect
                x={x}
                y={34}
                width={48}
                height={28}
                rx={5}
                fill={isTarget ? "var(--bg-1)" : "var(--bg-3)"}
                stroke={isTarget ? "var(--accent)" : "var(--line-1)"}
                strokeWidth={isTarget ? 2 : 1}
              />
              <text x={x + 24} y={52} fill={isTarget ? "var(--accent)" : "var(--text-2)"} fontSize={10} textAnchor="middle">
                B{l}
              </text>
              {isTarget && (
                <text x={x + 24} y={72} fill="var(--accent)" fontSize={8.5} textAnchor="middle">
                  {kind} · patch here
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {tokens.length > 0 && (
        <div className="token-row" aria-label="Patch position on the clean tokens">
          {tokens.map((t, i) => (
            <span key={i} className={`token${i === position ? " hl" : ""}`} title={`position ${i}`}>
              {t}
              <span className="idx">{i}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Playground({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const params = useRef(readParams()).current;
  const [clean, setClean] = useState(PRESET.clean);
  const [corrupted, setCorrupted] = useState(PRESET.corrupted);
  const [hook, setHook] = useState(params.hook ?? "blocks.5.hook_out");
  const [position, setPosition] = useState(params.position ?? PRESET.s2_position);
  const [tokenization, setTokenization] = useState<Tokenization | null>(null);
  const [baseline, setBaseline] = useState<Prediction | null>(null);

  const [candA, setCandA] = useState(PRESET.candidate_a);
  const [candB, setCandB] = useState(PRESET.candidate_b);
  const [comparison, setComparison] = useState<{ D: number; prob_a: number; prob_b: number } | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [contResult, setContResult] = useState<ContinueResult | null>(null);
  const [contBusy, setContBusy] = useState(false);
  const [contMode, setContMode] = useState<"greedy" | "sample">("greedy");
  const [contTemp, setContTemp] = useState(0.9);
  const [contTopK, setContTopK] = useState(20);
  const [contSeed, setContSeed] = useState(1234);
  const [wizardStep, setWizardStep] = useState(0);
  const [history, setHistory] = useState<SavedExperiment[]>([]);
  const [qContext, setQContext] = useState("The father asked his son for water because the father was sick.");
  const [qQuestion, setQQuestion] = useState("Who was sick?");
  const [qCandidates, setQCandidates] = useState(" father, son");
  const [qResult, setQResult] = useState<QuestionResult | null>(null);
  const [qBusy, setQBusy] = useState(false);
  const [qError, setQError] = useState<string | null>(null);
  const [delConfirm, setDelConfirm] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<{ id: string; delta: number | null; recovery: number | null }[]>([]);

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
        .predict(clean, 6, candA, candB)
        .then((r) => {
          setTokenization(r.tokenization);
          setBaseline(r.prediction);
          setComparison(r.comparison ? { D: r.comparison.D, prob_a: r.comparison.prob_a, prob_b: r.comparison.prob_b } : null);
          setComparisonError(r.comparison_error ?? null);
        })
        .catch(() => {});
    }, 350);
    return () => clearTimeout(t);
  }, [clean, candA, candB, ready]);

  // poll experiment status while active
  useEffect(() => {
    if (!jobId || (runState !== "running")) return;
    pollRef.current = window.setInterval(() => {
      api
        .experiment(jobId)
        .then((r) => {
          setRecord(r);
          if (r.status === "complete") {
            setRunState("complete");
            api.listExperiments().then((lr) => setHistory(lr.saved.filter((x) => x.status === "complete").slice(0, 10))).catch(() => {});
          }
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
        candidate_a: candA,
        candidate_b: candB,
      })
      .then((r) => setJobId(r.job_id))
      .catch((e) => {
        setError(String(e.message ?? e));
        setRunState("failed");
      });
  };

  const refreshHistory = () => {
    api
      .listExperiments()
      .then((r) => setHistory(r.saved.filter((x) => x.status === "complete").slice(0, 10)))
      .catch(() => {});
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refreshHistory, [ready]);

  const loadSaved = (id: string) => {
    api
      .getSavedExperiment(id)
      .then((r) => {
        if (r.results) {
          setRecord({ ...r, status: "complete" as const, error: r.error ?? null });
          setJobId(id);
          setRunState("complete");
        }
      })
      .catch((e) => setError(String(e.message ?? e)));
  };

  const runQuestion = () => {
    setQBusy(true);
    setQError(null);
    api
      .question({
        context: qContext,
        question: qQuestion,
        candidates: qCandidates.split(",").map((c) => (c.startsWith(" ") ? c : " " + c.trim())).filter((c) => c.length > 1),
      })
      .then(setQResult)
      .catch((e) => setQError(String(e.message ?? e)))
      .finally(() => setQBusy(false));
  };

  const toggleCompare = (id: string) => {
    setCompareIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : ids.length >= 2 ? [ids[1], id] : [...ids, id]));
  };

  const runCompare = () => {
    Promise.all(compareIds.map((id) => api.getSavedExperiment(id)))
      .then((rs) =>
        setCompareData(
          rs.map((r) => ({
            id: r.id,
            delta: r.results?.metrics?.Delta ?? null,
            recovery: r.results?.metrics?.Recovery ?? null,
          })),
        ),
      )
      .catch((e) => setError(String(e.message ?? e)));
  };

  const deleteSaved = (id: string) => {
    api
      .deleteExperiment(id)
      .then(() => {
        setDelConfirm(null);
        refreshHistory();
        if (jobId === id) {
          setRecord(null);
          setRunState("idle");
        }
      })
      .catch((e) => setError(String(e.message ?? e)));
  };

  const runContinuation = (nTokens: number) => {
    setContBusy(true);
    const base = contResult?.text_final ?? clean;
    api
      .continue(base, nTokens, {
        mode: contMode,
        temperature: contTemp,
        top_k: contTopK,
        seed: contSeed,
      })
      .then((r) => {
        if (contResult && base !== clean) {
          // step mode: append the new tokens to the displayed run
          setContResult({
            ...r,
            text_start: contResult.text_start,
            steps: [...contResult.steps, ...r.steps.map((s) => ({ ...s, step: contResult.steps.length + s.step }))],
          });
        } else {
          setContResult(r);
        }
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setContBusy(false));
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

      {ready && (
        <div className="panel" style={{ borderColor: "var(--brass)" }}>
          <h2>Guided experiment tour</h2>
          <p className="inspector-meta" style={{ marginBottom: "0.6rem" }}>
            Curated examples — each configures the real controls below and runs actual model actions.
            Nothing is simulated, and measured outcomes may differ from any expectation.
          </p>
          <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginBottom: "0.8rem" }}>
            {[
              { id: "ramu", label: "① What comes next?", clean: "Ramu was hungry. He went outside to buy", corr: "Ramu was hungry. He went outside to buy", cand: [" food", " bread"] },
              { id: "father", label: "② Who was sick?", clean: "The father asked his son for water because the father was sick. The father was sick, so the", corr: "The father asked his son for water because the father was sick. The father was sick, so the", cand: [" father", " son"] },
              { id: "ioi", label: "③ Activation patch", clean: PRESET.clean, corr: PRESET.corrupted, cand: [PRESET.candidate_a, PRESET.candidate_b] },
            ].map((ex) => (
              <button
                key={ex.id}
                onClick={() => {
                  setClean(ex.clean);
                  setCorrupted(ex.corr);
                  setCandA(ex.cand[0]);
                  setCandB(ex.cand[1]);
                  setWizardStep(0);
                  setRunState("idle");
                  setRecord(null);
                }}
              >
                {ex.label}
              </button>
            ))}
            <button
              onClick={() => {
                window.location.hash = "#/explorer?mode=flow";
              }}
            >
              ④ Attention inspection →
            </button>
          </div>
          <p className="inspector-meta" style={{ marginBottom: "0.8rem" }}>
            A four-step walkthrough of one complete intervention. Each step uses the same real controls
            below — nothing is simulated.
          </p>
          <div className="wizard-steps">
            {[
              { n: 1, label: "Predict", done: comparison != null || comparisonError != null },
              { n: 2, label: "Inspect", done: wizardStep > 1 },
              { n: 3, label: "Patch", done: runState === "complete" },
              { n: 4, label: "Compare", done: runState === "complete" && wizardStep > 3 },
            ].map((s, i) => (
              <button
                key={s.n}
                className={`wizard-step${wizardStep === i ? " active" : s.done ? " done" : ""}`}
                onClick={() => {
                  setWizardStep(i);
                  if (i === 3 && runState !== "complete") runExperiment();
                  if (i === 0) {
                    api.predict(clean, 6, candA, candB).catch(() => {});
                  }
                }}
              >
                <span className="wizard-num">{s.n}</span> {s.label}
              </button>
            ))}
          </div>
          {wizardStep === 0 && (
            <p className="inspector-meta">
              Step 1 — the live comparison below (updated as you type) shows the model's measured
              preference between candidates A and B on the clean prompt. When you can read D, continue.
            </p>
          )}
          {wizardStep === 1 && (
            <p className="inspector-meta">
              Step 2 — open the block you plan to patch in the Model Explorer to see its real attention
              patterns:{" "}
              <a href={`#/explorer?mode=flow`}>
                <button>Inspect block {parseInt(hook.replace(/[^0-9]/g, "")) || 5} →</button>
              </a>
            </p>
          )}
          {wizardStep === 2 && (
            <p className="inspector-meta">
              Step 3 — run the patch: three real forward passes (clean capture, corrupted, patched).{" "}
              {clean === corrupted
                ? "⚠ This example is prediction-focused: the clean and corrupted prompts are identical, so a patch would have no contrast. Switch to example ③ for a valid patch pair."
                : 'Press "Run experiment" below if you have not already.'}
            </p>
          )}
          {wizardStep === 3 && (
            <p className="inspector-meta">
              Step 4 — read the comparison: clean vs corrupted vs patched D, the patch effect Δ, and the
              unclamped recovery. Include no-effect results as they measure.
            </p>
          )}
        </div>
      )}

      <div className="panel">
        <h2>1 · Choose the experiment</h2>
        <p className="inspector-meta" style={{ marginBottom: "0.8rem" }}>
          Intervention target in the real pipeline — the highlighted block is where the clean activation
          will replace the corrupted one:
        </p>
        <TargetStrip
          hook={hook}
          position={position}
          tokens={tokenization?.tokens ?? []}
        />
        <div className="grid-2" style={{ marginTop: "1rem" }}>
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
            <div className="grid-2" style={{ marginTop: "0.8rem" }}>
              <div>
                <label htmlFor="cand-a" style={{ fontSize: "0.8rem", color: "var(--text-2)" }}>
                  Candidate A
                </label>
                <input id="cand-a" type="text" value={candA} onChange={(e) => setCandA(e.target.value)} />
              </div>
              <div>
                <label htmlFor="cand-b" style={{ fontSize: "0.8rem", color: "var(--text-2)" }}>
                  Candidate B
                </label>
                <input id="cand-b" type="text" value={candB} onChange={(e) => setCandB(e.target.value)} />
              </div>
            </div>
            <p className="inspector-meta" style={{ marginTop: "0.4rem" }}>
              Validated as single tokens by the backend (leading spaces matter). Live comparison below
              updates as you type.
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
            {comparisonError && (
              <div className="state-banner warn" style={{ marginTop: "0.8rem" }}>
                Candidate comparison unavailable: {comparisonError}
              </div>
            )}
            {comparison && (
              <div className="metric-cards" style={{ marginTop: "0.8rem" }}>
                <div className="metric-card">
                  <div className="name">D · logit(A) − logit(B)</div>
                  <div className="value">{comparison.D.toFixed(3)}</div>
                  <div className="sub">positive = A preferred at the final position</div>
                </div>
                <div className="metric-card">
                  <div className="name">P(A) · {JSON.stringify(candA)}</div>
                  <div className="value">{(comparison.prob_a * 100).toFixed(2)}%</div>
                  <div className="sub">measured probability on the clean prompt</div>
                </div>
                <div className="metric-card">
                  <div className="name">P(B) · {JSON.stringify(candB)}</div>
                  <div className="value">{(comparison.prob_b * 100).toFixed(2)}%</div>
                  <div className="sub">measured probability on the clean prompt</div>
                </div>
              </div>
            )}
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

      <div className="panel">
        <h2>Question &amp; Candidate Explorer</h2>
        <p className="inspector-meta" style={{ marginBottom: "0.8rem" }}>
          A context, a question, and candidate answers. GPT-2 is a next-token model, <strong>not</strong> a
          question-answering assistant — this compares text-continuation scores under a fixed prompt
          template. Multi-token candidates are scored by the sum of their conditional token
          log-probabilities, never by the first token alone.
        </p>
        <div className="grid-2">
          <div>
            <label htmlFor="q-context" style={{ fontSize: "0.8rem", color: "var(--text-2)" }}>Context</label>
            <textarea id="q-context" rows={2} value={qContext} onChange={(e) => setQContext(e.target.value)} />
            <label htmlFor="q-question" style={{ fontSize: "0.8rem", color: "var(--text-2)", display: "block", marginTop: "0.5rem" }}>Question</label>
            <input id="q-question" type="text" value={qQuestion} onChange={(e) => setQQuestion(e.target.value)} />
            <label htmlFor="q-cands" style={{ fontSize: "0.8rem", color: "var(--text-2)", display: "block", marginTop: "0.5rem" }}>
              Candidates (comma-separated, 2–5)
            </label>
            <input id="q-cands" type="text" value={qCandidates} onChange={(e) => setQCandidates(e.target.value)} />
            <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.8rem", flexWrap: "wrap" }}>
              <button className="primary" onClick={runQuestion} disabled={!ready || qBusy}>
                {qBusy ? "Scoring…" : "Score candidates"}
              </button>
              <button onClick={() => { setQContext("The father asked his son for water because the father was sick."); setQQuestion("Who was sick?"); setQCandidates(" father, son"); setQResult(null); }}>
                Father/son example
              </button>
            </div>
            {qError && <div className="state-banner error" style={{ marginTop: "0.6rem" }}>{qError}</div>}
          </div>
          <div>
            {qResult && (
              <>
                <p className="inspector-meta mono" style={{ background: "var(--bg-2)", padding: "0.5rem 0.7rem", borderRadius: 6 }}>
                  template: {qResult.prompt.split("\n").join(" ⏎ ")}
                </p>
                <div className="bar-row" key="hdr" style={{ marginTop: "0.6rem" }}>
                  <span className="label" style={{ color: "var(--text-2)" }}>candidate</span>
                  <span className="inspector-meta">total log-P · mean · first-token P</span>
                </div>
                {qResult.candidates.map((c) => {
                  const maxLP = Math.max(...qResult.candidates.map((x) => x.total_logprob));
                  const minLP = Math.min(...qResult.candidates.map((x) => x.total_logprob));
                  const frac = (c.total_logprob - minLP) / Math.max(maxLP - minLP, 1e-9);
                  return (
                    <div className="bar-row" key={c.candidate}>
                      <span className="label">{c.candidate}</span>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ left: 0, width: `${8 + frac * 92}%`, background: c.candidate === qResult.preferred_by_total_logprob ? "var(--accent)" : "var(--bg-3)" }} />
                      </div>
                      <span className="value">{c.total_logprob.toFixed(2)}</span>
                      <span className="inspector-meta">
                        mean {c.mean_logprob.toFixed(2)} · first-P {(c.first_token_prob * 100).toFixed(2)}% · {c.n_tokens} tok
                      </span>
                    </div>
                  );
                })}
                <p className="inspector-meta" style={{ marginTop: "0.6rem" }}>
                  Measured preference by total log-probability:{" "}
                  <strong>{JSON.stringify(qResult.preferred_by_total_logprob)}</strong>.{" "}
                  {qResult.length_warning
                    ? "⚠ candidates have different token lengths — compare mean log-P as well; raw totals favor longer candidates."
                    : "Candidates have equal token lengths; total and mean log-P rank them identically."}{" "}
                  These are continuation scores — not a claim that GPT-2 answers questions correctly.
                </p>
              </>
            )}
            {!qResult && (
              <div className="state-banner info">
                Enter a context, question and candidates, then score. Scores are conditional log-probabilities
                under the fixed template — see the method note in the result.
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="panel">
        <h2>Experiment history</h2>
        <p className="inspector-meta" style={{ marginBottom: "0.8rem" }}>
          Completed experiments persisted in the local database. Loading one restores its exact stored
          results (config, metrics, provenance).
        </p>
        {history.length === 0 ? (
          <p className="inspector-meta">No completed experiments yet — run one above.</p>
        ) : (
          <div className="table-scroll">
          <table className="kv-table">
            <tbody>
              {history.map((h) => (
                <tr key={h.id}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Select experiment ${h.id} for comparison`}
                      checked={compareIds.includes(h.id)}
                      onChange={() => toggleCompare(h.id)}
                    />
                  </td>
                  <td>{new Date(h.created_at * 1000).toLocaleTimeString()}</td>
                  <td>
                    {(h.config as Record<string, unknown>).hook
                      ? `intervention · ${String((h.config as Record<string, unknown>).hook)}`
                      : "candidate comparison"}
                  </td>
                  <td>
                    pos {JSON.stringify((h.config as Record<string, unknown>).positions ?? [])}
                  </td>
                  <td>
                    <button onClick={() => loadSaved(h.id)}>Load</button>{" "}
                    {delConfirm === h.id ? (
                      <>
                        <button className="danger" onClick={() => deleteSaved(h.id)}>
                          Confirm delete
                        </button>
                        <button className="ghost" onClick={() => setDelConfirm(null)}>
                          ✕
                        </button>
                      </>
                    ) : (
                      <button className="danger" onClick={() => setDelConfirm(h.id)}>
                        Delete
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
        {compareIds.length === 2 && (
          <div style={{ marginTop: "0.8rem", display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" }}>
            <button onClick={runCompare}>Compare selected (Δ / Recovery)</button>
            {compareData.map((c) => (
              <span key={c.id} className="inspector-meta mono">
                {c.id.slice(0, 6)}: Δ {c.delta?.toFixed(3) ?? "n/a"} · rec {c.recovery?.toFixed(3) ?? "n/a"}
              </span>
            ))}
          </div>
        )}
      </div>

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
