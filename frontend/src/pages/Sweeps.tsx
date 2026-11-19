import { useEffect, useRef, useState } from "react";
import {
  api,
  type AgreementJobStatus,
  type AgreementResult,
  type DlaResult,
  type Health,
  type LensResult,
  type SweepJobStatus,
  type SweepResult,
  type TaskSummary,
} from "../api/client";
import SweepHeatmap from "../components/SweepHeatmap";
import AgreementScatter from "../components/AgreementScatter";
import ReproduceButton from "../components/ReproduceButton";
import PlainWords from "./report/PlainWords";

/**
 * Sweeps page: patching sweeps (denoising), direct logit attribution (DLA)
 * and the logit lens. Every number comes from an API response computed on the
 * real model; sections state what they can and cannot prove.
 */

const NEG = [201, 106, 94]; // corrupted / negative
const ZERO = [43, 48, 45]; // zero
const POS = [145, 170, 145]; // sage / positive

function mix(a: number[], b: number[], t: number): string {
  const c = a.map((av, i) => Math.round(av + (b[i] - av) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function diverging(v: number, maxAbs: number): string {
  if (!Number.isFinite(v)) return "var(--bg-3)";
  return mix(ZERO, v >= 0 ? POS : NEG, Math.min(1, Math.abs(v) / (maxAbs || 1)));
}

export default function Sweeps({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [taskId, setTaskId] = useState<string>("");
  const [mode, setMode] = useState<"layer_position" | "layer_head">("layer_position");

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<SweepJobStatus | null>(null);
  const [sweepError, setSweepError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const [dlaClean, setDlaClean] = useState<DlaResult | null>(null);
  const [dlaCorrupted, setDlaCorrupted] = useState<DlaResult | null>(null);
  const [dlaBusy, setDlaBusy] = useState<"clean" | "corrupted" | null>(null);
  const [dlaError, setDlaError] = useState<string | null>(null);

  const [lensText, setLensText] = useState("");
  const [lensA, setLensA] = useState("");
  const [lensB, setLensB] = useState("");
  const [lensResult, setLensResult] = useState<LensResult | null>(null);
  const [lensBusy, setLensBusy] = useState(false);
  const [lensError, setLensError] = useState<string | null>(null);

  const [agreeJobId, setAgreeJobId] = useState<string | null>(null);
  const [agreeJob, setAgreeJob] = useState<AgreementJobStatus | null>(null);
  const [agreeError, setAgreeError] = useState<string | null>(null);

  useEffect(() => {
    api
      .tasks()
      .then((r) => {
        setTasks(r.tasks);
        if (r.tasks.length > 0) setTaskId((cur) => cur || r.tasks[0].id);
      })
      .catch((e) => setSweepError(String(e.message ?? e)));
  }, []);

  // default the lens text to the selected task's clean prompt
  useEffect(() => {
    if (!taskId) return;
    api
      .task(taskId)
      .then((t) => {
        const clean = t.pairs[0]?.clean_text ?? "";
        setLensText((cur) => (cur === "" ? clean : cur));
      })
      .catch(() => {});
  }, [taskId]);

  // poll sweep job status while running
  useEffect(() => {
    if (!jobId) return;
    pollRef.current = window.setInterval(() => {
      api
        .jobStatus(jobId)
        .then((r) => {
          setJob(r);
          if (r.status === "failed") setSweepError(r.error ?? "sweep failed");
        })
        .catch((e) => setSweepError(String(e.message ?? e)));
    }, 400);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId]);

  const sweepResult: SweepResult | null = job?.status === "complete" ? job.results : null;
  const running = job?.status === "running" || job?.status === "queued";

  useEffect(() => {
    if (!agreeJobId) return;
    const id = window.setInterval(() => {
      api
        .agreementStatus(agreeJobId)
        .then((r) => {
          setAgreeJob(r);
          if (r.status === "failed") setAgreeError(r.error ?? "agreement failed");
        })
        .catch((e) => setAgreeError(String(e.message ?? e)));
    }, 400);
    return () => clearInterval(id);
  }, [agreeJobId]);

  const runSweep = () => {
    setSweepError(null);
    setJob(null);
    setJobId(null);
    api
      .startSweep({ task_id: taskId, mode })
      .then((r) => setJobId(r.job_id))
      .catch((e) => setSweepError(String(e.message ?? e)));
  };
  const cancelSweep = () => {
    if (jobId) api.cancelExperiment(jobId).catch(() => {});
  };

  const runDla = (which: "clean" | "corrupted") => {
    setDlaBusy(which);
    setDlaError(null);
    api
      .dla(taskId, null, which)
      .then((r) => (which === "clean" ? setDlaClean(r) : setDlaCorrupted(r)))
      .catch((e) => setDlaError(String(e.message ?? e)))
      .finally(() => setDlaBusy(null));
  };

  const runLens = () => {
    setLensBusy(true);
    setLensError(null);
    api
      .logitLens(lensText, lensA || null, lensB || null, 5)
      .then(setLensResult)
      .catch((e) => setLensError(String(e.message ?? e)))
      .finally(() => setLensBusy(false));
  };

  const runAgreement = () => {
    setAgreeError(null);
    setAgreeJob(null);
    setAgreeJobId(null);
    api
      .startAgreement({ task_id: taskId, pair_id: null })
      .then((r) => setAgreeJobId(r.job_id))
      .catch((e) => setAgreeError(String(e.message ?? e)));
  };

  const agreementResult: AgreementResult | null =
    agreeJob?.status === "complete" ? agreeJob.results : null;
  const agreeRunning = agreeJob?.status === "running" || agreeJob?.status === "queued";
  const sharedLabels = agreementResult
    ? new Set(
        agreementResult.top10.exact
          .map((e) => e.label)
          .filter(
            (label) =>
              agreementResult.top10.atp.some((x) => x.label === label) &&
              agreementResult.top10.dla.some((x) => x.label === label),
          ),
      )
    : new Set<string>();

  return (
    <main className="page">
      <h1>Sweeps</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "75ch" }}>
        Three measurements on the real model: activation-patching sweeps over layers × positions/heads,
        direct logit attribution per component, and the logit lens. Patching direction is fixed:
        clean activations are copied into the corrupted run (denoising).
      </p>

      {!ready && (
        <div className="state-banner info" style={{ marginBottom: "1rem" }}>
          <span className="spinner" aria-hidden /> Waiting for the model. These measurements need the loaded weights.
        </div>
      )}

      {/* ---- 1. patching sweep ---- */}
      <section className="panel">
        <h2>Patching sweep</h2>
        <p className="inspector-meta">
          What you are looking at: for each cell, the clean activation at that site is copied into the corrupted
          run (denoising) and the chart shows how much of the clean behaviour is restored (Recovery).
        </p>
        <div className="lab-input-grid" style={{ margin: "0.8rem 0" }}>
          <div className="lab-input">
            <label htmlFor="sweep-task">Task</label>
            <select id="sweep-task" value={taskId} onChange={(e) => setTaskId(e.target.value)} disabled={running}>
              {tasks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>
          <div className="lab-input">
            <label htmlFor="sweep-mode">Mode</label>
            <select
              id="sweep-mode"
              value={mode}
              onChange={(e) => setMode(e.target.value as "layer_position" | "layer_head")}
              disabled={running}
            >
              <option value="layer_position">layer × position (12 × seq)</option>
              <option value="layer_head">layer × head (12 × 12)</option>
            </select>
          </div>
          <div className="lab-input" style={{ gap: "0.5rem" }}>
            <button className="primary" onClick={runSweep} disabled={!ready || !taskId || running}>
              Run sweep
            </button>
            <button onClick={cancelSweep} disabled={!running}>
              Cancel
            </button>
          </div>
        </div>

        {running && job && (
          <div style={{ margin: "0.6rem 0" }}>
            <div className="inspector-meta" style={{ marginBottom: "0.25rem" }} role="status">
              {Math.round((job.progress ?? 0) * 100)}% — {job.progress_message || "running"}
            </div>
            <div style={{ background: "var(--bg-2)", borderRadius: 4, height: 10, overflow: "hidden" }}>
              <div
                style={{
                  width: `${Math.round((job.progress ?? 0) * 100)}%`,
                  height: "100%",
                  background: "var(--sage)",
                  transition: "width 0.3s",
                }}
              />
            </div>
          </div>
        )}
        {job?.status === "cancelled" && (
          <div className="state-banner warn" role="status" style={{ marginTop: "0.5rem" }}>
            Sweep cancelled. No results were saved.
          </div>
        )}
        {sweepError && (
          <div className="state-banner error" role="alert" style={{ marginTop: "0.5rem" }}>
            {sweepError}
          </div>
        )}

        {sweepResult ? (
          <>
            <div className="metric-cards" style={{ margin: "0.8rem 0" }}>
              <div className="metric-card">
                <div className="name">D_clean</div>
                <div className="value">{sweepResult.baselines.D_clean.toFixed(4)}</div>
                <div className="sub">logit( Mary) − logit( John), clean run</div>
              </div>
              <div className="metric-card">
                <div className="name">D_corrupted</div>
                <div className="value">{sweepResult.baselines.D_corrupted.toFixed(4)}</div>
                <div className="sub">same metric, corrupted run</div>
              </div>
              <div className="metric-card">
                <div className="name">Runs</div>
                <div className="value">{sweepResult.n_runs}</div>
                <div className="sub">{sweepResult.elapsed_seconds.toFixed(1)} s · one clean cache reused</div>
              </div>
            </div>
            {sweepResult.baselines.warnings.map((w, i) => (
              <div key={i} className="state-banner warn" role="status">
                {w}
              </div>
            ))}
            <SweepHeatmap
              rowLabels={sweepResult.row_labels}
              colLabels={sweepResult.col_labels}
              recovery={sweepResult.recovery}
              dPatched={sweepResult.d_patched}
              direction={sweepResult.direction}
            />
            <p className="inspector-meta" style={{ marginTop: "0.5rem" }}>
              Reading the grid: a cell near 1 means patching that site alone restores almost the whole clean
              behaviour; near 0 means that site alone does not matter for this metric. This is causal evidence
              about the patch, not proof the component is "the" circuit member.
            </p>
            {jobId && <PlainWords jobId={jobId} />}
            {jobId && (
              <ReproduceButton
                spec={{ kind: "sweep", job_id: jobId, mode: "sample" }}
                runtime="The script checks a documented sample in about 15–40 s; --mode full checks every cell."
              />
            )}
          </>
        ) : (
          !running &&
          !sweepError && (
            <div className="state-banner info" role="status">
              No sweep yet — choose a task and mode, then Run sweep. A full layer × position sweep is one clean
              capture plus one patched run per cell (a few seconds on CPU).
            </div>
          )
        )}
      </section>

      {/* ---- 2. direct logit attribution ---- */}
      <section className="panel">
        <h2>Direct logit attribution</h2>
        <p className="inspector-meta">
          What you are looking at: how much each component directly pushed the final logits toward
          candidate A over candidate B — a partial view that ignores effects mediated by later layers.
        </p>
        <div style={{ display: "flex", gap: "0.6rem", margin: "0.8rem 0", flexWrap: "wrap" }}>
          <button className="primary" onClick={() => runDla("clean")} disabled={!ready || !taskId || dlaBusy !== null}>
            {dlaBusy === "clean" ? "Computing…" : "Run DLA (clean prompt)"}
          </button>
          <button onClick={() => runDla("corrupted")} disabled={!ready || !taskId || dlaBusy !== null}>
            {dlaBusy === "corrupted" ? "Computing…" : "Run DLA (corrupted prompt)"}
          </button>
        </div>
        {dlaError && (
          <div className="state-banner error" role="alert" style={{ marginTop: "0.5rem" }}>
            {dlaError}
          </div>
        )}
        {["clean", "corrupted"].map((which) => {
          const d = which === "clean" ? dlaClean : dlaCorrupted;
          if (!d) return null;
          const top = [...d.components]
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
            .slice(0, 12);
          const maxAbsVal = Math.max(...d.components.map((c) => Math.abs(c.value)), 1e-9);
          const heads = d.components.filter((c) => c.kind === "head");
          const headMax = Math.max(...heads.map((c) => Math.abs(c.value)), 1e-9);
          return (
            <div key={which} style={{ marginTop: "1rem" }}>
              <h3>
                {which} prompt · D = {d.D.toFixed(4)}{" "}
                <span className="inspector-meta">
                  (logit {d.candidate_a} − logit {d.candidate_b} at position {d.position})
                </span>
              </h3>
              <div style={{ margin: "0.6rem 0" }}>
                {top.map((c) => {
                  const frac = c.value / maxAbsVal;
                  return (
                    <div className="bar-row" key={c.name}>
                      <span className="label">{c.name}</span>
                      <span className="bar-track" style={{ background: "var(--bg-2)" }}>
                        <span
                          className="bar-fill"
                          style={{
                            background: c.value >= 0 ? mix(ZERO, POS, Math.abs(frac) + 0.3) : mix(ZERO, NEG, Math.abs(frac) + 0.3),
                            left: c.value >= 0 ? "50%" : `${50 - Math.abs(frac) * 50}%`,
                            width: `${Math.abs(frac) * 50}%`,
                          }}
                        />
                        <span style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--line-1)" }} />
                      </span>
                      <span className="value">{c.value.toFixed(4)}</span>
                    </div>
                  );
                })}
              </div>
              <div className="head-matrix" role="img" aria-label={`Head contributions grid, 12 layers by 12 heads, ${which} prompt`}>
                {Array.from({ length: 12 }, (_, l) => (
                  <div className="head-row" key={l}>
                    <span className="head-row-label">L{l}</span>
                    <span className="head-row-cells">
                      {heads.slice(l * 12, l * 12 + 12).map((c) => (
                        <span
                          key={c.head}
                          className="head-dot"
                          title={`L${l}.H${c.head}: ${c.value.toFixed(4)}`}
                          style={{
                            background: diverging(c.value, headMax),
                            borderColor: "var(--line-0)",
                          }}
                        />
                      ))}
                    </span>
                  </div>
                ))}
              </div>
              <div className={`state-banner ${d.reconciliation.ok ? "ok" : "error"}`} role="status" style={{ marginTop: "0.6rem" }}>
                Components sum to {d.reconciliation.sum_of_terms.toFixed(4)}; model D is {d.reconciliation.model_D.toFixed(4)} —
                abs error {d.reconciliation.abs_error.toExponential(2)} (tolerance {d.reconciliation.tolerance}) —{" "}
                {d.reconciliation.ok ? "ok" : "MISMATCH"}.
              </div>
              {d.notes.map((n, i) => (
                <p key={i} className="inspector-meta">
                  {n}
                </p>
              ))}
              <ReproduceButton
                spec={{ kind: "dla", task_id: taskId, pair_id: null, run: which as "clean" | "corrupted" }}
                runtime="One raw forward pass with hooks; usually a few seconds."
                compact
              />
            </div>
          );
        })}
      </section>

      {/* ---- 3. logit lens ---- */}
      <section className="panel">
        <h2>Logit lens</h2>
        <p className="inspector-meta">
          What you are looking at: what the model would predict if it stopped after each block — the final
          LayerNorm and unembedding applied to the intermediate residual stream (heuristic view).
        </p>
        <div style={{ margin: "0.8rem 0" }}>
          <label htmlFor="lens-text">Text</label>
          <textarea
            id="lens-text"
            value={lensText}
            onChange={(e) => setLensText(e.target.value)}
            rows={2}
            style={{ width: "100%", maxWidth: "60ch" }}
          />
          <div className="lab-input-grid" style={{ marginTop: "0.5rem" }}>
            <div className="lab-input">
              <label htmlFor="lens-a">Candidate A (optional)</label>
              <input id="lens-a" value={lensA} onChange={(e) => setLensA(e.target.value)} placeholder=" Mary" />
            </div>
            <div className="lab-input">
              <label htmlFor="lens-b">Candidate B (optional)</label>
              <input id="lens-b" value={lensB} onChange={(e) => setLensB(e.target.value)} placeholder=" John" />
            </div>
            <div className="lab-input">
              <button className="primary" onClick={runLens} disabled={!ready || lensBusy || lensText.length === 0}>
                {lensBusy ? "Computing…" : "Run lens"}
              </button>
            </div>
          </div>
        </div>
        {lensError && (
          <div className="state-banner error" role="alert" style={{ marginTop: "0.5rem" }}>
            {lensError}
          </div>
        )}
        {lensResult && (
          <>
            <div className="table-scroll">
              <table className="kv-table">
                <thead>
                  <tr>
                    <th scope="col">Stage</th>
                    <th scope="col">D</th>
                    <th scope="col">P(A)</th>
                    <th scope="col">P(B)</th>
                    <th scope="col">Top-1 token</th>
                  </tr>
                </thead>
                <tbody>
                  {lensResult.stages.map((s) => (
                    <tr key={s.stage}>
                      <td>{s.stage}</td>
                      <td>{s.D == null ? "—" : s.D.toFixed(3)}</td>
                      <td>{s.prob_a == null ? "—" : s.prob_a.toFixed(4)}</td>
                      <td>{s.prob_b == null ? "—" : s.prob_b.toFixed(4)}</td>
                      <td>
                        {s.top[0] ? `${s.top[0].token} (${s.top[0].prob.toFixed(3)})` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className={`state-banner ${lensResult.final_matches_model.ok ? "ok" : "error"}`} role="status" style={{ marginTop: "0.6rem" }}>
              Final stage reproduces the model's real logits: {lensResult.final_matches_model.ok ? "ok" : "MISMATCH"}{" "}
              (candidate-logit diff {lensResult.final_matches_model.max_abs_logit_diff_candidates?.toExponential(2) ?? "n/a"},
              top-1 {lensResult.final_matches_model.top1_agrees ? "agrees" : "DISAGREES"}).
            </div>
            <p className="inspector-meta" style={{ marginTop: "0.5rem" }}>
              {lensResult.note}
            </p>
            <ReproduceButton
              spec={{
                kind: "lens",
                text: lensText,
                candidate_a: lensA || null,
                candidate_b: lensB || null,
                mode: "sample",
              }}
              runtime="Recomputes every stage with raw hooks; usually a few seconds."
              compact
            />
          </>
        )}
      </section>

      {/* ---- 4. do the three methods agree? ---- */}
      <section className="panel" aria-labelledby="agreement-heading">
        <h2 id="agreement-heading">Do the three methods agree?</h2>
        <p className="inspector-meta">
          Exact patching (the reference), attribution patching (a cheap first-order approximation) and
          direct logit attribution (direct pushes only) are compared on the same prompt pair, for all 144
          heads and 12 MLPs. Agreement here is about these scores on this pair, not proof that the methods
          measure the same thing.
        </p>
        <div style={{ display: "flex", gap: "0.6rem", margin: "0.8rem 0", flexWrap: "wrap" }}>
          <button className="primary" onClick={runAgreement} disabled={!ready || agreeRunning}>
            {agreeRunning ? "Measuring…" : "Run agreement measurement"}
          </button>
          {agreeRunning && agreeJobId && (
            <button onClick={() => api.cancelExperiment(agreeJobId).catch(() => {})}>Cancel</button>
          )}
        </div>
        {agreeRunning && agreeJob && (
          <div style={{ margin: "0.6rem 0" }}>
            <div className="inspector-meta" role="status" style={{ marginBottom: "0.25rem" }}>
              {Math.round((agreeJob.progress ?? 0) * 100)}% — {agreeJob.progress_message || "running"}
            </div>
            <div style={{ background: "var(--bg-2)", borderRadius: 4, height: 10, overflow: "hidden" }}>
              <div
                style={{
                  width: `${Math.round((agreeJob.progress ?? 0) * 100)}%`,
                  height: "100%",
                  background: "var(--sage)",
                  transition: "width 0.3s",
                }}
              />
            </div>
          </div>
        )}
        {agreeError && (
          <div className="state-banner error" role="alert" style={{ marginTop: "0.5rem" }}>
            {agreeError}
          </div>
        )}
        {agreementResult ? (
          <>
            <div className="metric-cards" style={{ margin: "0.8rem 0" }}>
              <div className="metric-card">
                <div className="name">AtP vs exact · Pearson</div>
                <div className="value">
                  {agreementResult.stats.atp.pearson_heads == null
                    ? "n/a"
                    : agreementResult.stats.atp.pearson_heads.toFixed(3)}
                </div>
                <div className="sub">
                  top-10 overlap {agreementResult.stats.atp.top10_overlap}/10 · random{" "}
                  {agreementResult.stats.atp.expected_random_top10.toFixed(2)}
                </div>
              </div>
              <div className="metric-card">
                <div className="name">DLA vs exact · Pearson</div>
                <div className="value">
                  {agreementResult.stats.dla.pearson_heads == null
                    ? "n/a"
                    : agreementResult.stats.dla.pearson_heads.toFixed(3)}
                </div>
                <div className="sub">
                  top-10 overlap {agreementResult.stats.dla.top10_overlap}/10 · random{" "}
                  {agreementResult.stats.dla.expected_random_top10.toFixed(2)}
                </div>
              </div>
              <div className="metric-card">
                <div className="name">DLA reconciliation</div>
                <div className="value">
                  {agreementResult.dla_reconciliation.abs_error.toExponential(2)}
                </div>
                <div className="sub">
                  abs error vs tolerance {agreementResult.dla_reconciliation.tolerance} —{" "}
                  {agreementResult.dla_reconciliation.ok ? "ok" : "MISMATCH"}
                </div>
              </div>
            </div>

            <div className="grid-2" style={{ alignItems: "start" }}>
              <AgreementScatter
                title="Attribution patching vs exact patching"
                xLabel="Exact recovery"
                yLabel="AtP recovery"
                points={agreementResult.scatter.map((p) => ({
                  label: p.label,
                  kind: p.kind,
                  layer: p.layer,
                  head: p.head,
                  x: p.exact,
                  y: p.atp,
                }))}
              />
              <AgreementScatter
                title="Direct logit attribution vs exact patching"
                xLabel="Exact recovery"
                yLabel="DLA value"
                points={agreementResult.scatter.map((p) => ({
                  label: p.label,
                  kind: p.kind,
                  layer: p.layer,
                  head: p.head,
                  x: p.exact,
                  y: p.dla,
                }))}
              />
            </div>

            <h3 style={{ marginTop: "1rem" }}>Top-10 heads by each method</h3>
            <p className="inspector-meta">
              A head appears in all three lists is marked &quot;shared&quot;. Badges carry the meaning;
              colour is not required.
            </p>
            <div className="grid-3">
              {(["exact", "atp", "dla"] as const).map((method) => (
                <div key={method}>
                  <h4>
                    {method === "exact" ? "Exact patching" : method === "atp" ? "Attribution patching" : "DLA"}
                  </h4>
                  <ol className="tour-list">
                    {agreementResult.top10[method].map((item) => (
                      <li key={`${method}-${item.rank}`}>
                        {item.label} · <span className="mono">{item.value.toFixed(3)}</span>{" "}
                        {sharedLabels.has(item.label) && <span className="badge">shared</span>}
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
            </div>

            <div className="table-scroll" style={{ marginTop: "1rem" }}>
              <table className="kv-table">
                <caption>Agreement statistics over the 144 heads. Values are Pearson of raw values.</caption>
                <thead>
                  <tr>
                    <th scope="col">Comparison</th>
                    <th scope="col">Pearson</th>
                    <th scope="col">Spearman</th>
                    <th scope="col">Blocks 0–5</th>
                    <th scope="col">Blocks 6–11</th>
                  </tr>
                </thead>
                <tbody>
                  {(["atp", "dla"] as const).map((method) => {
                    const s = agreementResult.stats[method];
                    return (
                      <tr key={method}>
                        <td>{method === "atp" ? "AtP vs exact" : "DLA vs exact"}</td>
                        <td className="mono">{s.pearson_heads == null ? "n/a" : s.pearson_heads.toFixed(3)}</td>
                        <td className="mono">{s.spearman_heads == null ? "n/a" : s.spearman_heads.toFixed(3)}</td>
                        <td className="mono">
                          {s.pearson_groups.blocks_0_5?.pearson == null
                            ? "n/a"
                            : s.pearson_groups.blocks_0_5.pearson.toFixed(3)}
                        </td>
                        <td className="mono">
                          {s.pearson_groups.blocks_6_11?.pearson == null
                            ? "n/a"
                            : s.pearson_groups.blocks_6_11.pearson.toFixed(3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="inspector-meta">{agreementResult.stats.procedure}</p>

            <div style={{ marginTop: "0.8rem" }}>
              {(["exact", "atp", "dla"] as const).map((method) => (
                <p key={method} className="inspector-meta">
                  <strong>
                    {method === "exact" ? "Exact patching" : method === "atp" ? "Attribution patching" : "DLA"}:
                  </strong>{" "}
                  {agreementResult.method_notes[method]}
                </p>
              ))}
            </div>
            {agreeJobId && <PlainWords jobId={agreeJobId} />}
          </>
        ) : (
          !agreeRunning &&
          !agreeError && (
            <div className="state-banner info" role="status">
              No agreement measurement yet — run one to compare the three methods on the selected task
              pair. It takes about 8 seconds on CPU.
            </div>
          )
        )}
      </section>
    </main>
  );
}
