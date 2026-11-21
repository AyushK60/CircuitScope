import { useEffect, useRef, useState, type ReactNode } from "react";
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
import { DlaBars, LensChart } from "./sweeps/Charts";
import { agreementCallouts, dlaCallouts, lensCallouts, patchingCallouts, type Callout } from "./sweeps/insights";
import { useSeen } from "./sweeps/useSeen";

/**
 * Sweeps: one question, four measurements, read top to bottom. Every number comes from an API
 * response computed on the real model. The first step runs by itself; each later step runs when it
 * scrolls into view. Sections state what they can and cannot prove.
 */

function Step({
  id,
  num,
  title,
  intro,
  callouts,
  waiting,
  stepRef,
  children,
}: {
  id: string;
  num: string;
  title: string;
  intro: ReactNode;
  callouts: Callout[];
  waiting: string;
  stepRef?: React.RefObject<HTMLElement | null>;
  children: ReactNode;
}) {
  return (
    <section className="sw-step" id={id} ref={stepRef}>
      <div className="sw-copy">
        <p className="sw-num">{num}</p>
        <h2>{title}</h2>
        <p>{intro}</p>
        {callouts.length > 0 ? (
          <div className="sw-callouts">
            {callouts.map((c) => (
              <span key={c.lead}>
                <b>{c.lead}</b> {c.text}
              </span>
            ))}
          </div>
        ) : (
          <p className="sw-waiting">{waiting}</p>
        )}
      </div>
      <div className="sw-card">{children}</div>
    </section>
  );
}

function Progress({ value, message }: { value: number; message?: string }) {
  return (
    <div className="sw-progress" role="status">
      <span>
        {Math.round(value * 100)}% · {message || "running"}
      </span>
      <div>
        <i style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

function Banner({ kind, children }: { kind: "error" | "warn" | "info" | "ok"; children: ReactNode }) {
  return (
    <div className={`state-banner ${kind}`} role={kind === "error" ? "alert" : "status"} style={{ marginTop: "0.6rem" }}>
      {children}
    </div>
  );
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
  const [dlaView, setDlaView] = useState<"clean" | "corrupted">("clean");
  const [dlaBusy, setDlaBusy] = useState<"clean" | "corrupted" | null>(null);
  const [dlaError, setDlaError] = useState<string | null>(null);

  const [lens, setLens] = useState({ task: "", text: "" });
  const [lensA, setLensA] = useState("");
  const [lensB, setLensB] = useState("");
  const [lensResult, setLensResult] = useState<{ task: string; result: LensResult } | null>(null);
  const [lensBusy, setLensBusy] = useState(false);
  const [lensError, setLensError] = useState<string | null>(null);

  const [agreeJobId, setAgreeJobId] = useState<string | null>(null);
  const [agreeJob, setAgreeJob] = useState<AgreementJobStatus | null>(null);
  const [agreeError, setAgreeError] = useState<string | null>(null);

  const [pushRef, pushSeen] = useSeen<HTMLElement>();
  const [whenRef, whenSeen] = useSeen<HTMLElement>();
  const [agreeRef, agreeSeen] = useSeen<HTMLElement>();
  const auto = useRef({ sweep: "", dla: "", lens: "", agree: "" });

  useEffect(() => {
    api
      .tasks()
      .then((r) => {
        setTasks(r.tasks);
        if (r.tasks.length > 0) setTaskId((cur) => cur || r.tasks[0].id);
      })
      .catch((e) => setSweepError(String(e.message ?? e)));
  }, []);

  // the logit lens starts from the selected task's clean prompt
  useEffect(() => {
    if (!taskId) return;
    api
      .task(taskId)
      .then((t) => setLens({ task: taskId, text: t.pairs[0]?.clean_text ?? "" }))
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

  const sweepResult: SweepResult | null = job?.status === "complete" ? job.results : null;
  const running = job?.status === "running" || job?.status === "queued";

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
    const forTask = taskId;
    api
      .logitLens(lens.text, lensA || null, lensB || null, 5)
      .then((r) => setLensResult({ task: forTask, result: r }))
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

  /* ---- automatic runs: the sweep right away, the rest as each step comes into view ---- */
  useEffect(() => {
    if (!ready || !taskId) return;
    const key = `${taskId}|${mode}`;
    if (auto.current.sweep === key) return;
    auto.current.sweep = key;
    runSweep();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, taskId, mode]);

  useEffect(() => {
    if (!ready || !taskId || !pushSeen) return;
    const key = `${taskId}|clean`;
    if (auto.current.dla === key) return;
    auto.current.dla = key;
    runDla("clean");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, taskId, pushSeen]);

  useEffect(() => {
    if (!ready || !whenSeen || lens.task !== taskId || !lens.text) return;
    if (auto.current.lens === taskId) return;
    auto.current.lens = taskId;
    runLens();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, whenSeen, lens, taskId]);

  useEffect(() => {
    if (!ready || !taskId || !agreeSeen) return;
    if (auto.current.agree === taskId) return;
    auto.current.agree = taskId;
    runAgreement();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, taskId, agreeSeen]);

  const pickDla = (which: "clean" | "corrupted") => {
    setDlaView(which);
    const have = which === "clean" ? dlaClean : dlaCorrupted;
    if (!(have && have.task_id === taskId) && dlaBusy === null) runDla(which);
  };

  const dlaCurrent = (() => {
    const d = dlaView === "clean" ? dlaClean : dlaCorrupted;
    return d && d.task_id === taskId ? d : null;
  })();
  const lensCurrent = lensResult && lensResult.task === taskId ? lensResult.result : null;
  const agreementResult: AgreementResult | null =
    agreeJob?.status === "complete" && agreeJob.results?.task_id === taskId ? agreeJob.results : null;
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

  const jump = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <main className="page sw-page">
      <p className="ex-kick">One question, four measurements</p>
      <h1 className="ex-title">Which parts of GPT-2 matter for this answer?</h1>
      <p className="ex-sub">
        Four measurements on the same sentence, each looking from a different angle. They run as you scroll.
      </p>

      <div className="sw-sticky">
        <label htmlFor="sweep-task">Task</label>
        <select id="sweep-task" value={taskId} onChange={(e) => setTaskId(e.target.value)} disabled={running}>
          {tasks.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
        {lens.task === taskId && lens.text && <span className="sw-prompt">“{lens.text}”</span>}
        <div className="sw-jumps">
          <button onClick={() => jump("sw-patch")}>1 Patch</button>
          <button onClick={() => jump("sw-push")}>2 Push</button>
          <button onClick={() => jump("sw-when")}>3 When</button>
          <button onClick={() => jump("sw-agree")}>4 Agree</button>
        </div>
      </div>

      {!ready && (
        <Banner kind="info">
          <span className="spinner" aria-hidden /> Waiting for the model. These measurements need the loaded weights.
        </Banner>
      )}

      {/* ---- 1. patching sweep ---- */}
      <Step
        id="sw-patch"
        num="STEP 1 · PATCH"
        title="Put a clean signal back, one spot at a time"
        intro="We break the sentence, then copy the original signal into one place and see how much of the right answer returns. Patching always goes clean into corrupted (denoising)."
        callouts={sweepResult ? patchingCallouts(sweepResult) : []}
        waiting={running ? "Measuring…" : "The result appears here."}
      >
        <div className="sw-cap">
          <span>{mode === "layer_position" ? "block × word position" : "block × attention head"}</span>
          {sweepResult ? <span className="live-badge">MEASURED</span> : <span className="static-badge">WAITING</span>}
        </div>
        <div className="sw-bar">
          <div className="mode-tabs" role="group" aria-label="Sweep mode">
            <button className={mode === "layer_position" ? "active" : ""} onClick={() => setMode("layer_position")} disabled={running}>
              layer × position
            </button>
            <button className={mode === "layer_head" ? "active" : ""} onClick={() => setMode("layer_head")} disabled={running}>
              layer × head
            </button>
          </div>
          <button onClick={runSweep} disabled={!ready || !taskId || running}>
            Run again
          </button>
          {running && <button onClick={cancelSweep}>Cancel</button>}
        </div>
        {running && job && <Progress value={job.progress ?? 0} message={job.progress_message} />}
        {job?.status === "cancelled" && <Banner kind="warn">Sweep cancelled. No results were saved.</Banner>}
        {sweepError && <Banner kind="error">{sweepError}</Banner>}
        {sweepResult ? (
          <>
            <div className="sw-stats">
              <div>
                <small>D_clean</small>
                <strong>{sweepResult.baselines.D_clean.toFixed(4)}</strong>
                <span>logit({sweepResult.candidate_a}) − logit({sweepResult.candidate_b}), clean run</span>
              </div>
              <div>
                <small>D_corrupted</small>
                <strong>{sweepResult.baselines.D_corrupted.toFixed(4)}</strong>
                <span>same metric, corrupted run</span>
              </div>
              <div>
                <small>Runs</small>
                <strong>{sweepResult.n_runs}</strong>
                <span>{sweepResult.elapsed_seconds.toFixed(1)} s · one clean cache reused</span>
              </div>
            </div>
            {sweepResult.baselines.warnings.map((w, i) => (
              <Banner key={i} kind="warn">
                {w}
              </Banner>
            ))}
            <SweepHeatmap
              rowLabels={sweepResult.row_labels}
              colLabels={sweepResult.col_labels}
              recovery={sweepResult.recovery}
              dPatched={sweepResult.d_patched}
              direction={sweepResult.direction}
            />
            <p className="sw-note">
              A cell near 1 means patching that site alone restores almost the whole clean behaviour; near 0 means
              that site alone does not matter for this metric. This is causal evidence about the patch, not proof the
              component is “the” circuit member.
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
          !sweepError && <div className="skeleton" style={{ height: "18rem" }} />
        )}
      </Step>

      {/* ---- 2. direct logit attribution ---- */}
      <Step
        id="sw-push"
        num="STEP 2 · PUSH"
        title="Who pushes the answer?"
        intro="Every part of the model adds a little to the final scores. This shows which parts push the first candidate up and which push the second. It is a partial view: it ignores effects passed through later blocks."
        callouts={dlaCurrent ? dlaCallouts(dlaCurrent) : []}
        waiting={dlaBusy ? "Measuring…" : "Runs when you scroll here."}
        stepRef={pushRef}
      >
        <div className="sw-cap">
          <span>
            {dlaCurrent
              ? `${dlaCurrent.run} prompt · D = ${dlaCurrent.D.toFixed(4)}`
              : "direct logit attribution"}
          </span>
          {dlaCurrent ? <span className="live-badge">MEASURED</span> : <span className="static-badge">WAITING</span>}
        </div>
        <div className="sw-bar">
          <div className="mode-tabs" role="group" aria-label="Which prompt">
            <button className={dlaView === "clean" ? "active" : ""} onClick={() => pickDla("clean")}>
              clean prompt
            </button>
            <button className={dlaView === "corrupted" ? "active" : ""} onClick={() => pickDla("corrupted")}>
              corrupted prompt
            </button>
          </div>
          {dlaBusy && <span className="spinner" aria-label="Computing" />}
        </div>
        {dlaError && <Banner kind="error">{dlaError}</Banner>}
        {dlaCurrent ? (
          <>
            <p className="sw-note" style={{ marginTop: 0 }}>
              logit {dlaCurrent.candidate_a} − logit {dlaCurrent.candidate_b} at position {dlaCurrent.position}
            </p>
            <DlaBars d={dlaCurrent} />
            <Banner kind={dlaCurrent.reconciliation.ok ? "ok" : "error"}>
              Components sum to {dlaCurrent.reconciliation.sum_of_terms.toFixed(4)}; model D is{" "}
              {dlaCurrent.reconciliation.model_D.toFixed(4)}. Absolute error{" "}
              {dlaCurrent.reconciliation.abs_error.toExponential(2)} (tolerance {dlaCurrent.reconciliation.tolerance}):{" "}
              {dlaCurrent.reconciliation.ok ? "ok" : "MISMATCH"}.
            </Banner>
            {dlaCurrent.notes.map((n, i) => (
              <p key={i} className="sw-note">
                {n}
              </p>
            ))}
            <ReproduceButton
              spec={{ kind: "dla", task_id: taskId, pair_id: null, run: dlaCurrent.run }}
              runtime="One raw forward pass with hooks; usually a few seconds."
              compact
            />
          </>
        ) : (
          !dlaError && <div className="skeleton" style={{ height: "16rem" }} />
        )}
      </Step>

      {/* ---- 3. logit lens ---- */}
      <Step
        id="sw-when"
        num="STEP 3 · WHEN"
        title="When does the answer appear?"
        intro="If the model had to stop after each block, what would it guess? This applies the final scoring step to the model's half-finished notes. It is a heuristic view: early guesses are not what the model was thinking."
        callouts={lensCurrent ? lensCallouts(lensCurrent) : []}
        waiting={lensBusy ? "Measuring…" : "Runs when you scroll here."}
        stepRef={whenRef}
      >
        <div className="sw-cap">
          <span>top guess after each block</span>
          {lensCurrent ? <span className="live-badge">MEASURED</span> : <span className="static-badge">WAITING</span>}
        </div>
        {lensError && <Banner kind="error">{lensError}</Banner>}
        {lensCurrent ? (
          <>
            <LensChart lens={lensCurrent} />
            <details className="sw-details">
              <summary>Show the numbers</summary>
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
                    {lensCurrent.stages.map((s) => (
                      <tr key={s.stage}>
                        <td>{s.stage}</td>
                        <td>{s.D == null ? "—" : s.D.toFixed(3)}</td>
                        <td>{s.prob_a == null ? "—" : s.prob_a.toFixed(4)}</td>
                        <td>{s.prob_b == null ? "—" : s.prob_b.toFixed(4)}</td>
                        <td>{s.top[0] ? `${s.top[0].token} (${s.top[0].prob.toFixed(3)})` : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
            <Banner kind={lensCurrent.final_matches_model.ok ? "ok" : "error"}>
              Final stage reproduces the model's real logits: {lensCurrent.final_matches_model.ok ? "ok" : "MISMATCH"}{" "}
              (candidate-logit diff{" "}
              {lensCurrent.final_matches_model.max_abs_logit_diff_candidates?.toExponential(2) ?? "n/a"}, top-1{" "}
              {lensCurrent.final_matches_model.top1_agrees ? "agrees" : "DISAGREES"}).
            </Banner>
            <p className="sw-note">{lensCurrent.note}</p>
          </>
        ) : (
          !lensError && <div className="skeleton" style={{ height: "14rem" }} />
        )}
        <details className="sw-details">
          <summary>Try your own text</summary>
          <label htmlFor="lens-text">Text</label>
          <textarea
            id="lens-text"
            value={lens.text}
            onChange={(e) => setLens((s) => ({ ...s, text: e.target.value }))}
            rows={2}
            style={{ width: "100%" }}
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
              <button className="primary" onClick={runLens} disabled={!ready || lensBusy || lens.text.length === 0}>
                {lensBusy ? "Computing…" : "Run lens"}
              </button>
            </div>
          </div>
        </details>
        {lensCurrent && (
          <ReproduceButton
            spec={{
              kind: "lens",
              text: lens.text,
              candidate_a: lensA || null,
              candidate_b: lensB || null,
              mode: "sample",
            }}
            runtime="Recomputes every stage with raw hooks; usually a few seconds."
            compact
          />
        )}
      </Step>

      {/* ---- 4. do the three methods agree? ---- */}
      <Step
        id="sw-agree"
        num="STEP 4 · AGREE"
        title="Do the methods agree?"
        intro="Exact patching (the reference), attribution patching (a cheap first-order approximation) and direct logit attribution are compared on the same prompt pair, for all 144 heads and 12 MLPs. Agreement is about these scores on this pair, not proof the methods measure the same thing."
        callouts={agreementResult ? agreementCallouts(agreementResult) : []}
        waiting={agreeRunning ? "Measuring…" : "Runs when you scroll here."}
        stepRef={agreeRef}
      >
        <div className="sw-cap">
          <span>three methods, 156 components</span>
          {agreementResult ? <span className="live-badge">MEASURED</span> : <span className="static-badge">WAITING</span>}
        </div>
        <div className="sw-bar">
          <button onClick={runAgreement} disabled={!ready || agreeRunning}>
            {agreeRunning ? "Measuring…" : "Run again"}
          </button>
          {agreeRunning && agreeJobId && (
            <button onClick={() => api.cancelExperiment(agreeJobId).catch(() => {})}>Cancel</button>
          )}
        </div>
        {agreeRunning && agreeJob && <Progress value={agreeJob.progress ?? 0} message={agreeJob.progress_message} />}
        {agreeError && <Banner kind="error">{agreeError}</Banner>}
        {agreementResult ? (
          <>
            <div className="sw-stats">
              <div>
                <small>AtP vs exact · Pearson</small>
                <strong>
                  {agreementResult.stats.atp.pearson_heads == null ? "n/a" : agreementResult.stats.atp.pearson_heads.toFixed(3)}
                </strong>
                <span>
                  top-10 overlap {agreementResult.stats.atp.top10_overlap}/10 · random{" "}
                  {agreementResult.stats.atp.expected_random_top10.toFixed(2)}
                </span>
              </div>
              <div>
                <small>DLA vs exact · Pearson</small>
                <strong>
                  {agreementResult.stats.dla.pearson_heads == null ? "n/a" : agreementResult.stats.dla.pearson_heads.toFixed(3)}
                </strong>
                <span>
                  top-10 overlap {agreementResult.stats.dla.top10_overlap}/10 · random{" "}
                  {agreementResult.stats.dla.expected_random_top10.toFixed(2)}
                </span>
              </div>
              <div>
                <small>DLA reconciliation</small>
                <strong>{agreementResult.dla_reconciliation.abs_error.toExponential(2)}</strong>
                <span>
                  abs error vs tolerance {agreementResult.dla_reconciliation.tolerance}:{" "}
                  {agreementResult.dla_reconciliation.ok ? "ok" : "MISMATCH"}
                </span>
              </div>
            </div>

            <div className="grid-2" style={{ alignItems: "start", marginTop: "1rem" }}>
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

            <details className="sw-details">
              <summary>Top-10 lists, statistics and method notes</summary>
              <p className="sw-note">A head that appears in all three lists is marked “shared”. Badges carry the meaning; colour is not required.</p>
              <div className="grid-3">
                {(["exact", "atp", "dla"] as const).map((method) => (
                  <div key={method}>
                    <h4>{method === "exact" ? "Exact patching" : method === "atp" ? "Attribution patching" : "DLA"}</h4>
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
                            {s.pearson_groups.blocks_0_5?.pearson == null ? "n/a" : s.pearson_groups.blocks_0_5.pearson.toFixed(3)}
                          </td>
                          <td className="mono">
                            {s.pearson_groups.blocks_6_11?.pearson == null ? "n/a" : s.pearson_groups.blocks_6_11.pearson.toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="sw-note">{agreementResult.stats.procedure}</p>
              {(["exact", "atp", "dla"] as const).map((method) => (
                <p key={method} className="sw-note">
                  <strong>{method === "exact" ? "Exact patching" : method === "atp" ? "Attribution patching" : "DLA"}:</strong>{" "}
                  {agreementResult.method_notes[method]}
                </p>
              ))}
            </details>
            {agreeJobId && <PlainWords jobId={agreeJobId} />}
          </>
        ) : (
          !agreeRunning && !agreeError && <div className="skeleton" style={{ height: "14rem" }} />
        )}
      </Step>
    </main>
  );
}
