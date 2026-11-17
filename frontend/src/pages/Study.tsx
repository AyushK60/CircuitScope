import { useEffect, useRef, useState } from "react";
import {
  api,
  type Health,
  type SavedExperiment,
  type StudyJobStatus,
  type StudyResult,
} from "../api/client";
import Rq1Table from "./study/Rq1Table";
import OverlapMatrix from "./study/OverlapMatrix";
import TopHeads from "./study/TopHeads";

/**
 * Study page (Stage 3): the pre-registered prompt-set study. Every number
 * comes from the API response of a real study run; the page shows fixed
 * caveat texts and labels only — no generated prose claims.
 */

export default function Study({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [nPrompts, setNPrompts] = useState(40);
  const [seed, setSeed] = useState(0);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<StudyJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const [recent, setRecent] = useState<SavedExperiment[]>([]);

  const refreshRecent = () => {
    api
      .listExperiments()
      .then((r) =>
        setRecent(
          r.saved.filter((s) => (s.config as Record<string, unknown> | null)?.job_type === "study"),
        ),
      )
      .catch(() => {});
  };
  useEffect(refreshRecent, []);

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = window.setInterval(() => {
      api
        .studyStatus(jobId)
        .then((r) => {
          setJob(r);
          if (r.status === "failed") setError(r.error ?? "study failed");
          if (r.status === "complete") refreshRecent();
        })
        .catch((e) => setError(String(e.message ?? e)));
    }, 500);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId]);

  const running = job?.status === "running" || job?.status === "queued";
  const result: StudyResult | null = job?.status === "complete" ? job.results : null;

  const runStudy = () => {
    setError(null);
    setJob(null);
    setJobId(null);
    api
      .startStudy({ n_prompts: nPrompts, seed })
      .then((r) => setJobId(r.job_id))
      .catch((e) => setError(String(e.message ?? e)));
  };
  const cancelStudy = () => {
    if (jobId) api.cancelExperiment(jobId).catch(() => {});
  };

  const loadStudy = (id: string) => {
    setError(null);
    api
      .studyStatus(id)
      .then((r) => {
        setJob(r);
        setJobId(id);
        if (r.status === "complete") setNPrompts(r.results?.params.n_prompts ?? nPrompts);
      })
      .catch((e) => setError(String(e.message ?? e)));
  };

  const rq1 = result?.summary.rq1.all_prompts;
  const rq2 = result?.summary.rq2;

  return (
    <main className="page">
      <h1>Study</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "80ch" }}>
        The pre-registered Stage 3 study over a generated IOI prompt set: exact per-component effects
        (denoising under three corruption methods, zero and mean ablation), attribution patching and DLA,
        with the statistics fixed in <code className="mono">docs/PAPER_NOTES.md §4b</code> before any full run.
        Direction: denoising copies clean activations into the corrupted run (sufficiency); ablation replaces
        a component's output in the clean run (necessity).
      </p>

      {/* ---- controls ---- */}
      <section className="panel">
        <h2>Run a study</h2>
        <p className="inspector-meta">
          A full 40-prompt study takes roughly 20–25 minutes on CPU (~30 s per prompt). You can also load a
          previously completed study below — the page is fully usable without running anything.
        </p>
        <div className="lab-input-grid" style={{ margin: "0.8rem 0" }}>
          <div className="lab-input">
            <label htmlFor="study-n">n_prompts (4–100)</label>
            <input
              id="study-n"
              type="number"
              min={4}
              max={100}
              value={nPrompts}
              disabled={running}
              onChange={(e) => setNPrompts(Math.max(4, Math.min(100, Number(e.target.value) || 4)))}
            />
          </div>
          <div className="lab-input">
            <label htmlFor="study-seed">seed</label>
            <input
              id="study-seed"
              type="number"
              min={0}
              value={seed}
              disabled={running}
              onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
            />
          </div>
          <div className="lab-input" style={{ gap: "0.5rem" }}>
            <button className="primary" onClick={runStudy} disabled={!ready || running}>
              Run study
            </button>
            <button onClick={cancelStudy} disabled={!running}>
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
            Study cancelled. No partial results were saved.
          </div>
        )}
        {error && (
          <div className="state-banner error" role="alert" style={{ marginTop: "0.5rem" }}>
            {error}
          </div>
        )}

        {recent.length > 0 && (
          <div style={{ marginTop: "0.8rem" }}>
            <h3>Completed studies</h3>
            <div className="table-scroll">
              <table className="kv-table">
                <thead>
                  <tr>
                    <th scope="col">id</th>
                    <th scope="col">n_prompts</th>
                    <th scope="col">seed</th>
                    <th scope="col"></th>
                  </tr>
                </thead>
                <tbody>
                  {recent.slice(0, 10).map((s) => (
                    <tr key={s.id}>
                      <td className="mono">{s.id}</td>
                      <td className="mono">{String((s.config as Record<string, unknown>).n_prompts ?? "")}</td>
                      <td className="mono">{String((s.config as Record<string, unknown>).seed ?? "")}</td>
                      <td>
                        <button className="ghost" onClick={() => loadStudy(s.id)}>
                          Load
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {!result && !running && (
        <div className="state-banner info" role="status">
          No study loaded yet — run one or load a completed study from the table above.
        </div>
      )}

      {result && (
        <>
          {/* ---- 1. headline numbers ---- */}
          <section className="panel">
            <h2>
              Study {result.params.n_prompts} prompts · seed {result.params.seed}
            </h2>
            <div className="metric-cards" style={{ margin: "0.8rem 0" }}>
              <div className="metric-card">
                <div className="name">Prompts</div>
                <div className="value">{result.per_prompt.length}</div>
                <div className="sub">
                  excluded per condition: swap {result.exclusions.swap} · third {result.exclusions.third} ·
                  resample {result.exclusions.resample} · mean-ablation skipped {result.exclusions.abl_mean}
                </div>
              </div>
              <div className="metric-card">
                <div className="name">Model accuracy</div>
                <div className="value">{(result.model_accuracy * 100).toFixed(1)}%</div>
                <div className="sub">fraction of prompts with D_clean &gt; 0</div>
              </div>
              <div className="metric-card">
                <div className="name">Runtime</div>
                <div className="value">{Math.round(result.elapsed_seconds)} s</div>
                <div className="sub">exclusion rule: {result.exclusion_rule}</div>
              </div>
            </div>
          </section>

          {/* ---- 2. RQ1 table ---- */}
          <section className="panel">
            <h2>RQ1 — do the cheap methods agree with exact patching?</h2>
            <Rq1Table block={rq1!} />
            {result.summary.rq1.subset_D_clean_gt0 && (
              <details style={{ marginTop: "0.6rem" }}>
                <summary className="inspector-meta">
                  Same statistics on the subset with D_clean &gt; 0 ({result.summary.rq1.subset_D_clean_gt0.n_prompts}{" "}
                  prompts) — pre-registered sensitivity check
                </summary>
                <Rq1Table block={result.summary.rq1.subset_D_clean_gt0} />
              </details>
            )}
          </section>

          {/* ---- 3. top-10 heads per condition ---- */}
          <section className="panel">
            <h2>Top-10 heads per condition</h2>
            <TopHeads rq2={rq2!} known={result.summary.anchors.known_heads_reference} />
            {result.summary.anchors.mean_ranks_by_group && (
              <p className="inspector-meta" style={{ marginTop: "0.5rem" }}>
                Known-heads anchor (descriptive only, rank 1 = highest prompt-averaged Rec_D[swap]):{" "}
                {Object.entries(result.summary.anchors.mean_ranks_by_group)
                  .map(([name, g]) => `${name}: mean rank ${g.mean_rank?.toFixed(1) ?? "n/a"}`)
                  .join(" · ")}
              </p>
            )}
          </section>

          {/* ---- 4. overlap matrices ---- */}
          <section className="panel">
            <h2>RQ2 — ranking stability across conditions</h2>
            <p className="inspector-meta">
              Conditions: 3 corruption methods × 2 metrics (denoising) + zero/mean ablation. Overlap of the
              top-k heads ranked by signed prompt-averaged score.
            </p>
            <div className="grid-2" style={{ alignItems: "start" }}>
              <div>
                <h3>Top-5 overlap</h3>
                <OverlapMatrix
                  data={rq2!.topk.k5}
                  k={5}
                  expectedRandom={rq2!.topk.expected_random.k5}
                  conditionNames={rq2!.condition_names}
                />
              </div>
              <div>
                <h3>Top-10 overlap</h3>
                <OverlapMatrix
                  data={rq2!.topk.k10}
                  k={10}
                  expectedRandom={rq2!.topk.expected_random.k10}
                  conditionNames={rq2!.condition_names}
                />
              </div>
            </div>
          </section>

          {/* ---- 5. caveats ---- */}
          <section className="panel">
            <h2>What these numbers do and do not show</h2>
            <ul className="caveat-list">
              <li>
                <strong>Attribution patching (AtP) is a first-order approximation</strong> of the exact
                patching effect; it is known to be unreliable when attention saturates. Disagreement with
                exact patching is expected for some components and is a result, not a bug.
              </li>
              <li>
                <strong>DLA measures direct effects only</strong> — it ignores effects mediated by later
                layers, so it is a partial view, not a patching replacement.
              </li>
              <li>
                <strong>Denoising and ablation answer different questions.</strong> Denoising (copying clean
                activations into the corrupted run) is sufficiency-style evidence; ablation (replacing a
                component's output in the clean run) is necessity-style evidence. They are never mixed in one
                figure without labelling.
              </li>
              <li>
                <strong>Exclusions:</strong> per (prompt, corruption) pair, prompts with
                |D_clean − D_corr| &lt; 0.5 are excluded (counted above); no other prompt was dropped.
              </li>
            </ul>
          </section>
        </>
      )}
    </main>
  );
}
