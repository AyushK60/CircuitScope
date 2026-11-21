import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, type Health, type SavedExperiment, type StudyJobStatus, type StudyResult } from "../api/client";
import Rq1Table from "./study/Rq1Table";
import OverlapMatrix from "./study/OverlapMatrix";
import TopHeads from "./study/TopHeads";
import { ForestPlot, HeadsMap, OverlapGrid, TopList } from "./study/Charts";
import { agreementCallouts, stabilityCallouts, topHeadCallouts } from "./study/insights";
import type { Callout } from "./sweeps/insights";
import ReproduceButton from "../components/ReproduceButton";
import PlainWords from "./report/PlainWords";

/**
 * Study: the pre-registered prompt-set study. Every number comes from the API response of a real study
 * run. The callouts beside each figure are computed from those numbers; the caveats are fixed texts.
 */

type Tab = "agreement" | "heads" | "stability" | "limits";
const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "agreement", label: "Agreement", hint: "Do cheap methods match?" },
  { id: "heads", label: "Top heads", hint: "Which ones matter most" },
  { id: "stability", label: "Stability", hint: "Do rankings hold up?" },
  { id: "limits", label: "Limits", hint: "What it does not show" },
];

/** "swap_D" -> "swap · D" (D = score gap, P = probability); ablation setups keep their kind. */
const conditionLabel = (name: string) => name.replace(/^abl_/, "").replace(/_/g, " · ");

function Finding({
  num,
  title,
  intro,
  callouts,
  children,
}: {
  num: string;
  title: string;
  intro: ReactNode;
  callouts: Callout[];
  children: ReactNode;
}) {
  return (
    <section className="sw-step st-finding">
      <div className="sw-copy">
        <p className="sw-num">{num}</p>
        <h2>{title}</h2>
        <p>{intro}</p>
        {callouts.length > 0 && (
          <div className="sw-callouts">
            {callouts.map((c) => (
              <span key={c.lead}>
                <b>{c.lead}</b> {c.text}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="sw-card">{children}</div>
    </section>
  );
}

export default function Study({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [nPrompts, setNPrompts] = useState(40);
  const [seed, setSeed] = useState(0);
  const [tab, setTab] = useState<Tab>("agreement");
  const [runOpen, setRunOpen] = useState<boolean | null>(null);
  const [condition, setCondition] = useState("swap_D");
  const [topK, setTopK] = useState<5 | 10>(10);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<StudyJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const autoLoaded = useRef(false);

  const [recent, setRecent] = useState<SavedExperiment[]>([]);
  const [picked, setPicked] = useState("");

  const refreshRecent = () => {
    api
      .listExperiments()
      .then((r) => {
        const studies = r.saved
          .filter((s) => (s.config as Record<string, unknown> | null)?.job_type === "study")
          .sort((a, b) => b.created_at - a.created_at);
        setRecent(studies);
        setPicked((cur) => cur || studies[0]?.id || "");
      })
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

  // the common case is looking at a study that already ran, so open the latest one straight away
  useEffect(() => {
    if (autoLoaded.current || recent.length === 0 || jobId || job) return;
    autoLoaded.current = true;
    loadStudy(recent[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recent]);

  const rq1 = result?.summary.rq1.all_prompts;
  const rq2 = result?.summary.rq2;
  const known = result?.summary.anchors.known_heads_reference;
  const showRun = runOpen ?? recent.length === 0;
  const vector = rq2?.prompt_averaged_vectors[condition] ?? null;

  return (
    <main className="page st-page">
      <p className="ex-kick">Does it hold across many sentences?</p>
      <h1 className="ex-title">Study</h1>
      <p className="ex-sub">
        One example can mislead. A study repeats the measurements over a generated set of sentences, with the
        statistics fixed before any full run. Denoising copies clean activations into the corrupted run
        (sufficiency); ablation replaces a component&apos;s output in the clean run (necessity).
      </p>

      <div className="st-picker">
        <div className="st-pick-field">
          <label htmlFor="study-pick">Showing study</label>
          <select
            id="study-pick"
            value={picked}
            onChange={(e) => {
              setPicked(e.target.value);
              if (e.target.value) loadStudy(e.target.value);
            }}
            disabled={recent.length === 0 || running}
          >
            {recent.length === 0 && <option value="">No completed study yet</option>}
            {recent.map((s) => {
              const c = s.config as Record<string, unknown>;
              return (
                <option key={s.id} value={s.id}>
                  {String(c.n_prompts ?? "?")} prompts · seed {String(c.seed ?? "?")} · {s.id}
                </option>
              );
            })}
          </select>
        </div>
        <span className="st-picker-gap" />
        <button className={showRun ? "" : "primary"} onClick={() => setRunOpen(!showRun)} aria-expanded={showRun}>
          {showRun ? "Hide the run form" : "Run a new study"}
        </button>
      </div>

      {showRun && (
        <div className="st-run">
          <div className="st-field">
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
          <div className="st-field">
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
          <div className="st-actions">
            <button className="primary" onClick={runStudy} disabled={!ready || running}>
              Run study
            </button>
            <button onClick={cancelStudy} disabled={!running}>
              Cancel
            </button>
          </div>
          <p className="st-hint">
            A full 40-prompt study takes roughly 20 to 30 minutes on CPU (about 30 s per prompt). The page is fully
            usable without running anything: load a completed study above.
          </p>
        </div>
      )}

      {running && job && (
        <div className="sw-progress" role="status">
          <span>
            {Math.round((job.progress ?? 0) * 100)}% · {job.progress_message || "running"}
          </span>
          <div>
            <i style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} />
          </div>
        </div>
      )}
      {job?.status === "cancelled" && (
        <div className="state-banner warn" role="status" style={{ marginTop: "0.6rem" }}>
          Study cancelled. No partial results were saved.
        </div>
      )}
      {error && (
        <div className="state-banner error" role="alert" style={{ marginTop: "0.6rem" }}>
          {error}
        </div>
      )}
      {!result && !running && (
        <div className="state-banner info" role="status" style={{ marginTop: "1rem" }}>
          No study loaded yet. Run one, or load a completed study from the picker above.
        </div>
      )}

      {result && rq1 && rq2 && (
        <>
          <div className="st-facts">
            <div>
              <small>Prompts</small>
              <strong>{result.per_prompt.length}</strong>
              <span>
                excluded per condition: swap {result.exclusions.swap} · third {result.exclusions.third} · resample{" "}
                {result.exclusions.resample} · mean-ablation skipped {result.exclusions.abl_mean}
              </span>
            </div>
            <div>
              <small>Model accuracy</small>
              <strong>{(result.model_accuracy * 100).toFixed(1)}%</strong>
              <span>fraction of prompts with D_clean &gt; 0</span>
            </div>
            <div>
              <small>Seed</small>
              <strong>{result.params.seed}</strong>
              <span>{result.params.n_prompts} prompts requested</span>
            </div>
            <div>
              <small>Runtime</small>
              <strong>
                {result.elapsed_seconds >= 120 ? `${Math.round(result.elapsed_seconds / 60)} min` : `${Math.round(result.elapsed_seconds)} s`}
              </strong>
              <span>exclusion rule: {result.exclusion_rule}</span>
            </div>
          </div>

          <div className="st-tabs" role="tablist" aria-label="Study findings">
            {TABS.map((t, i) => (
              <button key={t.id} role="tab" aria-selected={tab === t.id} className={`st-tab${tab === t.id ? " on" : ""}`} onClick={() => setTab(t.id)}>
                <span className="st-tab-n">{i + 1}</span>
                <span className="st-tab-t">
                  <b>{t.label}</b>
                  <small>{t.hint}</small>
                </span>
              </button>
            ))}
          </div>

          {tab === "agreement" && (
            <Finding
              num="FINDING 1 · THE CHEAP METHODS"
              title="Do the cheap methods agree with exact patching?"
              intro="Exact patching is the reference. We check how well two cheaper methods rank the parts of the model in the same order, as a rank correlation averaged over the prompts."
              callouts={agreementCallouts(rq1)}
            >
              <div className="sw-cap">
                <span>Agreement with exact patching</span>
                <span className="live-badge">MEASURED</span>
              </div>
              <ForestPlot block={rq1} />
              <details className="sw-details">
                <summary>Show the full table (RQ1)</summary>
                <Rq1Table block={rq1} />
                {result.summary.rq1.subset_D_clean_gt0 && (
                  <details style={{ marginTop: "0.6rem" }}>
                    <summary className="inspector-meta">
                      Same statistics on the subset with D_clean &gt; 0 ({result.summary.rq1.subset_D_clean_gt0.n_prompts}{" "}
                      prompts), a pre-registered sensitivity check
                    </summary>
                    <Rq1Table block={result.summary.rq1.subset_D_clean_gt0} />
                  </details>
                )}
              </details>
            </Finding>
          )}

          {tab === "heads" && (
            <Finding
              num="FINDING 2 · THE TOP HEADS"
              title="Which heads matter most"
              intro="Averaged over all prompts, these ten heads score highest. Tags show which group of the published circuit each one belongs to. Choose a setup to see how the picture changes."
              callouts={topHeadCallouts(vector, known)}
            >
              <div className="sw-cap">
                <span>Score of each head, averaged over prompts</span>
                <span className="live-badge">MEASURED</span>
              </div>
              <div className="st-setups" role="radiogroup" aria-label="Setup">
                {[
                  { title: "Denoising", names: rq2.condition_names.filter((c) => !c.startsWith("abl")) },
                  { title: "Ablation", names: rq2.condition_names.filter((c) => c.startsWith("abl")) },
                ]
                  .filter((g) => g.names.length > 0)
                  .map((g) => (
                    <div className="st-setup-group" key={g.title}>
                      <span>{g.title}</span>
                      <div>
                        {g.names.map((c) => (
                          <button
                            key={c}
                            role="radio"
                            aria-checked={condition === c}
                            className={`st-chip${condition === c ? " on" : ""}`}
                            onClick={() => setCondition(c)}
                            title={c.replace(/_/g, " ")}
                          >
                            {conditionLabel(c)}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                <p className="st-key">D = how far the score gap is restored · P = how far the probability is restored.</p>
              </div>
              {vector ? (
                <div className="st-two">
                  <HeadsMap vector={vector} />
                  <TopList vector={vector} known={known} />
                </div>
              ) : (
                <p className="sw-note">No usable prompts for this setup.</p>
              )}
              <p className="st-key">
                Blue scores high, red works against the answer. Outlined = top ten. Rows are blocks, columns are heads.
              </p>
              <details className="sw-details">
                <summary>Top-10 heads per condition (full table)</summary>
                <TopHeads rq2={rq2} known={known} />
                {result.summary.anchors.mean_ranks_by_group && (
                  <p className="sw-note">
                    Known-heads anchor (descriptive only, rank 1 = highest prompt-averaged Rec_D[swap]):{" "}
                    {Object.entries(result.summary.anchors.mean_ranks_by_group)
                      .map(([name, g]) => `${name}: mean rank ${g.mean_rank?.toFixed(1) ?? "n/a"}`)
                      .join(" · ")}
                  </p>
                )}
              </details>
            </Finding>
          )}

          {tab === "stability" && (
            <Finding
              num="FINDING 3 · STABILITY"
              title="Do the rankings change with the setup?"
              intro="The same heads are ranked again under eight setups: three ways of corrupting the prompt, two metrics, and two kinds of ablation. The grid counts how many of the top heads two setups share."
              callouts={stabilityCallouts(rq2, topK === 5 ? "k5" : "k10")}
            >
              <div className="sw-cap">
                <span>Heads shared in the top {topK}</span>
                <span className="live-badge">MEASURED</span>
              </div>
              <div className="st-bar">
                <div className="mode-tabs" role="group" aria-label="How many top heads to compare">
                  <button className={topK === 5 ? "active" : ""} onClick={() => setTopK(5)}>
                    top 5
                  </button>
                  <button className={topK === 10 ? "active" : ""} onClick={() => setTopK(10)}>
                    top 10
                  </button>
                </div>
              </div>
              <OverlapGrid
                data={topK === 5 ? rq2.topk.k5 : rq2.topk.k10}
                k={topK}
                names={rq2.condition_names}
                expected={topK === 5 ? rq2.topk.expected_random.k5 : rq2.topk.expected_random.k10}
              />
              <details className="sw-details">
                <summary>Show both overlap tables</summary>
                <div className="grid-2" style={{ alignItems: "start" }}>
                  <div>
                    <h3>Top-5 overlap</h3>
                    <OverlapMatrix data={rq2.topk.k5} k={5} expectedRandom={rq2.topk.expected_random.k5} conditionNames={rq2.condition_names} />
                  </div>
                  <div>
                    <h3>Top-10 overlap</h3>
                    <OverlapMatrix data={rq2.topk.k10} k={10} expectedRandom={rq2.topk.expected_random.k10} conditionNames={rq2.condition_names} />
                  </div>
                </div>
              </details>
            </Finding>
          )}

          {tab === "limits" && (
            <Finding
              num="LIMITS"
              title="What these numbers do and do not show"
              intro="These are fixed statements, not computed from the results."
              callouts={[]}
            >
              <ul className="caveat-list">
                <li>
                  <strong>Attribution patching (AtP) is a first-order approximation</strong> of the exact patching effect; it
                  is known to be unreliable when attention saturates. Disagreement with exact patching is expected for some
                  components and is a result, not a bug.
                </li>
                <li>
                  <strong>DLA measures direct effects only.</strong> It ignores effects mediated by later layers, so it is a
                  partial view, not a patching replacement.
                </li>
                <li>
                  <strong>Denoising and ablation answer different questions.</strong> Denoising (copying clean activations into
                  the corrupted run) is sufficiency-style evidence; ablation (replacing a component&apos;s output in the clean
                  run) is necessity-style evidence. They are never mixed in one figure without labelling.
                </li>
                <li>
                  <strong>Exclusions:</strong> per (prompt, corruption) pair, prompts with |D_clean − D_corr| &lt; 0.5 are
                  excluded (counted above); no other prompt was dropped.
                </li>
              </ul>
              {jobId && <PlainWords jobId={jobId} />}
              {jobId && (
                <ReproduceButton
                  spec={{ kind: "study_subset", job_id: jobId, mode: "sample" }}
                  runtime="Recomputes exact Rec_D and Rec_AtP for the first 2 prompts (all 156 components); about 30–60 s."
                />
              )}
            </Finding>
          )}
        </>
      )}
    </main>
  );
}
