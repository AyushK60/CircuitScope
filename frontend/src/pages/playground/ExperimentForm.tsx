import { api, type Tokenization } from "../../api/client";
import { EXAMPLES, HOOK_OPTIONS, PRESET } from "./presets";
import TargetStrip from "./TargetStrip";
import type { RunState } from "./types";

/** The control rail: guided examples, prompts, candidates, patch site and the
 *  run button. It stays in view beside the story on wide screens. */
export default function ExperimentForm({
  ready,
  clean,
  setClean,
  corrupted,
  setCorrupted,
  candA,
  setCandA,
  candB,
  setCandB,
  hook,
  setHook,
  position,
  setPosition,
  tokenization,
  comparison,
  comparisonError,
  runState,
  jobId,
  runExperiment,
  setRunState,
  setRecord,
  setJobId,
}: {
  ready: boolean;
  clean: string;
  setClean: (v: string) => void;
  corrupted: string;
  setCorrupted: (v: string) => void;
  candA: string;
  setCandA: (v: string) => void;
  candB: string;
  setCandB: (v: string) => void;
  hook: string;
  setHook: (v: string) => void;
  position: number;
  setPosition: (v: number) => void;
  tokenization: Tokenization | null;
  comparison: { D: number; prob_a: number; prob_b: number } | null;
  comparisonError: string | null;
  runState: RunState;
  jobId: string | null;
  runExperiment: () => void;
  setRunState: (s: RunState) => void;
  setRecord: (r: null) => void;
  setJobId: (id: string | null) => void;
}) {
  const pickExample = (ex: (typeof EXAMPLES)[number]) => {
    setClean(ex.clean);
    setCorrupted(ex.corrupted);
    setCandA(ex.candidates[0]);
    setCandB(ex.candidates[1]);
    setRunState("idle");
    setRecord(null);
    setJobId(null);
  };
  const activeExample = EXAMPLES.find((e) => e.clean === clean && e.corrupted === corrupted)?.id ?? null;

  return (
    <aside className="pg-rail" aria-label="Experiment setup">
      <div className="pg-rail-body">
        <section>
          <h2>Guided experiments</h2>
          <p className="pg-note">Each one fills in the controls below and runs real model passes. Measured results may differ from any expectation.</p>
          <div className="pg-chips">
            {EXAMPLES.map((ex) => (
              <button key={ex.id} type="button" className={`pg-chip${activeExample === ex.id ? " on" : ""}`} onClick={() => pickExample(ex)}>
                {ex.label}
              </button>
            ))}
            <a className="pg-chip" href="#/explorer">Attention inspection →</a>
          </div>
          {clean === corrupted && (
            <p className="pg-warn">
              This example is about prediction: the two prompts are identical, so a patch has nothing to contrast.
              Choose “Activation patch” for a valid pair.
            </p>
          )}
        </section>

        <section>
          <h2>Prompts</h2>
          <label htmlFor="clean-text">Clean prompt</label>
          <textarea id="clean-text" rows={2} value={clean} onChange={(e) => setClean(e.target.value)} />
          <label htmlFor="corr-text">Corrupted prompt (same length, one word swapped)</label>
          <textarea id="corr-text" rows={2} value={corrupted} onChange={(e) => setCorrupted(e.target.value)} />
          <div className="pg-two">
            <div>
              <label htmlFor="cand-a">Candidate A</label>
              <input id="cand-a" type="text" value={candA} onChange={(e) => setCandA(e.target.value)} />
            </div>
            <div>
              <label htmlFor="cand-b">Candidate B</label>
              <input id="cand-b" type="text" value={candB} onChange={(e) => setCandB(e.target.value)} />
            </div>
          </div>
          <p className="pg-note">Each must be one token (a leading space matters). The comparison updates as you type.</p>
          {comparisonError && <div className="state-banner warn">Candidate comparison unavailable: {comparisonError}</div>}
          {comparison && (
            <dl className="pg-live">
              <div>
                <dt>D · logit(A) − logit(B)</dt>
                <dd>{comparison.D.toFixed(3)}</dd>
              </div>
              <div>
                <dt>P(A) · {JSON.stringify(candA)}</dt>
                <dd>{(comparison.prob_a * 100).toFixed(2)}%</dd>
              </div>
              <div>
                <dt>P(B) · {JSON.stringify(candB)}</dt>
                <dd>{(comparison.prob_b * 100).toFixed(2)}%</dd>
              </div>
            </dl>
          )}
          <p className="pg-note">Measured on the clean prompt. Positive D means A is preferred at the final position.</p>
        </section>

        <section>
          <h2>Where to patch</h2>
          <label htmlFor="hook-sel">Activation to replace (clean → corrupted)</label>
          <select id="hook-sel" value={hook} onChange={(e) => setHook(e.target.value)}>
            {HOOK_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <label htmlFor="pos-sel">Token position to patch</label>
          <select id="pos-sel" value={position} onChange={(e) => setPosition(Number(e.target.value))}>
            {(tokenization?.tokens ?? []).map((t, i) => (
              <option key={i} value={i}>
                {i}: {JSON.stringify(t)}
              </option>
            ))}
          </select>
          <TargetStrip hook={hook} position={position} />
        </section>
      </div>

      <div className="pg-rail-run">
        <button className="primary" onClick={runExperiment} disabled={!ready || runState === "running"}>
          {runState === "running" ? "Running 3 model passes…" : "Run experiment"}
        </button>
        {runState === "running" && jobId && (
          <button type="button" onClick={() => api.cancelExperiment(jobId)}>Cancel</button>
        )}
        <button
          type="button"
          className="ghost"
          onClick={() => {
            setClean(PRESET.clean);
            setCorrupted(PRESET.corrupted);
            setCandA(PRESET.candidate_a);
            setCandB(PRESET.candidate_b);
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
    </aside>
  );
}
