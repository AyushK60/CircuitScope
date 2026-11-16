import { api, type Tokenization } from "../../api/client";
import TokenRow from "../../components/TokenRow";
import { HOOK_OPTIONS, PRESET } from "./presets";
import TargetStrip from "./TargetStrip";
import type { RunState } from "./types";

/** "1 · Choose the experiment" panel: prompts, candidates, patch target and
 *  the run button. */
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
  return (
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
  );
}
