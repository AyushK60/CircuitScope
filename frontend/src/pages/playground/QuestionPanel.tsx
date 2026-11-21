import type { QuestionResult } from "../../api/client";

/** "Question & Candidate Explorer" panel: scores candidate continuations
 *  under a fixed prompt template with the real model. */
export default function QuestionPanel({
  ready,
  qContext,
  setQContext,
  qQuestion,
  setQQuestion,
  qCandidates,
  setQCandidates,
  qResult,
  setQResult,
  qBusy,
  qError,
  runQuestion,
}: {
  ready: boolean;
  qContext: string;
  setQContext: (v: string) => void;
  qQuestion: string;
  setQQuestion: (v: string) => void;
  qCandidates: string;
  setQCandidates: (v: string) => void;
  qResult: QuestionResult | null;
  setQResult: (r: QuestionResult | null) => void;
  qBusy: boolean;
  qError: string | null;
  runQuestion: () => void;
}) {
  return (
    <div className="pg-card">
      <h2>Question &amp; Candidate Explorer</h2>
      <p className="inspector-meta" style={{ marginBottom: "0.8rem" }}>
        A context, a question, and candidate answers. GPT-2 is a next-token model, <strong>not</strong> a
        question-answering assistant - this compares text-continuation scores under a fixed prompt
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
                  ? "⚠ candidates have different token lengths - compare mean log-P as well; raw totals favor longer candidates."
                  : "Candidates have equal token lengths; total and mean log-P rank them identically."}{" "}
                These are continuation scores - not a claim that GPT-2 answers questions correctly.
              </p>
            </>
          )}
          {!qResult && (
            <div className="state-banner info">
              Enter a context, question and candidates, then score. Scores are conditional log-probabilities
              under the fixed template - see the method note in the result.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
