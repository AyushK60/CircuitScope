import { api } from "../../api/client";
import { PRESET } from "./presets";
import type { RunState } from "./types";

/** "Guided experiment tour" panel: curated examples + the four-step wizard. */
export default function GuidedWizard({
  comparison,
  comparisonError,
  wizardStep,
  setWizardStep,
  runState,
  runExperiment,
  clean,
  corrupted,
  candA,
  candB,
  setClean,
  setCorrupted,
  setCandA,
  setCandB,
  setRunState,
  setRecord,
  hook,
}: {
  comparison: { D: number; prob_a: number; prob_b: number } | null;
  comparisonError: string | null;
  wizardStep: number;
  setWizardStep: (s: number) => void;
  runState: RunState;
  runExperiment: () => void;
  clean: string;
  corrupted: string;
  candA: string;
  candB: string;
  setClean: (v: string) => void;
  setCorrupted: (v: string) => void;
  setCandA: (v: string) => void;
  setCandB: (v: string) => void;
  setRunState: (s: RunState) => void;
  setRecord: (r: null) => void;
  hook: string;
}) {
  return (
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
  );
}
