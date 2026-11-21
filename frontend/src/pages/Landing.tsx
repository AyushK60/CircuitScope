import type { Health } from "../api/client";
import HeroSentence from "./landing/HeroSentence";

type NavRoute = "explorer" | "playground" | "learn" | "lab" | "sweeps" | "study" | "report" | "verify" | "tour";

const STEPS: { title: string; body: string; label: string; to: NavRoute }[] = [
  {
    title: "Look",
    body: "Follow a sentence through 12 blocks and 144 attention heads, and read the measurements inside.",
    label: "Open the Model Explorer",
    to: "explorer",
  },
  {
    title: "Change",
    body: "Copy one internal signal from a clean run into a corrupted one and see what the prediction does.",
    label: "Open the Playground",
    to: "playground",
  },
  {
    title: "Check",
    body: "Independent reference scripts recompute every headline number and report how far they differ.",
    label: "Open Verify",
    to: "verify",
  },
];

export default function Landing({
  health,
  navigate,
}: {
  health: Health | null;
  navigate: (r: NavRoute) => void;
}) {
  return (
    <main className="lh">
      <HeroSentence live={health?.model_loaded === true} navigate={navigate} />
      <section className="lh-steps" aria-label="What you can do here">
        <p className="lh-what">
          <strong>CircuitScope</strong> runs GPT-2 on your own computer, lets you look inside it and change one
          internal signal, then measures what changes. Nothing is simulated, and nothing leaves your machine.
        </p>
        <div className="lh-step-grid">
          {STEPS.map((s, i) => (
            <button key={s.title} className="lh-step" onClick={() => navigate(s.to)}>
              <span className="lh-step-num">{i + 1}</span>
              <span className="lh-step-title">{s.title}</span>
              <span className="lh-step-body">{s.body}</span>
              <span className="lh-step-link">{s.label} →</span>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}
