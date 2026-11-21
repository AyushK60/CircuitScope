import { ANALOGIES, GLOSSARY, type ReadingLevel } from "../lib/glossary";

/**
 * Full glossary listing: reachable from the Report and Learn
 * pages. Every term with all three levels; analogies with their breaks_when.
 */
export default function GlossaryList({ level = "curious" }: { level?: ReadingLevel }) {
  const entries = Object.entries(GLOSSARY);
  return (
    <section className="panel" aria-labelledby="glossary-heading">
      <h2 id="glossary-heading">Glossary</h2>
      <p className="inspector-meta">
        Every technical term CircuitScope uses, at three depths. The same definitions power the
        dotted-underline terms in the report.
      </p>
      <dl className="glossary-list">
        {entries.map(([id, t]) => (
          <div key={id} className="glossary-entry">
            <dt>{t.display}</dt>
            <dd>
              <p>
                <strong>Curious:</strong> {t.curious}
              </p>
              <p>
                <strong>Student:</strong> {t.student}
              </p>
              <p>
                <strong>Expert:</strong> {t.expert}
              </p>
              {t.analogy && (
                <p className="inspector-meta">
                  <em>Analogy:</em> {t.analogy.text} — <em>where it breaks:</em> {t.analogy.breaks_when}
                </p>
              )}
              {t.see_also && t.see_also.length > 0 && (
                <p className="inspector-meta">See also: {t.see_also.map((o) => GLOSSARY[o]?.display ?? o).join(", ")}</p>
              )}
            </dd>
          </div>
        ))}
      </dl>
      <h3>Analogies</h3>
      <ul className="analogy-list">
        {ANALOGIES.map((a) => (
          <li key={a.id}>
            <strong>Think of it like {a.title}</strong> — an analogy, not a description of the model. {a.text}{" "}
            <em>Where it breaks:</em> {a.breaks_when}
          </li>
        ))}
      </ul>
      <p className="inspector-meta" style={{ marginTop: "0.5rem" }}>
        Reading level shown in the report: {level}.
      </p>
    </section>
  );
}
