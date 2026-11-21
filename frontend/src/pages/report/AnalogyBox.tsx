/** "Think of it like..." box: fixed analogies with a
 * mandatory breaks_when line, always labelled as an analogy. */
export default function AnalogyBox({
  analogy,
}: {
  analogy: { id: string; title: string; text: string; breaks_when: string };
}) {
  return (
    <aside className="analogy-box" aria-label="Analogy">
      <p className="analogy-title">
        Think of it like {analogy.title} — <strong>an analogy, not a description of the model</strong>
      </p>
      <p>{analogy.text}</p>
      <p className="analogy-breaks">
        <em>Where the analogy breaks:</em> {analogy.breaks_when}
      </p>
    </aside>
  );
}
