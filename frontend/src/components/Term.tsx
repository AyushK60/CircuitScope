import { useEffect, useId, useRef, useState } from "react";
import { GLOSSARY, type ReadingLevel } from "../lib/glossary";

/**
 * Glossary term: a keyboard-focusable button with a dotted
 * underline that opens an accessible popover. Escape closes; clicking outside
 * closes; works on touch (onClick); aria-describedby links button to popover.
 */
export default function Term({
  id,
  level = "curious",
  label,
}: {
  id: string;
  level?: ReadingLevel;
  label?: string;
}) {
  const term = GLOSSARY[id];
  const [open, setOpen] = useState(false);
  const popId = useId();
  const rootRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  if (!term) {
    // unknown term ids are a defect caught by tests; render plain text here
    return <>{label ?? id}</>;
  }

  const definition = term[level] ?? term.curious;

  return (
    <span ref={rootRef} style={{ position: "relative", display: "inline-block" }}>
      <button
        type="button"
        className="term"
        aria-expanded={open}
        aria-describedby={open ? popId : undefined}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Escape" && open) {
            e.stopPropagation();
            setOpen(false);
          }
        }}
      >
        {label ?? term.display}
      </button>
      {open && (
        <span
          id={popId}
          role="note"
          aria-label={`Glossary: ${term.display}`}
          className="term-pop"
        >
          <strong>{term.display}</strong>
          <span className="term-pop-body">{definition}</span>
          {term.analogy && (
            <span className="term-pop-analogy">
              <em>Analogy:</em> {term.analogy.text} <em>Where it breaks:</em> {term.analogy.breaks_when}
            </span>
          )}
          {term.see_also && term.see_also.length > 0 && (
            <span className="term-pop-also">
              See also: {term.see_also.map((other) => GLOSSARY[other]?.display ?? other).join(", ")}
            </span>
          )}
        </span>
      )}
    </span>
  );
}
