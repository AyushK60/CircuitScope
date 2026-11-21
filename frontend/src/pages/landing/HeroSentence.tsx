import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, type Candidate, type FlowData } from "../../api/client";

/* The landing hero. One sentence runs through the real model; the arcs show how
 * much attention the last word pays to each earlier word (measured, averaged over
 * every head), the ribbon shows the measured size of the residual stream after
 * each block, and clicking a word re-runs the model without it.
 * When the model is not available the page shows a stored reference result,
 * clearly labelled, and the words are not clickable. */

const SENTENCE = "John and Mary went to the shop. John gave a book to";
const STATIC_TOKENS = ["John", " and", " Mary", " went", " to", " the", " shop", ".", " John", " gave", " a", " book", " to"];
/* Measured on this project's own stored report (Tour and Report pages): " Mary" 66.2 %, " John" 5.4 %. */
const REFERENCE_TOP: Candidate[] = [
  { token: " Mary", token_id: -1, logit: 0, prob: 0.662 },
  { token: " John", token_id: -2, logit: 0, prob: 0.054 },
];
/* From scripts/reference_patch.py: patching block 5's residual at the second " John". */
const REFERENCE_RECOVERY = 0.9148;
const PATCH_BLOCK = 5;
const ARC_H = 150;

type NavRoute = "explorer" | "playground" | "tour";

const pct = (p: number) => `${(p * 100).toFixed(1)}%`;
const word = (t: string) => t.trim();

export default function HeroSentence({
  live,
  navigate,
}: {
  live: boolean;
  navigate: (r: NavRoute) => void;
}) {
  const [flow, setFlow] = useState<FlowData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hidden, setHidden] = useState<number | null>(null);
  const [alt, setAlt] = useState<Candidate[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [xs, setXs] = useState<number[]>([]);
  const request = useRef(0);
  const touched = useRef(false);
  const wrap = useRef<HTMLDivElement>(null);
  const spans = useRef<(HTMLSpanElement | null)[]>([]);

  useEffect(() => {
    if (!live || flow) return;
    let alive = true;
    api
      .flow(SENTENCE)
      .then((f) => alive && setFlow(f))
      .catch((e) => alive && setError(String(e.message ?? e)));
    return () => {
      alive = false;
    };
  }, [live, flow]);

  const skip = flow && flow.tokens[0]?.startsWith("<|") ? 1 : 0;
  const tokens = useMemo(() => (flow ? flow.tokens.slice(skip) : STATIC_TOKENS), [flow, skip]);
  const last = tokens.length - 1;

  /* mean attention paid by the last token to each earlier token, over all blocks and heads */
  const attn = useMemo(() => {
    if (!flow) return null;
    const n = flow.tokens.length;
    const sum = new Array<number>(n).fill(0);
    let count = 0;
    for (const block of flow.blocks) {
      for (const head of block.attention) {
        const row = head[n - 1];
        if (!row) continue;
        for (let k = 0; k < n; k++) sum[k] += row[k];
        count++;
      }
    }
    if (!count) return null;
    const mean = sum.slice(skip, n - 1).map((v) => v / count);
    const max = Math.max(...mean, 1e-9);
    const rel = mean.map((v) => v / max);
    return { rel, best: rel.indexOf(1) };
  }, [flow, skip]);

  const norms = flow ? flow.blocks.map((b) => b.resid_norm_mean) : null;

  /* the repeated name before the last word: the classic example this page is built around */
  const demoIndex = useMemo(() => {
    if (!flow) return -1;
    const first = word(tokens[0]);
    for (let i = last - 1; i > 0; i--) if (word(tokens[i]) === first) return i;
    return attn ? attn.best : -1;
  }, [flow, tokens, last, attn]);

  const choose = (i: number | null) => {
    setHidden(i);
    const id = ++request.current;
    if (i === null) {
      setAlt(null);
      setBusy(false);
      return;
    }
    setBusy(true);
    const text = tokens.filter((_, j) => j !== i).join("").trim();
    api
      .predict(text, 5)
      .then((r) => {
        if (id !== request.current) return;
        setAlt(r.prediction.top_predictions);
        setBusy(false);
      })
      .catch((e) => {
        if (id !== request.current) return;
        setBusy(false);
        setHidden(null);
        setError(String(e.message ?? e));
      });
  };
  const chooseRef = useRef(choose);
  useEffect(() => {
    chooseRef.current = choose;
  });

  /* one automatic demonstration, then it stays out of the way */
  useEffect(() => {
    if (!flow || demoIndex < 0) return;
    const t1 = setTimeout(() => !touched.current && chooseRef.current(demoIndex), 2600);
    const t2 = setTimeout(() => !touched.current && chooseRef.current(null), 8000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [flow, demoIndex]);

  /* arc anchors follow the rendered words */
  useLayoutEffect(() => {
    const measure = () => {
      const box = wrap.current?.getBoundingClientRect();
      if (!box) return;
      setXs(
        tokens.map((_, i) => {
          const r = spans.current[i]?.getBoundingClientRect();
          return r ? r.left + r.width / 2 - box.left : 0;
        }),
      );
    };
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    if (ro && wrap.current) ro.observe(wrap.current);
    document.fonts?.ready.then(measure);
    return () => ro?.disconnect();
  }, [tokens]);

  const base = flow?.top_predictions ?? REFERENCE_TOP;
  const cur = hidden !== null && alt ? alt : base;
  const top = cur[0];
  const second = cur[1];
  const flipped = !!top && !!base[0] && top.token !== base[0].token;
  const hiddenWord = hidden !== null ? word(tokens[hidden]) : null;

  let note: string;
  if (!flow) {
    note = live
      ? error
        ? "The model is up but this sentence could not be run. The numbers shown are a stored reference result."
        : "Running the sentence through the model…"
      : "Start the model to run this on your own computer. The numbers shown are a stored reference result.";
  } else if (hiddenWord === null) {
    note = `Top word for the sentence, measured live. Arcs: how much the last word attends to each earlier word, averaged over all ${flow.blocks.length * (flow.blocks[0]?.attention.length ?? 0)} heads.`;
  } else if (busy) {
    note = `Re-running the model without “${hiddenWord}”…`;
  } else {
    note = `Without “${hiddenWord}” the model's top word is “${word(top.token)}” (${pct(top.prob)}). ${flipped ? "The answer changed." : "The answer did not change."}`;
  }

  const maxNorm = norms ? Math.max(...norms, 1e-9) : 1;
  const arcs = attn
    ? attn.rel
        .map((rel, i) => ({ rel, i }))
        .filter(({ rel, i }) => rel > 0.08 && xs[i] && xs[last])
        .sort((a, b) => a.rel - b.rel)
    : [];

  return (
    <section className="lh-hero" aria-labelledby="lh-title">
      <p className="lh-kicker">An observatory for a real language model</p>
      <h1 id="lh-title" className="lh-title">
        When GPT-2 finishes this sentence, which words is it looking at?
      </h1>

      <div className={`lh-sentence-wrap${flow ? " has-arcs" : ""}`} ref={wrap}>
        <svg className={`lh-arcs${hidden !== null ? " stale" : ""}`} height={ARC_H} aria-hidden>
          {arcs.map(({ rel, i }) => {
            const x = xs[i];
            const xl = xs[last];
            const h = Math.min(ARC_H - 12, 28 + Math.abs(xl - x) * 0.28);
            const isBest = attn?.best === i;
            return (
              <g key={i}>
                <path
                  d={`M ${x} ${ARC_H} C ${x} ${ARC_H - h}, ${xl} ${ARC_H - h}, ${xl} ${ARC_H}`}
                  className={`lh-arc${isBest ? " best" : ""}`}
                  style={{ strokeWidth: 1 + rel * 6, opacity: 0.22 + rel * 0.68 }}
                />
              </g>
            );
          })}
        </svg>
        <p className="lh-sentence" aria-label={tokens.join("").trim()}>
          {tokens.map((t, i) => {
            const m = /^(\s*)([\s\S]*)$/.exec(t) as RegExpExecArray;
            const isLast = i === last;
            const clickable = !!flow && !isLast;
            return (
              <span key={i}>
                {m[1]}
                <span
                  ref={(el) => {
                    spans.current[i] = el;
                  }}
                  className={`lh-word${hidden === i ? " off" : ""}${isLast ? " last" : ""}${clickable ? " click" : ""}${attn?.best === i && hidden === null ? " best" : ""}`}
                  role={clickable ? "button" : undefined}
                  tabIndex={clickable ? 0 : undefined}
                  aria-pressed={clickable ? hidden === i : undefined}
                  aria-label={clickable ? `Hide the word ${m[2]} and re-run the model` : undefined}
                  onClick={clickable ? () => {
                    touched.current = true;
                    choose(hidden === i ? null : i);
                  } : undefined}
                  onKeyDown={clickable ? (e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      touched.current = true;
                      choose(hidden === i ? null : i);
                    }
                  } : undefined}
                >
                  {m[2]}
                </span>
              </span>
            );
          })}
          <span className="lh-gap"> …</span>
        </p>
      </div>
      <p className="lh-hint">
        {flow
          ? "Arcs show where the last word looks back; red marks the most attended word. Click any word to remove it and re-run the model."
          : "The words become clickable once the model is ready."}
      </p>

      <div className="lh-row">
        <div className="lh-ribbon-col">
          <div className="lh-cap">
            <span>12 blocks · size of the signal after each one</span>
            <span className={flow ? "live-badge" : "static-badge"}>{flow ? "MEASURED · LIVE" : "WAITING FOR MODEL"}</span>
          </div>
          <ol className="lh-ribbon" aria-label="Residual stream size after each of the 12 blocks">
            {Array.from({ length: 12 }, (_, b) => {
              const v = norms?.[b];
              const fill = v !== undefined ? Math.max(0.06, v / maxNorm) : 0;
              const label =
                v !== undefined ? `Block ${b} · mean residual norm ${v.toFixed(1)}` : `Block ${b} · no measurement yet`;
              return (
                <li
                  key={b}
                  className={`lh-block${b === PATCH_BLOCK ? " patch" : ""}`}
                  style={{ ["--fill" as string]: `${fill * 100}%`, ["--i" as string]: b }}
                  title={label}
                  aria-label={label}
                >
                  <span className="lh-track">
                    <span className="lh-fill" />
                  </span>
                  <span className="lh-bnum">{b}</span>
                  {b === PATCH_BLOCK && (
                    <span className="lh-pin">
                      patch here
                      <small>{pct(REFERENCE_RECOVERY)} restored · reference run</small>
                    </span>
                  )}
                </li>
              );
            })}
          </ol>
        </div>

        <div className={`lh-answer${flipped ? " flip" : ""}`} aria-live="polite">
          <span className="lh-cap-small">The model's next word</span>
          <div className="lh-big">
            <span>{top ? word(top.token) : "…"}</span>
            <i>{top ? pct(top.prob) : ""}</i>
          </div>
          {second && (
            <p className="lh-second">
              then <b>{word(second.token)}</b> {pct(second.prob)}
            </p>
          )}
          <p className="lh-note">
            {!flow && <span className="static-badge">REFERENCE</span>} {note}
          </p>
        </div>
      </div>

      <div className="lh-cta">
        <button className="primary" onClick={() => navigate("explorer")}>
          Explore the real model →
        </button>
        <button onClick={() => navigate("tour")}>90-second tour</button>
        <button className="ghost" onClick={() => navigate("playground")}>
          Try an intervention
        </button>
      </div>
    </section>
  );
}
