import type { ExperimentResult } from "../../api/client";
import { headline, recoveryPercent, residualBlock, triedKey, type Tried } from "./insights";
import type { RunState } from "./types";

type Run = { token: string; prob: number }[];

function Ring({ value }: { value: number | null }) {
  const size = 150;
  const R = size / 2 - 11;
  const C = 2 * Math.PI * R;
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <svg className="pg-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Recovery ${recoveryPercent(value)}`}>
      <circle cx={size / 2} cy={size / 2} r={R} fill="none" stroke="var(--bg-3)" strokeWidth={11} />
      <circle
        cx={size / 2} cy={size / 2} r={R} fill="none" stroke="var(--measure)" strokeWidth={11} strokeLinecap="round"
        strokeDasharray={`${C * v} ${C}`} transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="50%" textAnchor="middle" dy=".02em" className="pg-ring-n">
        {value == null ? "n/a" : `${Math.round(value * 100)}%`}
      </text>
      <text x="50%" y={size * 0.66} textAnchor="middle" className="pg-ring-s">recovered</text>
    </svg>
  );
}

function RunBars({ list, a, b, n = 4 }: { list: Run; a: string; b: string; n?: number }) {
  return (
    <div className="pg-bars">
      {list.slice(0, n).map((x) => (
        <div key={x.token} className={`pg-bar${x.token === a ? " a" : x.token === b ? " b" : ""}`}>
          <span className="t">{x.token}</span>
          <span className="tr"><i style={{ width: `${(x.prob * 100).toFixed(1)}%` }} /></span>
          <span className="v">{(x.prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

/** The story at the top of the page: click a word to choose where to patch,
 *  click a block to choose which one, then read the measured recovery. */
export default function Hero({
  prompt,
  tokens,
  position,
  setPosition,
  hook,
  setHook,
  tried,
  runState,
  result,
}: {
  prompt: string;
  tokens: string[];
  position: number;
  setPosition: (p: number) => void;
  hook: string;
  setHook: (h: string) => void;
  tried: Tried;
  runState: RunState;
  result: ExperimentResult | null;
}) {
  const chosen = residualBlock(hook);
  const m = result?.metrics ?? null;
  const recovery = m ? (m.Recovery ?? null) : null;
  const ctoks = result?.tokens_corrupted ?? null;
  const word = tokens[position] ?? "";
  return (
    <section className="pg-hero" aria-label="Choose where to patch">
      <p className="pg-hint">Click a word to choose <em>where</em> to patch, then a block to choose <em>which</em>.</p>
      {tokens.length > 0 ? (
        <p className="pg-sent">
          {tokens.map((t, i) => {
            if (i === 0 && t.startsWith("<|")) return null;
            const lead = t.startsWith(" ");
            const txt = t.trim() || t;
            const differs = ctoks != null && ctoks[i] !== undefined && ctoks[i] !== t;
            return (
              <span key={i}>
                {lead ? " " : ""}
                <button
                  type="button"
                  className={`pg-tok${i === position ? " sel" : ""}${differs ? " swap" : ""}`}
                  aria-pressed={i === position}
                  aria-label={`Patch at position ${i}: ${txt}`}
                  title={differs ? `position ${i}: this word differs between the two prompts` : `position ${i}`}
                  onClick={() => setPosition(i)}
                >
                  {txt}
                </button>
              </span>
            );
          })}
        </p>
      ) : (
        <p className="pg-sent pg-sent-plain">{prompt}</p>
      )}
      <div className="pg-blocks" role="group" aria-label="Residual stream block to patch">
        {Array.from({ length: 12 }, (_, b) => {
          const h = `blocks.${b}.hook_out`;
          const r = tried[triedKey(h, position)];
          const shade = r == null ? 0 : Math.max(0, Math.min(1, r));
          return (
            <button
              key={b}
              type="button"
              className={`pg-blk${chosen === b ? " on" : ""}${r != null ? " tried" : ""}`}
              style={r != null ? { background: `color-mix(in srgb, var(--measure) ${Math.round(shade * 80)}%, var(--bg-1))`, color: shade > 0.55 ? "#fff" : undefined } : undefined}
              aria-pressed={chosen === b}
              aria-label={`Patch block ${b}`}
              onClick={() => setHook(h)}
            >
              {b}
              {r != null && <small>{Math.round(r * 100)}</small>}
            </button>
          );
        })}
      </div>
      <p className="pg-legend">
        Blocks you have run at this word are shaded by measured recovery (number = %).
        {chosen == null && " Another site is selected below, such as an attention or MLP output."}
      </p>

      {runState === "complete" && result && m ? (
        <div className="pg-answer">
          <div className="pg-answer-top">
            <Ring value={recovery} />
            <div>
              <span className="pg-badge">MEASURED</span>
              <p className="pg-headline">{headline(result.patch.hook, result.tokens_clean[result.patch.positions[0]] ?? word, recovery)}</p>
              <p className="pg-sub">
                D (logit of {JSON.stringify(result.config.candidate_a)} minus {JSON.stringify(result.config.candidate_b)}) went
                from <b>{m.D_corrupted?.toFixed(2)}</b> to <b>{m.D_patched?.toFixed(2)}</b>; the clean run scored <b>{m.D_clean?.toFixed(2)}</b>.
              </p>
            </div>
          </div>
          <div className="pg-ba">
            <div>
              <h3>Before: corrupted prompt</h3>
              <RunBars list={result.runs.corrupted} a={result.config.candidate_a} b={result.config.candidate_b} />
            </div>
            <span className="pg-arrow" aria-hidden>→</span>
            <div>
              <h3>After: patched</h3>
              <RunBars list={result.runs.patched} a={result.config.candidate_a} b={result.config.candidate_b} />
            </div>
          </div>
        </div>
      ) : (
        <p className="pg-empty">
          {runState === "running"
            ? "Running the model three times: clean, corrupted, then patched…"
            : "Press Run experiment to measure how much of the answer this patch brings back."}
        </p>
      )}
    </section>
  );
}
