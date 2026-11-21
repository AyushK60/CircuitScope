import { useEffect, useMemo, useRef, useState } from "react";
import type { FlowData, ModelMeta } from "../../api/client";
import AttentionHeatmap from "../../components/AttentionHeatmap";
import TokenRow from "../../components/TokenRow";
import TopPredictions from "../../components/TopPredictions";

/* A guided walk through ONE real forward pass (backend /api/flow): the rail shows
 * every stage, the panel explains the selected one in plain words and gives the
 * technical facts beside it. Every number and heatmap comes from the capture;
 * the automatic stepping is presentation pacing, not GPU timing. */

type Stage =
  | { kind: "tokenize" }
  | { kind: "embed" }
  | { kind: "block"; layer: number }
  | { kind: "lnf" }
  | { kind: "logits" };

const STAGE_MS = 2200;
const word = (t: string) => t.trim() || (t.includes("\n") ? "↵" : "␣");
const pct = (v: number) => `${Math.round(v * 100)}%`;

function stageLabel(s: Stage): string {
  switch (s.kind) {
    case "tokenize":
      return "Tokens";
    case "embed":
      return "Embeddings";
    case "block":
      return `Block ${s.layer}`;
    case "lnf":
      return "Final norm";
    case "logits":
      return "Answer";
  }
}

export default function GuidedTour({ flow, meta }: { flow: FlowData | null; meta: ModelMeta }) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [head, setHead] = useState(5);
  const [seen, setSeen] = useState(flow);
  if (flow !== seen) {
    setSeen(flow);
    setIdx(0);
    setPlaying(false);
  }

  const stages = useMemo<Stage[]>(() => {
    if (!flow) return [];
    return [
      { kind: "tokenize" },
      { kind: "embed" },
      ...Array.from({ length: flow.n_layers }, (_, l) => ({ kind: "block", layer: l }) as Stage),
      { kind: "lnf" },
      { kind: "logits" },
    ];
  }, [flow]);

  const count = stages.length;
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setIdx((i) => {
        if (i >= count - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, STAGE_MS);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, count]);

  if (!flow) return <div className="skeleton" style={{ height: "26rem" }} />;

  const stage = stages[idx];
  const finished = idx === count - 1 && !playing;
  const maxNorm = Math.max(...flow.blocks.map((b) => b.resid_norm_mean), 1e-9);
  const go = (i: number) => {
    setPlaying(false);
    setIdx(Math.max(0, Math.min(count - 1, i)));
  };

  return (
    <div className="ex-guided">
      <div className="ex-controls" role="group" aria-label="Playback controls">
        <button
          className="primary"
          onClick={() => {
            if (finished) setIdx(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? "❚❚ Pause" : finished ? "↻ Replay" : "▶ Play the walk"}
        </button>
        <button onClick={() => go(idx - 1)} disabled={idx === 0} aria-label="Previous stage">
          ← Back
        </button>
        <button onClick={() => go(idx + 1)} disabled={idx === count - 1} aria-label="Next stage">
          Next →
        </button>
        <span className="ex-where">
          Stage {idx + 1} of {count} · {stageLabel(stage)}
        </span>
      </div>

      <div
        className="ex-rail"
        role="tablist"
        aria-label="Stages of one forward pass"
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") {
            e.preventDefault();
            go(idx + 1);
          } else if (e.key === "ArrowLeft") {
            e.preventDefault();
            go(idx - 1);
          }
        }}
      >
        {stages.map((s, i) => {
          const small = s.kind !== "block";
          const fill = s.kind === "block" ? Math.max(0.06, flow.blocks[s.layer].resid_norm_mean / maxNorm) : 0;
          const cls = `ex-tile${small ? " small" : ""}${i === idx ? " on" : i > idx ? " future" : ""}${
            i === 2 || i === count - 2 ? " gap" : ""
          }`;
          const label = stageLabel(s);
          return (
            <button
              key={i}
              role="tab"
              aria-selected={i === idx}
              aria-label={label}
              tabIndex={i === idx ? 0 : -1}
              className={cls}
              style={{ ["--fill" as string]: `${fill * 100}%` }}
              title={
                s.kind === "block"
                  ? `Block ${s.layer} · mean residual norm ${flow.blocks[s.layer].resid_norm_mean.toFixed(1)}`
                  : label
              }
              onClick={() => go(i)}
            >
              <span className="ex-track">
                {small ? <span className="ex-glyph">{GLYPH[s.kind]}</span> : <span className="ex-fill" />}
              </span>
              <span className="ex-tlabel">{s.kind === "block" ? s.layer : label}</span>
            </button>
          );
        })}
      </div>
      <p className="ex-rail-note">
        Gauge height = measured size of the model's running notes after each block (mean over tokens).
      </p>

      <section className="ex-stage" aria-live="polite">
        <div className="ex-copy">
          <p className="ex-kicker">
            Stage {idx + 1} of {count}
          </p>
          <StageCopy stage={stage} flow={flow} meta={meta} head={head} />
        </div>
        <div className="ex-visual">
          <StageVisual stage={stage} flow={flow} head={head} setHead={setHead} />
        </div>
      </section>
      <p className="ex-pacing">
        Everything shown is measured from one real run on your computer. The automatic stepping is only pacing,
        not GPU timing.
      </p>
    </div>
  );
}

const GLYPH: Record<string, string> = { tokenize: "Aa", embed: "[ ⋯ ]", lnf: "σ", logits: "→ ?" };

function Facts({ items }: { items: [string, string][] }) {
  return (
    <dl className="ex-facts">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function blockNotice(flow: FlowData, layer: number, head: number) {
  const n = flow.tokens.length;
  const row = flow.blocks[layer].attention[head]?.[n - 1];
  if (!row) return null;
  let best = 1;
  for (let k = 2; k < n - 1; k++) if (row[k] > row[best]) best = k;
  return { query: word(flow.tokens[n - 1]), sink: row[0], best: word(flow.tokens[best]), bestW: row[best] };
}

function StageCopy({
  stage,
  flow,
  meta,
  head,
}: {
  stage: Stage;
  flow: FlowData;
  meta: ModelMeta;
  head: number;
}) {
  switch (stage.kind) {
    case "tokenize":
      return (
        <>
          <h2>Your sentence becomes tokens</h2>
          <p>
            The model cannot read letters. It first chops the text into pieces called tokens, usually words or
            word fragments, and swaps each piece for a number from a fixed list of {meta.d_vocab.toLocaleString()}.
          </p>
          <p className="ex-notice">
            Your sentence became <b>{flow.tokens.length}</b> tokens. A leading space belongs to the word that
            follows it.
          </p>
          <Facts
            items={[
              ["tokenizer", `${meta.tokenizer_class}, byte-level BPE`],
              ["vocabulary", meta.d_vocab.toLocaleString()],
              ["special token", "<|endoftext|> is placed first"],
            ]}
          />
        </>
      );
    case "embed":
      return (
        <>
          <h2>Each token becomes a list of numbers</h2>
          <p>
            Every token number is looked up in a learned table, which gives a list of {meta.d_model} numbers. A
            second table adds where the token sits in the sentence. Their sum starts the model's running notes,
            called the residual stream.
          </p>
          <Facts
            items={[
              ["token table W_E", `${meta.d_vocab.toLocaleString()} × ${meta.d_model}`],
              ["position table W_pos", `${meta.n_ctx} × ${meta.d_model}`],
              ["start of stream", "x₀ = W_E[token] + W_pos[position]"],
            ]}
          />
        </>
      );
    case "block": {
      const l = stage.layer;
      const note = blockNotice(flow, l, head);
      const now = flow.blocks[l].resid_norm_mean;
      const prev = l > 0 ? flow.blocks[l - 1].resid_norm_mean : null;
      return (
        <>
          <h2>Block {l}: read, think, write back</h2>
          <p>
            Each block does two jobs on the running notes. <b>Attention</b> lets every word look back at earlier
            words and copy useful information. Then a small network, the <b>MLP</b>, works on each word on its
            own. Both <i>add</i> their results to the notes instead of replacing them.
          </p>
          {note && (
            <p className="ex-notice">
              Head {head}, standing on the last word “{note.query}”, puts {pct(note.sink)} of its attention on the
              start-of-text token (a resting place) and its next biggest share, {pct(note.bestW)}, on “{note.best}”.
            </p>
          )}
          <p className="ex-notice">
            The notes' average size after this block is <b>{now.toFixed(1)}</b>
            {prev !== null ? `, from ${prev.toFixed(1)} after block ${l - 1}` : ""}.
          </p>
          <Facts
            items={[
              ["order", "LN1 → attention → add → LN2 → MLP → add"],
              ["attention", `${meta.n_heads} heads × d_head ${meta.d_head}, causal`],
              ["MLP", `${meta.d_model} → ${meta.d_mlp} → ${meta.d_model}, GELU`],
              ["shown", `blocks.${l}.attn.hook_pattern`],
            ]}
          />
          <a className="ex-link" href={`#/playground?hook=blocks.${l}.hook_out&position=9`}>
            Patch this block in the Playground →
          </a>
        </>
      );
    }
    case "lnf":
      return (
        <>
          <h2>A final tidy-up</h2>
          <p>
            Before the model makes its guess, the notes are rescaled one last time so their overall size does
            not distort the result.
          </p>
          <p className="ex-notice">
            Average size of the notes after the last block: <b>{flow.blocks[flow.n_layers - 1].resid_norm_mean.toFixed(1)}</b>
          </p>
          <Facts items={[["step", "final LayerNorm (ln_f)"], ["width", `${meta.d_model} numbers per token`]]} />
        </>
      );
    case "logits": {
      const top = flow.top_predictions[0];
      return (
        <>
          <h2>The answer: a score for every word</h2>
          <p>
            The notes for the last word are compared with all {meta.d_vocab.toLocaleString()} tokens, giving each
            one a score. The scores turn into the probabilities shown here.
          </p>
          {top && (
            <p className="ex-notice">
              The model's top choice is <b>“{word(top.token)}”</b> at <b>{(top.prob * 100).toFixed(1)}%</b>.
            </p>
          )}
          <Facts
            items={[
              ["step", "unembed W_U, then softmax"],
              ["W_U", `${meta.d_model} × ${meta.d_vocab.toLocaleString()}`],
            ]}
          />
        </>
      );
    }
  }
}

function StageVisual({
  stage,
  flow,
  head,
  setHead,
}: {
  stage: Stage;
  flow: FlowData;
  head: number;
  setHead: (h: number) => void;
}) {
  if (stage.kind === "tokenize") {
    return (
      <div className="ex-tokens">
        <TokenRow tok={{ tokens: flow.tokens, token_ids: flow.token_ids, n_tokens: flow.tokens.length }} />
        <p className="ex-cap">Hover a token to see its position and number.</p>
      </div>
    );
  }
  if (stage.kind === "embed") {
    return (
      <div className="ex-tokens">
        <TokenRow tok={{ tokens: flow.tokens, token_ids: flow.token_ids, n_tokens: flow.tokens.length }} />
        <p className="ex-cap">
          Each token above becomes one row of {768} numbers. The values are learned weights, not something this
          page measures, so none are drawn.
        </p>
      </div>
    );
  }
  if (stage.kind === "block") {
    const layer = stage.layer;
    const pattern = {
      hook: `blocks.${layer}.attn.hook_pattern`,
      shape: [flow.blocks[layer].attention.length, flow.tokens.length, flow.tokens.length],
      tokens: flow.tokens,
      pattern: flow.blocks[layer].attention,
    };
    return (
      <>
        <div className="ex-cap-row">
          <span>Where each word looks (attention), head {head}</span>
          <span className="live-badge">MEASURED</span>
        </div>
        <AttentionHeatmap att={pattern} head={head} />
        <div className="ex-cap-row">
          <span>All {pattern.pattern.length} heads of this block. Pick one.</span>
        </div>
        <HeadThumbs pattern={flow.blocks[layer].attention} selected={head} onSelect={setHead} />
        <TokenNorms norms={flow.blocks[layer].resid_norm_per_token} tokens={flow.tokens} />
      </>
    );
  }
  if (stage.kind === "lnf") {
    return <TokenNorms norms={flow.blocks[flow.n_layers - 1].resid_norm_per_token} tokens={flow.tokens} />;
  }
  return <TopPredictions candidates={flow.top_predictions} metric="prob" maxRows={8} barColor="var(--measure)" />;
}

function HeadThumbs({
  pattern,
  selected,
  onSelect,
}: {
  pattern: number[][][];
  selected: number;
  onSelect: (h: number) => void;
}) {
  return (
    <div className="ex-thumbs" role="group" aria-label="Choose an attention head">
      {pattern.map((m, h) => {
        const n = m.length;
        return (
          <button
            key={h}
            className={`ex-thumb${h === selected ? " on" : ""}`}
            onClick={() => onSelect(h)}
            aria-label={`Head ${h}`}
            aria-pressed={h === selected}
          >
            <svg viewBox={`0 0 ${n} ${n}`} width={44} height={44} aria-hidden>
              {m.map((row, q) =>
                row.map((v, k) =>
                  k <= q ? (
                    <rect
                      key={`${q}-${k}`}
                      x={k}
                      y={q}
                      width={1}
                      height={1}
                      style={{
                        fill: `color-mix(in srgb, var(--heat-hi) ${Math.round(Math.min(1, v) * 100)}%, var(--heat-lo))`,
                      }}
                    />
                  ) : null,
                ),
              )}
            </svg>
            <span>H{h}</span>
          </button>
        );
      })}
    </div>
  );
}

/** Per-token size of the notes. The first token always holds a far larger signal, so it is drawn
 *  clipped with its real value and the other bars are scaled to the rest. */
function TokenNorms({ norms, tokens }: { norms: number[]; tokens: string[] }) {
  const rest = Math.max(...norms.slice(1), 1e-9);
  return (
    <div className="ex-norms">
      <div className="ex-cap-row">
        <span>Size of the notes for each token after this stage</span>
        <span className="live-badge">MEASURED</span>
      </div>
      <div className="ex-norm-bars">
        {norms.map((n, i) => {
          const clipped = i === 0 && n > rest * 1.5;
          return (
            <div key={i} className="ex-norm" title={`${word(tokens[i])}: ${n.toFixed(1)}`}>
              <span className={`ex-norm-bar${clipped ? " clipped" : ""}`} style={{ height: `${clipped ? 100 : (n / rest) * 100}%` }} />
              <span className="ex-norm-tok">{word(tokens[i])}</span>
            </div>
          );
        })}
      </div>
      {norms[0] > rest * 1.5 && (
        <p className="ex-cap">
          The first token holds a very large signal ({norms[0].toFixed(0)}), so its bar is cut off and the others are
          scaled to each other.
        </p>
      )}
    </div>
  );
}
