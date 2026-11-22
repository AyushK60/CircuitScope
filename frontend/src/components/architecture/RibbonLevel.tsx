import { useEffect, useRef, type ReactNode } from "react";
import type { FlowData, ModelMeta } from "../../api/client";

/** The whole-model view: the sentence enters at the top and one red ribbon runs down
 *  the page, the residual stream. Every block reads it and adds its result back.
 *  When a real run is loaded, the ribbon's width after each block is the measured
 *  mean residual norm; without a run it stays a plain, uniform band. */

const CELL = 104; // rail column width in px
const MID = CELL / 2;
const W_MIN = 12;
const W_MAX = 46;
const W_PLAIN = 20;

interface Props {
  meta: ModelMeta;
  flow: FlowData | null;
  layer: number;
  intervention?: { hook: string; position: number } | null;
  onSelectLayer: (l: number) => void;
  onOpenBlock: (l: number) => void;
  onHead: (layer: number, head: number) => void;
}

function Row({
  wTop,
  wBot,
  num,
  on,
  children,
  rail,
}: {
  wTop: number;
  wBot: number;
  num?: string;
  on?: boolean;
  children: ReactNode;
  rail?: ReactNode;
}) {
  const d = `M${MID - wTop / 2} 0 L${MID + wTop / 2} 0 L${MID + wBot / 2} 100 L${MID - wBot / 2} 100 Z`;
  return (
    <div className="rb-row">
      <div className={`rb-num${on ? " on" : ""}`} aria-hidden>
        {num}
      </div>
      <div className="rb-rail">
        <svg viewBox={`0 0 ${CELL} 100`} preserveAspectRatio="none" aria-hidden focusable="false">
          <path d={d} />
        </svg>
        {rail}
      </div>
      <div className="rb-body">{children}</div>
    </div>
  );
}

export default function RibbonLevel({ meta, flow, layer, intervention, onSelectLayer, onOpenBlock, onHead }: Props) {
  const n = meta.n_layers;
  const norms = flow ? flow.blocks.map((b) => b.resid_norm_mean) : null;
  const max = norms ? Math.max(...norms, 1e-9) : 1;
  // square-root scale: the last blocks are far larger than the first, and a linear scale hides the early ones
  const width = (l: number) => (norms ? W_MIN + (W_MAX - W_MIN) * Math.sqrt(norms[l] / max) : W_PLAIN);
  const w = Array.from({ length: n }, (_, l) => width(l));
  const wTop = (l: number) => (l === 0 ? W_MIN : (w[l - 1] + w[l]) / 2);
  const wBot = (l: number) => (l === n - 1 ? w[l] : (w[l] + w[l + 1]) / 2);
  const targetLayer = intervention ? Number((intervention.hook.match(/blocks\.(\d+)\./) ?? [])[1] ?? -1) : -1;

  // scroll the chosen block into view when it changes from elsewhere (inspector bars, arrow keys)
  const rootRef = useRef<HTMLDivElement | null>(null);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    rootRef.current?.querySelector(".rb-card.on")?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [layer]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    if ((e.target as HTMLElement).closest("input, select, textarea")) return;
    e.preventDefault();
    onSelectLayer(Math.min(n - 1, Math.max(0, layer + (e.key === "ArrowDown" ? 1 : -1))));
  };

  const top = flow?.top_predictions.slice(0, 5) ?? [];

  return (
    <div className="rb" ref={rootRef} onKeyDown={onKeyDown}>
      <Row wTop={0} wBot={W_MIN} rail={<span className="rb-rail-label">residual stream</span>}>
        <div className="rb-plain">
          <h3>Your sentence</h3>
          {flow ? (
            <ul className="rb-tokens" aria-label="Tokens received by the model">
              {flow.tokens.map((t, i) => (
                <li key={i} className={intervention?.position === i ? "mark" : ""}>
                  {t.trim() || "␣"}
                </li>
              ))}
            </ul>
          ) : (
            <p>Run the model to see your sentence split into tokens.</p>
          )}
        </div>
      </Row>
      <Row wTop={W_MIN} wBot={W_MIN}>
        <div className="rb-plain">
          <h3>Tokens become numbers</h3>
          <p>
            {meta.tokenizer_class}, vocabulary of {meta.d_vocab.toLocaleString()}. Each token gets a vector of {meta.d_model} numbers,
            plus its position.
          </p>
        </div>
      </Row>

      {Array.from({ length: n }, (_, l) => {
        const on = layer === l;
        const norm = norms?.[l];
        const isTarget = l === targetLayer;
        return (
          <Row
            key={l}
            wTop={wTop(l)}
            wBot={wBot(l)}
            num={String(l).padStart(2, "0")}
            on={on}
            rail={
              <>
                <span className="rb-arrow read" style={{ left: `${((MID + w[l] / 2) / CELL) * 100}%` }} />
                <span className="rb-arrow write" style={{ left: `${((MID + w[l] / 2) / CELL) * 100}%` }} />
              </>
            }
          >
            <div className={`rb-card${on ? " on" : ""}${isTarget ? " target" : ""}`}>
              <button
                type="button"
                className="rb-card-head"
                aria-label={`Open transformer block ${l}`}
                aria-expanded={on}
                onClick={() => onSelectLayer(l)}
              >
                <span className="rb-title">Block {l}</span>
                <span className="rb-chain" aria-hidden>
                  <i>LN1</i>
                  <i className="hot">attention</i>
                  <i>LN2</i>
                  <i className="hot">MLP</i>
                </span>
                <span className="rb-meter">
                  {norm != null ? (
                    <>
                      <span className="rb-bar">
                        <span style={{ width: `${(norm / max) * 100}%` }} />
                      </span>
                      <b>{norm}</b>
                    </>
                  ) : (
                    <em>not measured yet</em>
                  )}
                </span>
              </button>
              {isTarget && intervention && (
                <p className="rb-target">
                  INTERVENTION TARGET · {intervention.hook} @ pos {intervention.position}
                </p>
              )}
              {on && (
                <div className="rb-open">
                  <p>
                    Reads the ribbon, adds what attention finds, then adds what the MLP finds. The ribbon keeps everything it
                    already had.
                  </p>
                  <p className="rb-sub">{meta.n_heads} attention heads in this block. Pick one to see where it looks.</p>
                  <div className="rb-heads">
                    {Array.from({ length: meta.n_heads }, (_, h) => (
                      <button key={h} type="button" aria-label={`Head ${h} of block ${l}`} onClick={() => onHead(l, h)}>
                        {h}
                      </button>
                    ))}
                  </div>
                  <button type="button" className="primary rb-go" aria-label={`Look inside block ${l}`} onClick={() => onOpenBlock(l)}>
                    Look inside block {l} →
                  </button>
                </div>
              )}
            </div>
          </Row>
        );
      })}

      <Row wTop={w[n - 1]} wBot={W_MIN}>
        <div className="rb-plain">
          <h3>Final normalization and output</h3>
          <p>
            One last LayerNorm, then W_U turns {meta.d_model} numbers into {meta.d_vocab.toLocaleString()} scores, one per word the
            model knows.
          </p>
        </div>
      </Row>
      <Row wTop={W_MIN} wBot={0}>
        <div className="rb-plain rb-answer">
          <h3>Next-word scores</h3>
          {top.length > 0 ? (
            <ol>
              {top.map((c, i) => (
                <li key={c.token_id} className={i === 0 ? "top" : ""}>
                  <span className="w">{c.token.trim() || (c.token.includes("\n") ? "new line" : "space")}</span>
                  <span className="bar">
                    <span style={{ width: `${(c.prob / top[0].prob) * 100}%` }} />
                  </span>
                  <span className="p">{(c.prob * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ol>
          ) : (
            <p>Run the model to see its top guesses for the next word.</p>
          )}
        </div>
      </Row>

      <p className="rb-legend">
        <span className="k red" /> residual stream{norms ? ", thickness grows with the measured size (square-root scale)" : ""}
        <span className="k blue" /> measured from the real run
        <span className="k hint">↑ ↓ move between blocks</span>
      </p>
    </div>
  );
}
