import { useEffect, useMemo, useRef, useState } from "react";
import type { FlowData } from "../api/client";
import AttentionHeatmap from "./AttentionHeatmap";
import TokenRow from "../components/TokenRow";
import TopPredictions from "./TopPredictions";

/** Computation-flow playback: a guided replay of recorded stages from ONE real
 *  forward pass (backend /api/flow). Every displayed number is measured. Stage
 *  durations are presentation pacing only — explicitly not GPU kernel timing. */

type Stage =
  | { kind: "tokenize" }
  | { kind: "embed" }
  | { kind: "block"; layer: number }
  | { kind: "lnf" }
  | { kind: "logits" };

const STAGE_MS = 1600;

export default function FlowPlayback({ flow }: { flow: FlowData | null }) {
  const [stageIdx, setStageIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [head, setHead] = useState(5);
  const timer = useRef<number | null>(null);

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

  useEffect(() => {
    setStageIdx(0);
    setPlaying(false);
  }, [flow]);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setStageIdx((i) => {
        if (i >= stages.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, STAGE_MS);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, stages.length]);

  if (!flow) return <div className="skeleton" style={{ height: "18rem" }} />;

  const stage = stages[stageIdx];
  const finished = stageIdx === stages.length - 1 && !playing;

  const step = (d: number) => {
    setPlaying(false);
    setStageIdx((i) => Math.max(0, Math.min(stages.length - 1, i + d)));
  };

  return (
    <div>
      <div className="playback-bar" role="group" aria-label="Playback controls">
        <button
          className="primary"
          onClick={() => {
            if (finished) setStageIdx(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? "❚❚ Pause" : finished ? "↻ Replay" : "▶ Play"}
        </button>
        <button onClick={() => step(-1)} disabled={stageIdx === 0} aria-label="Step backward">
          ◀ Step
        </button>
        <button onClick={() => step(1)} disabled={stageIdx === stages.length - 1} aria-label="Step forward">
          Step ▶
        </button>
        <span className="mono" style={{ fontSize: "0.85rem", color: "var(--text-2)" }}>
          stage {stageIdx + 1}/{stages.length}
        </span>
        <span className="playback-timing-note">
          schematic replay of recorded stages — durations are presentation pacing, not measured GPU time
        </span>
      </div>
      <div className="playback-stages" role="tablist" aria-label="Computation stages">
        {stages.map((s, i) => (
          <button
            key={i}
            role="tab"
            aria-selected={i === stageIdx}
            aria-label={s.kind === "block" ? `Block ${s.layer}` : s.kind}
            className={`playback-stage ${i === stageIdx ? "active" : i < stageIdx ? "done" : ""}`}
            onClick={() => {
              setPlaying(false);
              setStageIdx(i);
            }}
          />
        ))}
      </div>

      <div className="panel stage-panel" style={{ marginTop: "0.8rem" }}>
        {stage.kind === "tokenize" && (
          <>
            <h3>Stage 1 · Tokenization</h3>
            <p style={{ color: "var(--text-1)" }}>
              Byte-level BPE splits the text into {flow.tokens.length} tokens. The model sees these exact
              pieces — notice the space inside word tokens.
            </p>
            <TokenRow tok={{ tokens: flow.tokens, token_ids: flow.token_ids, n_tokens: flow.tokens.length }} />
          </>
        )}
        {stage.kind === "embed" && (
          <>
            <h3>Stage 2 · Embeddings</h3>
            <p style={{ color: "var(--text-1)" }}>
              Each token ID is looked up in a learned 768-dim table, and a position vector is added so the
              model knows word order. Their sum starts the residual stream.
            </p>
            <div className="stage-kv">
              <div className="kv"><div className="name">tokens in</div><div className="val">{flow.tokens.length}</div></div>
              <div className="kv"><div className="name">stream width</div><div className="val">768</div></div>
              <div className="kv"><div className="name">blocks ahead</div><div className="val">{flow.n_layers}</div></div>
            </div>
          </>
        )}
        {stage.kind === "block" && (
          <>
            <h3>
              Stage · Transformer block {stage.layer}
              <span style={{ color: "var(--text-2)", fontSize: "0.85rem", marginLeft: "0.6rem" }}>
                measured attention, head {head}
              </span>
            </h3>
            <AttentionHeatmap att={flowToPattern(flow, stage.layer)} head={head} />
            <ResidNormChart norms={flow.blocks[stage.layer].resid_norm_per_token} tokens={flow.tokens} />
            <HeadPicker nHeads={12} selected={head} onSelect={setHead} />
          </>
        )}
        {stage.kind === "lnf" && (
          <>
            <h3>Stage · Final layer normalization</h3>
            <p style={{ color: "var(--text-1)" }}>
              The residual stream is normalized one last time so its scale doesn't distort the output
              projection. Measured stream magnitude after the last block:
            </p>
            <div className="stage-kv">
              <div className="kv">
                <div className="name">residual norm (mean)</div>
                <div className="val">{flow.blocks[flow.n_layers - 1].resid_norm_mean}</div>
              </div>
            </div>
          </>
        )}
        {stage.kind === "logits" && (
          <>
            <h3>Stage · Output scores</h3>
            <p style={{ color: "var(--text-1)" }}>
              A linear projection turns the final vector into {50257} logits — one score per vocabulary
              token. Softmax turns them into the probabilities below.
            </p>
            <TopPredictions candidates={flow.top_predictions} metric="prob" />
          </>
        )}
        {finished && (
          <div className="state-banner ok" style={{ marginTop: "0.8rem" }}>
            Completed run. All {flow.n_layers} block stages replayed from recorded data — the run itself
            finished on the GPU/CPU in one parallel pass.
          </div>
        )}
      </div>
    </div>
  );
}

function flowToPattern(flow: FlowData, layer: number) {
  return {
    hook: `blocks.${layer}.attn.hook_pattern`,
    shape: [12, flow.tokens.length, flow.tokens.length],
    tokens: flow.tokens,
    pattern: flow.blocks[layer].attention,
  };
}

function HeadPicker({
  nHeads,
  selected,
  onSelect,
}: {
  nHeads: number;
  selected: number;
  onSelect: (h: number) => void;
}) {
  return (
    <div style={{ display: "flex", gap: "0.3rem", flexWrap: "wrap", marginTop: "0.6rem" }}>
      {Array.from({ length: nHeads }, (_, h) => (
        <button
          key={h}
          style={{ padding: "0.2rem 0.55rem", fontSize: "0.78rem" }}
          className={h === selected ? "primary" : ""}
          onClick={() => onSelect(h)}
        >
          H{h}
        </button>
      ))}
    </div>
  );
}

/** Measured per-token residual-stream norm after this block (L2 over 768 dims). */
function ResidNormChart({ norms, tokens }: { norms: number[]; tokens: string[] }) {
  const max = Math.max(...norms, 1e-9);
  return (
    <div style={{ marginTop: "0.6rem" }}>
      <div style={{ fontSize: "0.8rem", color: "var(--text-2)" }}>
        residual-stream L2 norm per token after this block (measured)
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 54, marginTop: 4 }}>
        {norms.map((n, i) => (
          <div
            key={i}
            title={`${tokens[i]}: ${n}`}
            style={{
              flex: 1,
              height: `${(n / max) * 100}%`,
              minHeight: 3,
              background: "var(--violet)",
              opacity: 0.85,
              borderRadius: 2,
            }}
          />
        ))}
      </div>
    </div>
  );
}
