import { useEffect, useMemo, useState } from "react";
import {
  api,
  type AttentionPattern,
  type FlowData,
  type Health,
  type ModelMeta,
  type Tokenization,
} from "../api/client";
import ArchitectureCanvas, { type CanvasLevel } from "../components/ArchitectureCanvas";
import AttentionHeatmap from "../components/AttentionHeatmap";
import TokenRow from "../components/TokenRow";
import GuidedTour from "./explorer/GuidedTour";

type Level = CanvasLevel | "head";
type Mode = "guided" | "map";

const EXAMPLE = "The doctor asked the nurse to help her because she was tired.";

export default function Explorer({ health }: { health: Health | null }) {
  const params = useMemo(() => new URLSearchParams(window.location.hash.split("?")[1] ?? ""), []);
  // the guided walk is the default (it replaces the old "flow" replay); the full map opens for
  // ?mode=map and for Playground hand-offs that carry a hook to mark on the canvas
  const [mode, setMode] = useState<Mode>(() => {
    const h = window.location.hash;
    return h.includes("mode=map") || h.includes("hook=") ? "map" : "guided";
  });
  const [level, setLevel] = useState<Level>("model");
  const urlParams = useMemo(() => new URLSearchParams(window.location.hash.split("?")[1] ?? ""), []);
  const initialHook = useMemo(() => urlParams.get("hook"), [urlParams]);
  const initialText = useMemo(() => urlParams.get("text") ?? EXAMPLE, [urlParams]);
  const [layer, setLayer] = useState<number>(() => {
    const m = initialHook?.match(/blocks\.(\d+)\./);
    return m ? Math.min(11, Math.max(0, Number(m[1]))) : 5;
  });
  const [head, setHead] = useState<number>(0);
  const [compareHead, setCompareHead] = useState<number | null>(null);
  const [text, setText] = useState(initialText);
  const [runText, setRunText] = useState(initialText);
  const [tokenization, setTokenization] = useState<Tokenization | null>(null);
  const [attention, setAttention] = useState<AttentionPattern | null>(null);
  const [flow, setFlow] = useState<FlowData | null>(null);
  const [intervention, setIntervention] = useState<{ hook: string; position: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const meta: ModelMeta | null = health?.model_loaded ? health.model : null;
  const ready = meta !== null;
  // the flow capture holds every layer's attention for the same run - derive
  // the displayed pattern from it so the heatmap can never go stale when the
  // user navigates to a different block
  const attentionForLayer = useMemo(() => {
    if (!flow) return attention;
    return {
      hook: `blocks.${layer}.attn.hook_pattern`,
      shape: [12, flow.tokens.length, flow.tokens.length],
      tokens: flow.tokens,
      pattern: flow.blocks[layer].attention,
    };
  }, [flow, layer, attention]);

  const runCapture = (t: string, l: number) => {
    if (!ready) return;
    setBusy(true);
    setError(null);
    Promise.all([api.inspect(t, l), api.flow(t)])
      .then(([insp, fl]) => {
        if (insp.tokenization) setTokenization(insp.tokenization);
        if (insp.attention) setAttention(insp.attention);
        if (insp.attention_error) setError(insp.attention_error);
        setFlow(fl);
        const hk = params.get("hook");
        const pos = params.get("position");
        if (hk && pos != null && !Number.isNaN(Number(pos))) {
          setIntervention({ hook: hk, position: Number(pos) });
        }
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    if (ready && !flow && !busy) runCapture(runText, layer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // inspect links from the Playground carry a ?text= prefix; re-capture on
  // hash changes even when this page is already mounted
  useEffect(() => {
    const onHash = () => {
      const t = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("text");
      if (t && t !== runText && t !== text) {
        setText(t);
        setRunText(t);
        setFlow(null);
        setAttention(null);
        runCapture(t, layer);
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runText, text, layer]);

  const crumbs = useMemo(() => {
    const items: { label: string; go: () => void }[] = [
      { label: "Whole model", go: () => setLevel("model") },
    ];
    if (level !== "model") items.push({ label: `Block ${layer}`, go: () => setLevel("block") });
    if (level === "attn" || level === "head")
      items.push({ label: "Attention", go: () => setLevel("attn") });
    if (level === "head") items.push({ label: `Head ${head}`, go: () => setLevel("head") });
    return items;
  }, [level, layer, head]);

  if (!ready) {
    return (
      <main className="page ex-page">
        <h1 className="ex-title">Model Explorer</h1>
        <div className="state-banner info">
          <span className="spinner" aria-hidden /> The model is loading. The explorer reads real layer/head
          counts and captured activations, so it needs the model ready (~1 minute after backend start).
        </div>
      </main>
    );
  }

  const run = (t: string) => {
    setRunText(t);
    setFlow(null);
    runCapture(t, layer);
  };

  return (
    <main className="page ex-page">
      <header className="ex-head">
        <div>
          <p className="ex-kick">Follow one sentence through GPT-2</p>
          <h1 className="ex-title">Model Explorer</h1>
          <p className="ex-sub">
            Type a sentence, then walk through every stage the model runs on it. Everything shown is measured
            on your own computer.
          </p>
        </div>
        <div className="ex-head-side">
          <div className="mode-tabs" role="tablist" aria-label="Explorer view">
            <button role="tab" aria-selected={mode === "guided"} className={mode === "guided" ? "active" : ""} onClick={() => setMode("guided")}>
              Guided tour
            </button>
            <button role="tab" aria-selected={mode === "map"} className={mode === "map" ? "active" : ""} onClick={() => setMode("map")}>
              Full map
            </button>
          </div>
          <div className="inspector-meta">
            {meta.model_id} · {meta.n_layers} blocks · {meta.n_heads} heads · d_model {meta.d_model} · {meta.device}
          </div>
        </div>
      </header>

      <form
        className="ex-input"
        onSubmit={(e) => {
          e.preventDefault();
          run(text);
        }}
      >
        <label htmlFor="explorer-text">Sentence</label>
        <input id="explorer-text" type="text" value={text} onChange={(e) => setText(e.target.value)} />
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Capturing…" : "Run & capture"}
        </button>
        <button
          type="button"
          onClick={() => {
            setText(EXAMPLE);
            setRunText(EXAMPLE);
            runCapture(EXAMPLE, layer);
          }}
        >
          Example
        </button>
      </form>
      {error && <div className="state-banner error" style={{ marginTop: "0.6rem" }}>{error}</div>}

      {mode === "guided" ? (
        <GuidedTour flow={flow} meta={meta} />
      ) : (
        <div className="ex-map">
          <div className="ex-tokenbar">
            <span>Tokens received</span>
            <TokenRow tok={tokenization} />
          </div>
          {intervention && (
            <div className="state-banner warn" style={{ marginTop: "1rem" }} role="status">
              <span>
                <strong>Intervention overlay:</strong> {intervention.hook} at token position {intervention.position} is
                marked on the canvas (dashed red outline).{" "}
                <a href={`#/playground?hook=${encodeURIComponent(intervention.hook)}&position=${intervention.position}`}>
                  Open it in the Playground →
                </a>
              </span>
            </div>
          )}
          <p className="breadcrumbs" aria-label="Breadcrumb" style={{ marginTop: "0.8rem" }}>
            {crumbs.map((b, i) => (
              <span key={i} style={{ display: "contents" }}>
                {i > 0 && <span aria-hidden> › </span>}
                <button onClick={b.go}>{b.label}</button>
              </span>
            ))}
          </p>
          <div className="explorer-grid" style={{ alignItems: "start", marginTop: "0.8rem" }}>
            <div className="panel" style={{ padding: "0.6rem" }}>
              <ArchitectureCanvas
                meta={meta}
                flow={flow}
                level={level === "head" ? "attn" : (level as CanvasLevel)}
                layer={layer}
                intervention={intervention}
                onLevel={(l) => setLevel(l)}
                onLayer={(l) => setLayer(l)}
                onHead={(h) => {
                  setHead(h);
                  setLevel("head");
                }}
              />
            </div>
            <div className="panel inspector-sticky">
              <InspectorConsole
                meta={meta}
                flow={flow}
                level={level}
                layer={layer}
                head={head}
                attention={attentionForLayer}
                onLayer={(l) => { setLayer(l); setLevel("block"); }}
                onHead={(h) => { setHead(h); setLevel("head"); }}
                runText={runText}
                compareHead={compareHead}
                onCompareHead={setCompareHead}
              />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function InspectorConsole({
  meta,
  flow,
  level,
  layer,
  head,
  attention,
  onLayer,
  onHead,
  runText,
  compareHead,
  onCompareHead,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  level: Level;
  layer: number;
  head: number;
  attention: AttentionPattern | null;
  onLayer: (l: number) => void;
  onHead: (h: number) => void;
  runText: string;
  compareHead: number | null;
  onCompareHead: (h: number | null) => void;
}) {
  if (level === "model") {
    return (
      <>
        <h3>Measured residual stream</h3>
        <p className="inspector-meta">
          L2 norm of the residual stream after each block, averaged over tokens - from the real captured
          run. Click a bar to open that block.
        </p>
        <ResidNormAcrossLayers flow={flow} onSelectLayer={onLayer} />
        <h3 style={{ marginTop: "1rem" }}>How to explore</h3>
        <p className="inspector-meta">
          Click a block on the ribbon to open it, then choose "Look inside". There you can open the attention
          anatomy and pick one of the 12 heads to see its real token-level pattern. The arrow keys move between
          blocks.
        </p>
        <p className="schematic-note">
          <span className="static-badge">STRUCTURE + MEASURED</span> stage order from live metadata; ribbon
          width, blue bars and token strip from the real run.
        </p>
      </>
    );
  }
  if (level === "block") {
    return (
      <>
        <h3>Block {layer} · verified pre-LN order</h3>
        <table className="kv-table">
          <tbody>
            <tr><td>1 · input</td><td>residual stream, {meta.d_model} dims</td></tr>
            <tr><td>2 · LN1</td><td>normalize → feeds attention only</td></tr>
            <tr><td>3 · attention</td><td>{meta.n_heads} heads × d_head {meta.d_head}, causal</td></tr>
            <tr><td>4 · add</td><td>attention output added to stream</td></tr>
            <tr><td>5 · LN2</td><td>normalize → feeds MLP only</td></tr>
            <tr><td>6 · MLP</td><td>{meta.d_model} → {meta.d_mlp} → {meta.d_model}, GELU</td></tr>
            <tr><td>7 · add</td><td>MLP output added → next block</td></tr>
          </tbody>
        </table>
        {flow && (
          <>
            <h3 style={{ marginTop: "1rem" }}>Residual norm per token (after this block)</h3>
            <p className="inspector-meta mono">{flow.blocks[layer].resid_norm_per_token.join(" · ")}</p>
          </>
        )}
        <a
          href={`#/playground?hook=blocks.${layer}.hook_out&position=9`}
          style={{ display: "inline-block", marginTop: "1rem" }}
        >
          <button className="primary">Patch this block in the Playground →</button>
        </a>
      </>
    );
  }
  if (level === "attn") {
    return (
      <>
        <h3>Attention anatomy · block {layer}</h3>
        <p className="inspector-meta">
          Per head: A = softmax(QKᵀ/√{meta.d_head} + mask), out = A·V. Heads are parallel lanes; the
          combined output is concat({meta.n_heads}×{meta.d_head}) → W_O.
        </p>
        <h3 style={{ marginTop: "1rem" }}>All heads of this block (measured)</h3>
        <HeadGallery flow={flow} layer={layer} onSelect={onHead} />
        <h3 style={{ marginTop: "1rem" }}>Select a head</h3>
        <HeadGrid nHeads={meta.n_heads} selected={head} onSelect={onHead} />
        <p className="inspector-meta" style={{ marginTop: "0.6rem" }}>
          An attention pattern is a descriptive measurement - where each query token distributes its
          attention. It is not proof of what the head causes; that requires the Playground interventions.
        </p>
      </>
    );
  }
  // head level
  return (
    <>
      <h3>Head {head} of block {layer}</h3>
      <AttentionHeatmap att={attention} head={head} compareHead={compareHead} onCompareHead={onCompareHead} />
      <h3 style={{ marginTop: "0.8rem" }}>All heads of this block (measured)</h3>
      <HeadGallery flow={flow} layer={layer} onSelect={onHead} />
      <div style={{ marginTop: "0.8rem" }}>
        <HeadGrid nHeads={meta.n_heads} selected={head} onSelect={onHead} />
      </div>
      <dl className="inspector-meta" style={{ marginTop: "0.8rem" }}>
        <dt>Hook identifier</dt>
        <dd>blocks.{layer}.attn.hook_pattern</dd>
        <dt>Tensor shape</dt>
        <dd>{attention ? attention.shape.join(" × ") : "-"} (heads × query × key)</dd>
        <dt>Displayed slice</dt>
        <dd>head {head}, all positions (unreduced)</dd>
        <dt>Value scale</dt>
        <dd>softmax weights ∈ [0, 1], rows sum to 1.0</dd>
        <dt>Source run</dt>
        <dd>{runText.slice(0, 44)}{runText.length > 44 ? "…" : ""}</dd>
        <dt>Provenance</dt>
        <dd>{meta.model_id} · {meta.device} · {meta.dtype}</dd>
      </dl>
    </>
  );
}

function ResidNormAcrossLayers({
  flow,
  onSelectLayer,
}: {
  flow: FlowData | null;
  onSelectLayer: (l: number) => void;
}) {
  if (!flow) return <div className="skeleton" style={{ height: "8rem" }} />;
  const means = flow.blocks.map((b) => b.resid_norm_mean);
  const max = Math.max(...means, 1e-9);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 120, marginTop: "0.8rem" }}>
      {means.map((m, l) => (
        <div
          key={l}
          role="button"
          tabIndex={0}
          aria-label={`Block ${l}: mean residual norm ${m}`}
          onClick={() => onSelectLayer(l)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSelectLayer(l);
            }
          }}
          title={`Block ${l}: ${m}`}
          style={{
            flex: 1,
            height: `${(m / max) * 100}%`,
            minHeight: 4,
            background: "var(--measure)",
            borderRadius: 3,
            cursor: "pointer",
            position: "relative",
          }}
        >
          <span style={{ position: "absolute", bottom: -18, left: 0, right: 0, textAlign: "center", fontSize: "0.62rem", color: "var(--text-2)" }}>
            {l}
          </span>
        </div>
      ))}
    </div>
  );
}

function HeadGrid({
  nHeads,
  selected,
  onSelect,
}: {
  nHeads: number;
  selected: number;
  onSelect: (h: number) => void;
}) {
  const size = 30;
  const cols = 6;
  return (
    <svg
      className="arch-svg"
      width={cols * (size + 6)}
      height={Math.ceil(nHeads / cols) * (size + 6)}
      role="group"
      aria-label={`Select one of ${nHeads} attention heads`}
    >
      {Array.from({ length: nHeads }, (_, h) => (
        <g
          key={h}
          className={`head-cell${h === selected ? " selected" : ""}`}
          onClick={() => onSelect(h)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSelect(h);
            }
          }}
          tabIndex={0}
          role="button"
          aria-label={`Head ${h}`}
        >
          <rect x={(h % cols) * (size + 6)} y={Math.floor(h / cols) * (size + 6)} width={size} height={size} fill="var(--bg-3)" stroke="var(--line-1)" />
          <text x={(h % cols) * (size + 6) + size / 2} y={Math.floor(h / cols) * (size + 6) + size / 2 + 4} fill={h === selected ? "var(--accent)" : "var(--text-1)"} fontSize={11} textAnchor="middle">
            {h}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Small multiples: the real attention pattern of every head in one block,
 *  from the captured flow run. Click to open a head at full size. */
function HeadGallery({
  flow,
  layer,
  onSelect,
}: {
  flow: FlowData | null;
  layer: number;
  onSelect: (h: number) => void;
}) {
  if (!flow) return <div className="skeleton" style={{ height: "10rem" }} />;
  const cell = 17;
  const pattern = flow.blocks[layer].attention;
  return (
    <svg
      width={12 * (cell + 8)}
      height={Math.ceil(pattern.length) * (cell + 10)}
      role="group"
      aria-label={`Measured attention patterns of all ${pattern.length} heads of block ${layer}`}
    >
      {pattern.map((head, h) => {
        const ox = (h % 6) * (cell + 8);
        const oy = Math.floor(h / 6) * (cell + 10) + 4;
        const px = (v: number) => {
          const t = Math.max(0, Math.min(1, v));
          return `rgb(${Math.round(241 - t * 45)},${Math.round(234 - t * 170)},${Math.round(219 - t * 172)})`;
        };
        const n = head.length;
        const cs = cell / n;
        return (
          <g
            key={h}
            className="head-cell"
            onClick={() => onSelect(h)}
            tabIndex={0}
            role="button"
            aria-label={`Open head ${h}`}
            onKeyDown={(e) => {
              if (e.key === "Enter") onSelect(h);
            }}
          >
            {head.map((row, q) =>
              row.map((v, k) => (
                <rect key={`${q}-${k}`} x={ox + k * cs} y={oy + q * cs} width={cs} height={cs} fill={px(v)} />
              )),
            )}
            <text x={ox + cell / 2} y={oy + cell + 8} fill="var(--text-2)" fontSize={7.5} textAnchor="middle">
              H{h}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
