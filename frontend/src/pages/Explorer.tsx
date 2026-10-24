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
import FlowPlayback from "../components/FlowPlayback";
import TokenRow from "../components/TokenRow";

type Level = CanvasLevel | "head";
type Mode = "anatomy" | "flow";

const EXAMPLE = "The doctor asked the nurse to help her because she was tired.";

export default function Explorer({ health }: { health: Health | null }) {
  const [mode, setMode] = useState<Mode>("anatomy");
  const [level, setLevel] = useState<Level>("model");
  const [layer, setLayer] = useState<number>(5);
  const [head, setHead] = useState<number>(0);
  const [text, setText] = useState(EXAMPLE);
  const [runText, setRunText] = useState(EXAMPLE);
  const [tokenization, setTokenization] = useState<Tokenization | null>(null);
  const [attention, setAttention] = useState<AttentionPattern | null>(null);
  const [flow, setFlow] = useState<FlowData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const meta: ModelMeta | null = health?.model_loaded ? health.model : null;
  const ready = meta !== null;

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
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    if (ready && !flow && !busy) runCapture(runText, layer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

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
      <main className="page" style={{ maxWidth: "1400px" }}>
        <h1>Model Explorer</h1>
        <div className="state-banner info">
          <span className="spinner" aria-hidden /> The model is loading. The explorer reads real layer/head
          counts and captured activations, so it needs the model ready (~1 minute after backend start).
        </div>
      </main>
    );
  }

  return (
    <main className="page" style={{ maxWidth: "1400px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.8rem" }}>
        <div>
          <h1>Model Explorer</h1>
          <p className="breadcrumbs" aria-label="Breadcrumb">
            {crumbs.map((b, i) => (
              <span key={i} style={{ display: "contents" }}>
                {i > 0 && <span aria-hidden> › </span>}
                <button onClick={b.go}>{b.label}</button>
              </span>
            ))}
          </p>
        </div>
        <div className="mode-tabs" role="tablist" aria-label="Explorer mode">
          <button role="tab" aria-selected={mode === "anatomy"} className={mode === "anatomy" ? "active" : ""} onClick={() => setMode("anatomy")}>
            Anatomy
          </button>
          <button role="tab" aria-selected={mode === "flow"} className={mode === "flow" ? "active" : ""} onClick={() => setMode("flow")}>
            Computation Flow
          </button>
        </div>
        <div className="inspector-meta" style={{ textAlign: "right" }}>
          {meta.model_id} · {meta.n_layers} blocks · {meta.n_heads} heads · d_model {meta.d_model} · {meta.device}
        </div>
      </div>

      <div className="panel" style={{ padding: "0.9rem 1.2rem" }}>
        <div style={{ display: "flex", gap: "0.8rem", alignItems: "end", flexWrap: "wrap" }}>
          <div style={{ flex: 2, minWidth: "260px" }}>
            <label htmlFor="explorer-text" style={{ fontSize: "0.85rem", color: "var(--text-2)" }}>
              Input for measured views
            </label>
            <textarea id="explorer-text" rows={2} value={text} onChange={(e) => setText(e.target.value)} />
          </div>
          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
            <button
              className="primary"
              onClick={() => {
                setRunText(text);
                setFlow(null);
                runCapture(text, layer);
              }}
              disabled={busy}
            >
              {busy ? "Capturing…" : "Run & capture"}
            </button>
            <button onClick={() => { setText(EXAMPLE); setRunText(EXAMPLE); runCapture(EXAMPLE, layer); }}>
              Example
            </button>
          </div>
          <div style={{ flex: 2, minWidth: "240px" }}>
            <span style={{ fontSize: "0.85rem", color: "var(--text-2)", display: "block" }}>Tokens received</span>
            <TokenRow tok={tokenization} />
          </div>
        </div>
        {error && <div className="state-banner error" style={{ marginTop: "0.6rem" }}>{error}</div>}
      </div>

      {mode === "flow" ? (
        <div style={{ marginTop: "1rem" }}>
          <p style={{ color: "var(--text-1)", maxWidth: "80ch" }}>
            A guided replay of one real forward pass: tokenization, embeddings, every transformer block,
            final normalization, and output scores. Every heatmap, norm, and score below was captured from
            the actual run on your text.
          </p>
          <FlowPlayback flow={flow} />
        </div>
      ) : (
        <div className="explorer-grid" style={{ alignItems: "start", marginTop: "1rem" }}>
          <div className="panel" style={{ padding: "0.6rem" }}>
            <ArchitectureCanvas
              meta={meta}
              flow={flow}
              level={level === "head" ? "attn" : (level as CanvasLevel)}
              layer={layer}
              keyboardFocusLayer={layer}
              onLevel={(l) => setLevel(l)}
              onLayer={(l) => setLayer(l)}
              onHead={(h) => {
                setHead(h);
                setLevel("head");
              }}
            />
          </div>
          <div className="panel">
            <InspectorConsole
              meta={meta}
              flow={flow}
              level={level}
              layer={layer}
              head={head}
              attention={attention}
              onLayer={(l) => { setLayer(l); setLevel("block"); }}
              onHead={(h) => { setHead(h); setLevel("head"); }}
              runText={runText}
            />
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
}) {
  if (level === "model") {
    return (
      <>
        <h3>Measured residual stream</h3>
        <p className="inspector-meta">
          L2 norm of the residual stream after each block, averaged over tokens — from the real captured
          run. Click a bar to open that block.
        </p>
        <ResidNormAcrossLayers flow={flow} onSelectLayer={onLayer} />
        <h3 style={{ marginTop: "1rem" }}>How to explore</h3>
        <p className="inspector-meta">
          Click any block on the canvas to zoom into it. Inside, open the attention anatomy and pick one of
          the 12 heads to see its real token-level pattern. Wheel zooms, drag pans, ↑↓ move the focus, Enter
          opens.
        </p>
        <p className="schematic-note">
          <span className="static-badge">STRUCTURE + MEASURED</span> stage order from live metadata; brass
          bars and token strip from the real run.
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
        <h3 style={{ marginTop: "1rem" }}>Select a head</h3>
        <HeadGrid nHeads={meta.n_heads} selected={head} onSelect={onHead} />
        <p className="inspector-meta" style={{ marginTop: "0.6rem" }}>
          An attention pattern is a descriptive measurement — where each query token distributes its
          attention. It is not proof of what the head causes; that requires the Playground interventions.
        </p>
      </>
    );
  }
  // head level
  return (
    <>
      <h3>Head {head} of block {layer}</h3>
      <AttentionHeatmap att={attention} head={head} />
      <div style={{ marginTop: "0.8rem" }}>
        <HeadGrid nHeads={meta.n_heads} selected={head} onSelect={onHead} />
      </div>
      <dl className="inspector-meta" style={{ marginTop: "0.8rem" }}>
        <dt>Hook identifier</dt>
        <dd>blocks.{layer}.attn.hook_pattern</dd>
        <dt>Tensor shape</dt>
        <dd>{attention ? attention.shape.join(" × ") : "—"} (heads × query × key)</dd>
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
            background: "linear-gradient(180deg, var(--brass), var(--accent-dim))",
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
          <text x={(h % cols) * (size + 6) + size / 2} y={Math.floor(h / cols) * (size + 6) + size / 2 + 4} fill={h === selected ? "var(--lavender)" : "var(--text-1)"} fontSize={11} textAnchor="middle">
            {h}
          </text>
        </g>
      ))}
    </svg>
  );
}
