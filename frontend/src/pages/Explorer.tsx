import { useEffect, useMemo, useState } from "react";
import {
  api,
  type AttentionPattern,
  type FlowData,
  type Health,
  type ModelMeta,
  type Tokenization,
} from "../api/client";
import { BlockDiagram, FullModelDiagram } from "../components/ArchDiagram";
import AttentionHeatmap from "../components/AttentionHeatmap";
import FlowPlayback from "../components/FlowPlayback";
import TokenRow from "../components/TokenRow";

type Level = "model" | "block" | "head";
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

  const breadcrumbs = useMemo(() => {
    const items: { label: string; go: () => void }[] = [
      { label: "Whole model", go: () => setLevel("model") },
    ];
    if (level !== "model") items.push({ label: `Block ${layer}`, go: () => setLevel("block") });
    if (level === "head") items.push({ label: `Head ${head}`, go: () => setLevel("head") });
    return items;
  }, [level, layer, head]);

  if (!ready) {
    return (
      <main className="page">
        <h1>Model Explorer</h1>
        <div className="state-banner info">
          <span className="spinner" aria-hidden /> The model is loading. The explorer reads real layer/head
          counts and captured activations, so it needs the model ready. This takes ~1 minute after backend
          start.
        </div>
      </main>
    );
  }

  return (
    <main className="page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.8rem" }}>
        <div>
          <h1>Model Explorer</h1>
          <p className="breadcrumbs" aria-label="Breadcrumb">
            {breadcrumbs.map((b, i) => (
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
          {meta.model_id} · {meta.n_layers} blocks · {meta.n_heads} heads · d_model {meta.d_model} ·{" "}
          {meta.device}
        </div>
      </div>

      <div className="panel">
        <div className="grid-2">
          <div>
            <label htmlFor="explorer-text" style={{ fontSize: "0.85rem", color: "var(--text-2)", display: "block", marginBottom: "0.3rem" }}>
              Input for measured views
            </label>
            <textarea id="explorer-text" rows={2} value={text} onChange={(e) => setText(e.target.value)} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
            <button
              className="primary"
              onClick={() => {
                setRunText(text);
                setFlow(null);
                runCapture(text, layer);
              }}
              disabled={busy}
            >
              {busy ? "Capturing…" : "Run & capture activations"}
            </button>
            <button onClick={() => { setText(EXAMPLE); setRunText(EXAMPLE); runCapture(EXAMPLE, layer); }}>
              Use example
            </button>
          </div>
        </div>
        <h3 style={{ marginTop: "0.8rem" }}>Tokens received by the model (position IDs shown)</h3>
        <TokenRow tok={tokenization} />
        {error && <div className="state-banner error" style={{ marginTop: "0.6rem" }}>{error}</div>}
      </div>

      {mode === "flow" ? (
        <div style={{ marginTop: "1rem" }}>
          <p style={{ color: "var(--text-1)", maxWidth: "80ch" }}>
            A guided replay of one real forward pass through the model: tokenization, embeddings, every
            transformer block, final normalization, and output scores. Every heatmap, norm, and score below
            was captured from the actual run on your text.
          </p>
          <FlowPlayback flow={flow} />
        </div>
      ) : (
        <>
          {level === "model" && (
            <div className="grid-2" style={{ alignItems: "start", marginTop: "1rem" }}>
              <div className="panel">
                <h2>End-to-end computational path</h2>
                <FullModelDiagram
                  meta={meta}
                  selectedLayer={null}
                  onSelectLayer={(l) => {
                    setLayer(l);
                    setLevel("block");
                  }}
                  tokenCount={tokenization?.n_tokens ?? null}
                />
              </div>
              <div className="panel">
                <h2>Measured residual stream (data view)</h2>
                <p className="inspector-meta">
                  L2 norm of the residual stream after each block, averaged over tokens — from the real
                  captured run on your input. How information accumulates as it travels upward.
                </p>
                <ResidNormAcrossLayers flow={flow} onSelectLayer={(l) => { setLayer(l); setLevel("block"); }} />
                <h3 style={{ marginTop: "1rem" }}>Jump to a block</h3>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem" }}>
                  {Array.from({ length: meta.n_layers }, (_, l) => (
                    <button
                      key={l}
                      style={{ padding: "0.25rem 0.6rem", fontSize: "0.8rem" }}
                      onClick={() => {
                        setLayer(l);
                        setLevel("block");
                      }}
                    >
                      B{l}
                    </button>
                  ))}
                </div>
                <p className="schematic-note">
                  <span className="static-badge">MEASURED</span> norms from blocks.N.hook_out; the diagram on
                  the left is structure from live metadata.
                </p>
              </div>
            </div>
          )}

          {level === "block" && (
            <div className="grid-2" style={{ alignItems: "start", marginTop: "1rem" }}>
              <div className="panel">
                <h2>Inside transformer block {layer}</h2>
                <BlockDiagram
                  meta={meta}
                  layer={layer}
                  onSelectAttention={() => setLevel("head")}
                  onSelectMlp={() => setLevel("head")}
                />
              </div>
              <div className="panel">
                <h3>Computation order (verified pre-LN)</h3>
                <table className="kv-table">
                  <tbody>
                    <tr><td>1 · input</td><td>residual stream, {meta.d_model} dims</td></tr>
                    <tr><td>2 · LN1</td><td>normalize → feeds attention only</td></tr>
                    <tr><td>3 · attention</td><td>{meta.n_heads} heads × d_head {meta.d_head}, causal mask</td></tr>
                    <tr><td>4 · add</td><td>attention output added to stream</td></tr>
                    <tr><td>5 · LN2</td><td>normalize → feeds MLP only</td></tr>
                    <tr><td>6 · MLP</td><td>{meta.d_model} → {meta.d_mlp} → {meta.d_model}, GELU</td></tr>
                    <tr><td>7 · add</td><td>MLP output added to stream → next block</td></tr>
                  </tbody>
                </table>
                <h3 style={{ marginTop: "1rem" }}>Measured residual norm after this block</h3>
                {flow && (
                  <TokenNormRow
                    norms={flow.blocks[layer].resid_norm_per_token}
                    tokens={flow.tokens}
                  />
                )}
                <h3 style={{ marginTop: "1rem" }}>All {meta.n_heads} heads of this block</h3>
                <p style={{ color: "var(--text-1)" }}>Select a head to inspect its measured attention pattern.</p>
                <HeadGrid
                  nHeads={meta.n_heads}
                  selected={head}
                  onSelect={(h) => {
                    setHead(h);
                    setLevel("head");
                  }}
                />
              </div>
            </div>
          )}

          {level === "head" && (
            <div className="grid-2" style={{ alignItems: "start", marginTop: "1rem" }}>
              <div className="panel">
                <h2>
                  Attention head {head} of block {layer}
                </h2>
                <AttentionHeatmap att={attention} head={head} />
                <div style={{ marginTop: "0.8rem" }}>
                  <HeadGrid nHeads={meta.n_heads} selected={head} onSelect={(h) => setHead(h)} />
                </div>
              </div>
              <div className="panel">
                <h3>Numerical inspector</h3>
                <dl className="inspector-meta">
                  <dt>Hook identifier</dt>
                  <dd>blocks.{layer}.attn.hook_pattern</dd>
                  <dt>Tensor shape</dt>
                  <dd>{attention ? attention.shape.join(" × ") : "—"} (heads × query pos × key pos)</dd>
                  <dt>Displayed slice</dt>
                  <dd>head {head}, all {attention?.tokens.length ?? "—"} query/key positions (unreduced)</dd>
                  <dt>Value scale</dt>
                  <dd>softmax attention weights ∈ [0, 1]; each query row sums to 1.0</dd>
                  <dt>Source run</dt>
                  <dd>{runText.slice(0, 48)}{runText.length > 48 ? "…" : ""}</dd>
                  <dt>Provenance</dt>
                  <dd>{meta.model_id} · {meta.device} · {meta.dtype} · captured live</dd>
                </dl>
                <p className="inspector-meta" style={{ marginTop: "0.8rem" }}>
                  <strong>Reading this honestly:</strong> an attention pattern shows where a head distributes
                  its query attention. It is a descriptive measurement, <em>not</em> proof of what the head
                  causes — that requires the intervention experiments in the Playground.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}

/** Measured residual norm per token, rendered under the token labels. */
function TokenNormRow({ norms, tokens }: { norms: number[]; tokens: string[] }) {
  const max = Math.max(...norms, 1e-9);
  return (
    <div className="norm-chart-wrap">
      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 60 }}>
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
      <div style={{ display: "flex", gap: 3 }}>
        {tokens.map((t, i) => (
          <div key={i} style={{ flex: 1, textAlign: "center", fontSize: "0.6rem", color: "var(--text-2)", overflow: "hidden", whiteSpace: "nowrap" }}>
            {t}
          </div>
        ))}
      </div>
    </div>
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
            background: "linear-gradient(180deg, var(--accent), var(--accent-dim))",
            borderRadius: 3,
            cursor: "pointer",
            position: "relative",
          }}
        >
          <span
            style={{
              position: "absolute", bottom: -18, left: 0, right: 0,
              textAlign: "center", fontSize: "0.62rem", color: "var(--text-2)",
            }}
          >
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
