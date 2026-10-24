import { useState } from "react";

type Depth = "simple" | "context" | "technical";

interface Topic {
  term: string;
  simple: string;
  context: string;
  technical: string;
}

const TOPICS: Topic[] = [
  {
    term: "Neural network",
    simple: "A program that learns patterns from examples instead of following hand-written rules.",
    context:
      "A neural network is a stack of simple units whose strengths ('weights') are learned from data. GPT-2 was trained to guess the next word in billions of sentences, and in doing so it learned grammar, facts, and patterns.",
    technical:
      "GPT-2 Small is a 12-layer decoder-only transformer with 124M parameters trained on WebText with next-token cross-entropy. Its weights are frozen here: CircuitScope only runs forward passes, never training.",
  },
  {
    term: "Token",
    simple: "A chunk of text — often a word or piece of a word — that the model actually reads.",
    context:
      "Models don't see letters. Text is split into tokens from a fixed vocabulary of about 50,000. Notice in the app that ' Mary' is one token, and that the space matters: 'Mary' and ' Mary' are different tokens.",
    technical:
      "GPT-2 uses byte-level BPE. Token IDs are mapped to vectors by the embedding matrix; TransformerBridge additionally prepends a BOS token (50256), which shifts positions — the app always shows the exact tokens of the running model.",
  },
  {
    term: "Embedding",
    simple: "Turning each token into a long list of numbers the network can process.",
    context:
      "Each token becomes a vector of 768 numbers (a point in a huge space). Similar words end up near each other. A position vector is added so the model knows word order.",
    technical:
      "hook_embed gives token embeddings (d_vocab → d_model matrix W_E), hook_pos_embed the learned positional embeddings. The residual stream starts as their sum: x₀ = E[tokens] + P[positions].",
  },
  {
    term: "Attention",
    simple: "A way for each word to look back at earlier words and pull in useful information.",
    context:
      "For each token, attention computes 'how much should I look at each earlier token?'. Those weights (visible in the Explorer heatmaps) always sum to 1 per row. It's a measurement of where information is gathered — not an explanation of what the model decides.",
    technical:
      "Scaled dot-product attention: A = softmax(QKᵀ/√d_head + causal mask), out = A·V. Q, K, V are linear projections of the pre-LN stream. hook_pattern stores A per head; rows sum to 1.",
  },
  {
    term: "Attention head",
    simple: "One of 12 small attention 'specialists' inside each layer.",
    context:
      "Each block's attention is split into 12 independent heads (block 5 has heads 0–11). Different heads often learn different behaviours — some track names, some copy words, some attend to punctuation.",
    technical:
      "GPT-2 Small: 12 heads × d_head 64 = d_model 768. Per-head z-values (hook_z, [seq, 12, 64]) are concatenated and projected by W_O. Heads are the smallest unit CircuitScope lets you patch.",
  },
  {
    term: "Activation",
    simple: "Any number inside the network while it processes your text.",
    context:
      "Between every operation the model holds vectors of intermediate results — the residual stream, attention outputs, MLP outputs. The Explorer shows real captured activations for your own input, always labeled with where they came from.",
    technical:
      "Activations are captured via forward hooks on the bridge's hook points (637 registered for gpt2). Shapes are [batch, seq, ...]; the backend records the exact shape and aggregation of anything displayed.",
  },
  {
    term: "Residual connection",
    simple: "A shortcut that lets information skip past a layer untouched.",
    context:
      "Instead of replacing the running signal, each block adds its contribution on top (shown as dashed violet paths in the diagram). This is why the internal signal is called the 'residual stream' — layers write into it, and later layers read from it.",
    technical:
      "x_{l+1} = x_l + Attn(LN1(x_l)) then + MLP(LN2(·)). The pre-LN ordering was verified against the running model. blocks.N.hook_out is the residual stream after block N — the primary patch site in the Playground.",
  },
  {
    term: "Logit",
    simple: "The model's raw score for each possible next token, before probabilities.",
    context:
      "The final layer gives every vocabulary token a score. Higher = the model thinks it's more likely. The Playground tracks the difference between two tokens' scores (like ' Mary' minus ' John') to measure an intervention's effect precisely.",
    technical:
      "logits = ln_final(x)·W_U (+b_U), shape [seq, 50257]. D = logit(A) − logit(B) at the final position; Δ = D_patched − D_corrupted; Recovery normalizes by (D_clean − D_corrupted), guarded against near-zero denominators and not clamped.",
  },
  {
    term: "Intervention (activation patching)",
    simple: "Swapping one internal signal with its value from a different run, then re-running the model.",
    context:
      "If patching the signal from sentence A into the run on sentence B moves the output toward A's answer, that's evidence the signal carried the relevant information. Small or zero effects are also real results — the app reports what it measures.",
    technical:
      "Forward hook replaces z[0, pos, :] (optionally per head) with the cloned clean-run tensor, then the corrupted prompt is re-run. The hook is removed after the run; baselines are never mutated (verified by tests).",
  },
  {
    term: "Why attention alone isn't an explanation",
    simple: "Looking at something isn't the same as using it.",
    context:
      "A head can attend strongly to a word and still have little effect on the answer — and it can shape the answer with only modest attention. That's why CircuitScope separates descriptive views (attention heatmaps) from causal tests (patching experiments).",
    technical:
      "Attention weights are a function of Q·K only; they don't account for W_O·V or downstream processing. Causal claims require interventions with controls (no-op patches, baseline reproducibility) — provided in the backend test suite.",
  },
];

const DEPTH_LABEL: Record<Depth, string> = {
  simple: "Simple",
  context: "More context",
  technical: "Technical",
};

/** Small accurate contextual diagrams — schematic, labeled, no fake data. */
function TopicDiagram({ term }: { term: string }) {
  const common = { fill: "none", stroke: "var(--line-1)" } as const;
  switch (term) {
    case "Token":
      return (
        <svg viewBox="0 0 300 54" width="100%" role="img" aria-label="A sentence split into tokens">
          <text x={8} y={33} fill="var(--text-1)" fontSize={13} fontFamily="var(--font-mono)">
            {`" the shop "`}
          </text>
          {[
            { x: 8, w: 34, t: `"·the` },
            { x: 46, w: 30, t: `shop` },
          ].map((s, i) => (
            <g key={i}>
              <rect x={140 + i * 76} y={14} width={s.w + 16} height={26} rx={4} fill="var(--bg-2)" stroke="var(--line-0)" />
              <text x={148 + i * 76} y={31} fill="var(--text-1)" fontSize={11} fontFamily="var(--font-mono)">
                {i === 0 ? "·the" : "shop"}
              </text>
            </g>
          ))}
          <text x={140} y={10} fill="var(--text-2)" fontSize={8}>
            tokens — note the leading space ·
          </text>
        </svg>
      );
    case "Attention":
      return (
        <svg viewBox="0 0 300 70" width="100%" role="img" aria-label="One query token attending back to earlier tokens with different weights">
          {[
            { x: 40, w: 0.9 },
            { x: 105, w: 0.5 },
            { x: 170, w: 0.25 },
            { x: 235, w: 0.1 },
          ].map((k, i) => (
            <g key={i}>
              <path
                d={`M 250 30 C 200 30, ${k.x + 20} 46, ${k.x} 54`}
                stroke="var(--accent)"
                strokeWidth={Math.max(0.8, k.w * 4)}
                opacity={0.35 + k.w * 0.6}
                fill="none"
              />
              <rect x={k.x - 14} y={54} width={44} height={12} rx={3} {...common} />
            </g>
          ))}
          <rect x={228} y={22} width={46} height={14} rx={3} fill="var(--bg-3)" stroke="var(--accent)" />
          <text x={251} y={32} fill="var(--text-0)" fontSize={9} textAnchor="middle">
            query
          </text>
          <text x={8} y={16} fill="var(--text-2)" fontSize={8}>
            thicker = larger attention weight (rows sum to 1)
          </text>
        </svg>
      );
    case "Residual connection":
      return (
        <svg viewBox="0 0 300 70" width="100%" role="img" aria-label="Blocks adding into a continuous residual stream">
          <path d="M 16 35 L 284 35" stroke="var(--violet)" strokeWidth={2} strokeDasharray="4 4" fill="none" />
          {[60, 150, 240].map((x, i) => (
            <g key={i}>
              <rect x={x - 26} y={12} width={52} height={16} rx={3} fill="var(--bg-3)" stroke="var(--line-1)" />
              <text x={x} y={23} fill="var(--text-2)" fontSize={8} textAnchor="middle">
                block {i}
              </text>
              <path d={`M ${x} 28 L ${x} 35`} stroke="var(--text-2)" fill="none" markerEnd="" />
            </g>
          ))}
          <text x={150} y={60} fill="var(--text-2)" fontSize={8} textAnchor="middle">
            each block adds its contribution to the same stream
          </text>
        </svg>
      );
    case "Attention head":
      return (
        <svg viewBox="0 0 300 60" width="100%" role="img" aria-label="Twelve parallel head lanes">
          {Array.from({ length: 12 }, (_, h) => (
            <g key={h}>
              <rect x={20 + h * 22} y={16} width={16} height={22} rx={3} fill="var(--bg-3)" stroke="var(--line-1)" />
              <text x={28 + h * 22} y={52} fill="var(--text-2)" fontSize={7} textAnchor="middle">
                H{h}
              </text>
            </g>
          ))}
          <text x={150} y={10} fill="var(--text-2)" fontSize={8} textAnchor="middle">
            12 parallel lanes — computed together, not one after another
          </text>
        </svg>
      );
    default:
      return null;
  }
}

export default function Learn() {
  const [depth, setDepth] = useState<Depth>("context");
  return (
    <main className="page page-narrow">
      <h1>Learn</h1>
      <p style={{ color: "var(--text-1)" }}>
        Short explanations of every concept CircuitScope uses, at three depths. These describe how the app
        works — every measured value you see elsewhere comes from the running model, not from this page.
      </p>

      <div className="depth-toggle" role="tablist" aria-label="Explanation depth">
        {(Object.keys(DEPTH_LABEL) as Depth[]).map((d) => (
          <button
            key={d}
            role="tab"
            aria-selected={depth === d}
            className={depth === d ? "active" : ""}
            onClick={() => setDepth(d)}
          >
            {DEPTH_LABEL[d]}
          </button>
        ))}
      </div>

      <div className="panel" style={{ marginTop: "1rem" }}>
        <h2>What CircuitScope is — and is not</h2>
        <p style={{ color: "var(--text-1)" }}>
          <strong>Is:</strong> a faithful diagram of a real transformer, live token-level measurements from
          named hooks, and controlled activation-patching experiments with reproducible numbers.
        </p>
        <p style={{ color: "var(--text-1)" }}>
          <strong>Is not:</strong> a claim that the model's complete reasoning is visible, a proof that any
          single head "understands" a concept, or a statistical study. An attention heatmap is a
          measurement; a patch result is evidence under that specific intervention.
        </p>
      </div>

      {TOPICS.map((t) => (
        <div className="panel" key={t.term}>
          <h3>{t.term}</h3>
          <TopicDiagram term={t.term} />
          <p style={{ color: "var(--text-1)" }}>{t[depth]}</p>
        </div>
      ))}

      <div className="panel">
        <h3>Run it yourself</h3>
        <p className="inspector-meta">
          Backend: <span className="mono">.venv\Scripts\python -m backend.app.main</span> (from the project
          root; serves on http://127.0.0.1:8000). Frontend:{" "}
          <span className="mono">cd frontend && npm run dev</span> (http://localhost:5173). The model
          downloads once (~500 MB) into the Hugging Face cache and is reused afterwards.
        </p>
      </div>
    </main>
  );
}
