/* Learn: every concept the app uses, at three depths, grouped into chapters. Text describes how the app
 * works; every measured value elsewhere comes from the running model, not from this page. */

export type Depth = "simple" | "context" | "technical";

export interface Topic {
  id: string;
  term: string;
  chapter: number;
  /** where to see the idea on a real measurement */
  live: { label: string; href: string };
  simple: string;
  context: string;
  technical: string;
}

export const CHAPTERS = ["The model", "Inside a block", "Reading the output", "Running experiments"];

export const DEPTHS: { id: Depth; label: string; hint: string }[] = [
  { id: "simple", label: "Simple", hint: "one plain sentence" },
  { id: "context", label: "More context", hint: "a short paragraph" },
  { id: "technical", label: "Technical", hint: "shapes and hooks" },
];

export const TOPICS: Topic[] = [
  {
    id: "nn",
    term: "Neural network",
    chapter: 0,
    live: { label: "Model Explorer", href: "#/explorer" },
    simple: "A program that learns patterns from examples instead of following hand-written rules.",
    context:
      "A neural network is a stack of simple units whose strengths ('weights') are learned from data. GPT-2 was trained to guess the next word in billions of sentences, and in doing so it learned grammar, facts, and patterns.",
    technical:
      "GPT-2 Small is a 12-layer decoder-only transformer with 124M parameters trained on WebText with next-token cross-entropy. Its weights are frozen here: CircuitScope only runs forward passes, never training.",
  },
  {
    id: "token",
    term: "Token",
    chapter: 0,
    live: { label: "the tokens stage of the Explorer", href: "#/explorer" },
    simple: "A chunk of text — often a word or piece of a word — that the model actually reads.",
    context:
      "Models don't see letters. Text is split into tokens from a fixed vocabulary of about 50,000. Notice in the app that ' Mary' is one token, and that the space matters: 'Mary' and ' Mary' are different tokens.",
    technical:
      "GPT-2 uses byte-level BPE. Token IDs are mapped to vectors by the embedding matrix; TransformerBridge additionally prepends a BOS token (50256), which shifts positions — the app always shows the exact tokens of the running model.",
  },
  {
    id: "embedding",
    term: "Embedding",
    chapter: 0,
    live: { label: "the embeddings stage of the Explorer", href: "#/explorer" },
    simple: "Turning each token into a long list of numbers the network can process.",
    context:
      "Each token becomes a vector of 768 numbers (a point in a huge space). Similar words end up near each other. A position vector is added so the model knows word order.",
    technical:
      "hook_embed gives token embeddings (d_vocab → d_model matrix W_E), hook_pos_embed the learned positional embeddings. The residual stream starts as their sum: x₀ = E[tokens] + P[positions].",
  },
  {
    id: "attention",
    term: "Attention",
    chapter: 1,
    live: { label: "the attention heatmaps in the Explorer", href: "#/explorer" },
    simple: "A way for each word to look back at earlier words and pull in useful information.",
    context:
      "For each token, attention computes 'how much should I look at each earlier token?'. Those weights (visible in the Explorer heatmaps) always sum to 1 per row. It's a measurement of where information is gathered — not an explanation of what the model decides.",
    technical:
      "Scaled dot-product attention: A = softmax(QKᵀ/√d_head + causal mask), out = A·V. Q, K, V are linear projections of the pre-LN stream. hook_pattern stores A per head; rows sum to 1.",
  },
  {
    id: "head",
    term: "Attention head",
    chapter: 1,
    live: { label: "the head gallery in the Explorer", href: "#/explorer" },
    simple: "One of 12 small attention 'specialists' inside each layer.",
    context:
      "Each block's attention is split into 12 independent heads (block 5 has heads 0–11). Different heads often learn different behaviours — some track names, some copy words, some attend to punctuation.",
    technical:
      "GPT-2 Small: 12 heads × d_head 64 = d_model 768. Per-head z-values (hook_z, [seq, 12, 64]) are concatenated and projected by W_O. Heads are the smallest unit CircuitScope lets you patch.",
  },
  {
    id: "residual",
    term: "Residual connection",
    chapter: 1,
    live: { label: "the full map in the Explorer", href: "#/explorer" },
    simple: "A shortcut that lets information skip past a layer untouched.",
    context:
      "Instead of replacing the running signal, each block adds its contribution on top. This is why the internal signal is called the 'residual stream' — layers write into it, and later layers read from it.",
    technical:
      "x_{l+1} = x_l + Attn(LN1(x_l)) then + MLP(LN2(·)). The pre-LN ordering was verified against the running model. blocks.N.hook_out is the residual stream after block N — the primary patch site in the Playground.",
  },
  {
    id: "activation",
    term: "Activation",
    chapter: 2,
    live: { label: "the measured views in the Explorer", href: "#/explorer" },
    simple: "Any number inside the network while it processes your text.",
    context:
      "Between every operation the model holds vectors of intermediate results — the residual stream, attention outputs, MLP outputs. The Explorer shows real captured activations for your own input, always labeled with where they came from.",
    technical:
      "Activations are captured via forward hooks on the bridge's hook points (637 registered for gpt2). Shapes are [batch, seq, ...]; the backend records the exact shape and aggregation of anything displayed.",
  },
  {
    id: "logit",
    term: "Logit",
    chapter: 2,
    live: { label: "the logit lens in Sweeps", href: "#/sweeps" },
    simple: "The model's raw score for each possible next token, before probabilities.",
    context:
      "The final layer gives every vocabulary token a score. Higher = the model thinks it's more likely. The Playground tracks the difference between two tokens' scores (like ' Mary' minus ' John') to measure an intervention's effect precisely.",
    technical:
      "logits = ln_final(x)·W_U (+b_U), shape [seq, 50257]. D = logit(A) − logit(B) at the final position; Δ = D_patched − D_corrupted; Recovery normalizes by (D_clean − D_corrupted), guarded against near-zero denominators and not clamped.",
  },
  {
    id: "patching",
    term: "Intervention (activation patching)",
    chapter: 3,
    live: { label: "the Intervention Playground", href: "#/playground" },
    simple: "Swapping one internal signal with its value from a different run, then re-running the model.",
    context:
      "If patching the signal from sentence A into the run on sentence B moves the output toward A's answer, that's evidence the signal carried the relevant information. Small or zero effects are also real results — the app reports what it measures.",
    technical:
      "Forward hook replaces z[0, pos, :] (optionally per head) with the cloned clean-run tensor, then the corrupted prompt is re-run. The hook is removed after the run; baselines are never mutated (verified by tests).",
  },
  {
    id: "why",
    term: "Why attention alone isn't an explanation",
    chapter: 3,
    live: { label: "the patching sweeps", href: "#/sweeps" },
    simple: "Looking at something isn't the same as using it.",
    context:
      "A head can attend strongly to a word and still have little effect on the answer — and it can shape the answer with only modest attention. That's why CircuitScope separates descriptive views (attention heatmaps) from causal tests (patching experiments).",
    technical:
      "Attention weights are a function of Q·K only; they don't account for W_O·V or downstream processing. Causal claims require interventions with controls (no-op patches, baseline reproducibility) — provided in the backend test suite.",
  },
];
