# CONCEPTS_STAGE2 — patching sweeps, DLA, logit lens (plain language)

Written for a smart non-expert. Every formula here is exactly what the code
computes; every number in the app comes from running the real model.

## 1. The residual stream is a shared whiteboard

GPT-2 processes a sentence token by token, but internally each token position
carries a 768-number vector called the **residual stream** — think of it as one
row of a shared whiteboard. Each of the 12 blocks reads the row and *adds*
something to it:

```
r_after_block_11 = embed + pos_embed
                   + (attn_out_0 + mlp_out_0)
                   + ...
                   + (attn_out_11 + mlp_out_11)
```

Each attention block's output is itself a sum over its 12 heads plus a bias:

```
attn_out_L = sum_h (z_{L,h} @ W_O[L,h]) + b_O[L]
```

so the whole final row is the sum of about 171 pieces: 2 embeddings, 144 heads,
12 attention biases, 12 MLPs. That additivity is a fact about the architecture
(verified in code: the pieces sum to the real residual to within float
rounding), and it is what makes the two techniques below possible.

At the very end, the row goes through a final LayerNorm and then a big matrix
(`W_U`) that turns 768 numbers into 50,257 scores — one per vocabulary token.
We compare two tokens, A (" Mary") and B (" John"):

```
D = LN(r) · (W_U[:, A] − W_U[:, B])
```

Positive D means the model scores A above B at that position. D for the IOI
clean prompt is about 2.51; for the corrupted prompt ("Mary gave…"), about
−4.18.

## 2. Direct logit attribution (DLA) — who wrote which bit of the answer?

**Analogy.** After the essay is finished, attribute each sentence of the final
draft to whoever wrote it on the whiteboard. It tells you who *directly* wrote
what — not who prompted whom.

The math is one line. For any component's output `c` (a 768-vector):

```
DLA(c) = ((c − mean(c)) / s) · (γ * w)          # s from the FULL residual r
w     = W_U[:, A] − W_U[:, B]
```

`s` is the scale the final LayerNorm divides by (computed from the whole row,
not from `c`), γ is the LayerNorm's learned gain, and the mean-subtraction is
the LayerNorm's centering. Because centering is linear, the pieces telescope:

```
sum over all components of DLA(c)  +  (final-LN bias term)  =  D
```

This **reconciliation identity** is checked on every single call and shown in
the UI ("components sum to X; model D is Y"). If it ever failed, the app says
so instead of hiding it.

**What DLA cannot tell you** (shown as a note in the app): it measures only
*direct* writes. If head 9.6 writes "Mary" information at position 13, DLA
credits head 9.6. But if head 5.9 *earlier* moved "the recipient is Mary"
information to the final position and head 9.6 merely read it and rewrote it,
DLA gives the early head little credit. Effects that flow *through later
layers* are invisible. That is why DLA is always shown as a partial view, and
why it is cross-checked against patching (which is causal).

*You should be able to defend:* "Why can you attribute logits to individual
heads at all?" — Because the residual stream is literally a sum of their
outputs, and (apart from the LayerNorm's linear pieces) the final logits are a
linear read of that sum. "What does DLA miss?" — Indirect paths through later
layers; a small DLA does not mean a small causal role.

## 3. Patching sweeps — swap one note between two students

**Analogy.** Two students solve the same problem with one detail changed (the
IOI clean and corrupted prompts). You copy one whiteboard note from the clean
run into the corrupted run and see how much of the clean answer comes back.

Direction is fixed and stated everywhere it is displayed: **clean activations
are copied into the corrupted run ("denoising")**. Intuitively: "the corrupted
run lost the correct behaviour; how much do we get back by restoring this
internal value?"

Per cell (a layer × position, or layer × head):

```
Recovery = (D_patched − D_corrupted) / (D_clean − D_corrupted)
```

Recovery ≈ 1: that site alone carries (almost) all of the behaviour.
Recovery ≈ 0: that site alone does not matter for this metric. It is not
clamped and not a probability; the denominator is the real clean-vs-corrupted
contrast, and if that contrast is near zero the sweep refuses to run rather
than show noise.

Honest caveats the UI carries: patching is causal evidence about *the patch*,
not proof that a component is "the" member of "the" circuit — GPT-2's IOI
circuit is redundant, so importance can be distributed. And a single-cell
Recovery of ~0 does not mean the component does nothing (the task may have
alternatives that took over).

*You should be able to defend:* "Why denoising and not the reverse?" — The
metric definition requires a direction; we state ours explicitly and keep it
consistent (the Playground has always used this direction). "Why is a low
Recovery cell not proof of irrelevance?" — Because other components can
compensate, and because Recovery depends on the metric and corruption choice.

## 4. Logit lens — peek at the draft after every paragraph

**Analogy.** After each block, apply the *final* LayerNorm and unembedding to
the current whiteboard row: "if the model had to answer right now, what would
it say?"

The math reuses the model's own readout exactly — same LayerNorm (biased
variance, the model's own epsilon, learned gain and bias), same `W_U`, no
invented parameters:

```
logits_after_block_i = LN(r_i) @ W_U
```

The app shows, per stage: D, P(A), P(B), and the top-5 tokens with
probabilities. A built-in identity check confirms the last stage (block 11)
reproduces the model's real final logits — so the lens is a *view* of the same
model, not a second model.

**The caveat (always shown):** the logit lens is a heuristic. Early-layer
residual streams are not trained to be read by the unembedding — nobody trains
block 2's output to be decodable — so early "predictions" are a rough trace of
how the output belief forms, not what the model "was thinking". Expect early
stages to look like noise or generic text; the interesting structure is usually
in the later blocks (for IOI, the correct token dominates from block 9).

*You should be able to defend:* "Is the lens causal evidence?" — No. It is a
readout of the residual stream at one point in time, nothing was intervened
on. "Why does the final stage match the model exactly?" — Because block 11's
output is exactly what the real model feeds to ln_f + W_U; the lens is the
same computation.

## 5. How the three cross-check each other

- Patching = causal (we changed something and observed the effect).
- DLA = attributional and *partial* (direct writes only).
- Lens = descriptive (no intervention at all).

Where DLA and patching agree (e.g. for the IOI name-mover heads, our measured
top DLA contributors at layers 9–10), the evidence is strong. Where they
disagree, the honest conclusion is "the direct view is incomplete here" — that
disagreement is exactly what the later agreement study (Stage 3+) will
quantify. Nothing in the app claims causation from the lens or from attention
patterns.
