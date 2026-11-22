# Prompt sets, controls and attribution patching

One-page notes on the ideas used here. Each has an analogy, the maths in a
few lines, and the question a reviewer is likely to ask.

## 1. Activation patching vs ablation - sufficiency vs necessity

**Analogy.** Patching is swapping one student's note into another student's
answer sheet and seeing how much of the first student's answer appears.
Ablation is taking the note away from the first student and seeing how much
of their answer disappears.

**Maths (denoising patch, the direction we use everywhere).** Run the model on
a clean prompt and a corrupted prompt. Copy component c's clean activation
a_clean into the corrupted run and measure the metric (D = logit(IO) − logit(S)
at the final position):

    Rec_D = (D_patched − D_corr) / (D_clean − D_corr)

Rec_D ≈ 1 means c alone carries (restores) most of the clean behaviour;
Rec_D ≈ 0 means c alone does not matter for this metric.

**Maths (ablation, on the clean run).** Replace c's activation in the CLEAN
run by zeros (or by a mean over other prompts) and measure:

    Abl_D = D_clean − D_ablated      (raw logit units, not normalised)

**Why they differ.** Patching asks "is c SUFFICIENT to (partly) cause the
behaviour, given everything else is broken?"; ablation asks "is c NECESSARY -
does the behaviour survive without it?". A redundant circuit makes these
answers differ: several components can each be partly sufficient while no
single one is necessary (removing one, the others take over). This is why the
study never mixes the two in one figure without labelling, and why a
component can score high in one and low in the other.

**Likely question.** "Which one is the real importance?" - Neither alone.
They answer different counterfactual questions; the claim to make is about the
measured effect under the stated intervention.

## 2. Attribution patching (AtP) and its limit

**Analogy.** Exact patching reruns the whole model for every component
(expensive). AtP asks the model for a slope: "if I nudged this activation by
the (clean − corrupted) difference, roughly how much would the metric move?"
- a first-order estimate obtained from one backward pass instead of 156
forward passes.

**Maths.** With the gradient of the metric D w.r.t. the activation a,
evaluated at the corrupted run:

    AtP_D(c) = Σ_positions Σ_dims (a_clean − a_corr) · dD/da |_corr
    Rec_AtP  = AtP_D / (D_clean − D_corr)

**The limit (stated openly everywhere it is shown).** A linear approximation
is exact only if D is linear in a. When attention saturates (softmax near 0/1),
D is locally flat and the gradient under- or over-estimates the true effect.
So disagreement between AtP and exact patching is expected for some
components and is a RESULT, not a bug. We gate the implementation, not the
science: the linearity gate (central difference of D around alpha = 0, via
`interpolated_patch_logits`) must match AtP_D (Pearson r > 0.999) - that
checks the calculus, not the approximation's real-world accuracy.

**Likely question.** "Why use it at all?" - Because it makes 156-component
studies cheap, and measuring WHERE the cheap method disagrees with exact
patching is itself one of the pre-registered research questions (RQ1).

## 3. Spearman rank correlation (average ranks)

**Analogy.** Pearson asks "do the numbers move together?"; Spearman asks "do
the ORDERINGS agree?" - only ranks matter, so a component scoring 10 vs 1
counts the same as 10 vs 0.1.

**Maths.** Rank each vector (ties get the average of their ranks), then take
the Pearson correlation of the ranks:

    rho = Pearson(rank(x), rank(y)),     rho ∈ [−1, 1]

Constant input has no defined ordering → we return `None`, never NaN. We
compute per-prompt rho over 144 heads, 12 MLPs (flagged: n=12 is low
resolution) and all 156 components, using signed values (primary) and
absolute values (secondary).

**Likely question.** "Why not the raw score magnitudes?" - Because the study
question is about RANKINGS (which heads matter most), which is what RQ2
needs too.

## 4. Bootstrap 95% CI over prompts

**Analogy.** You cannot collect more prompts retroactively, so you pretend to
re-collect them: resample the prompts you have (with replacement) many times
and see how much the statistic wobbles.

**Maths.** With per-prompt values ρ_1..ρ_n, draw B = 2000 resamples of size n
with replacement, compute the mean of each, and report the 2.5/97.5
percentiles of those means:

    CI = [percentile_2.5(means), percentile_97.5(means)]

Implemented with `numpy.random.default_rng(seed)` - deterministic for a fixed
seed. The CI is over PROMPTS (the sampling unit is the prompt), not over
components.

**Likely question.** "Is this a significance test?" - No; it quantifies how
stable the mean is under prompt resampling. The control that gives it
meaning is the permutation null below.

## 5. Permutation null (the noise floor)

**Analogy.** Shuffle the answer key: correlate the exact scores against a
RANDOMLY permuted copy of themselves. Any agreement left is pure coincidence
- the noise floor the real method must beat.

**Maths.** For each prompt, 1000 random permutations of the component labels;
correlate the exact scores with the permuted scores; report the pooled 2.5/97.5
percentiles. A cheap method "beats the control" only if the LOWER bound of its
bootstrap CI exceeds the null 97.5th percentile (i.e. even its pessimistic
end is above the coincidence floor at the same 95% level).

**Likely question.** "Why this double-95% rule?" - It compares the method's
CI (95% confident about the mean) against the null's 97.5th percentile, so a
"beats" verdict is conservative rather than generous.

## 6. Top-k overlap and the random expectation

**Analogy.** Two conditions each nominate their top-5 heads; overlap |A ∩ B|
asks how much the shortlists agree. Two RANDOM shortlists would still share
some heads by luck.

**Maths.** Rank the 144 heads by signed score per condition; |A ∩ B| for the
top-k sets. Two independent random k-subsets of 144 share on average

    E[|A ∩ B|] = k·k / 144     (k=5 → 0.17; k=10 → 0.69)

The random expectation is printed next to every overlap number.

**Likely question.** "Why both Spearman and overlap?" - Spearman sees the
whole ranking; top-k overlap focuses on the heads people would actually name.
They can disagree (a shuffle in the tail changes rho little but overlap not at
all), which is exactly the metric-sensitivity RQ2 studies.

## 7. Why we exclude small-denominator prompts

**Analogy.** If the corrupted prompt barely changes the model's answer
(D_clean ≈ D_corr), then the "yardstick" (D_clean − D_corr) that normalises
every recovery is nearly zero: tiny measurement noise gets blown up into
huge, meaningless recoveries.

**Maths.** Rec_D divides by (D_clean − D_corr). The pre-registered rule
(fixed before the full run): per (prompt, corruption) pair,
exclude iff |D_clean − D_corr| < 0.5; count exclusions per condition; report
model accuracy (fraction with D_clean > 0) and repeat headline numbers on the
D_clean > 0 subset. Zero denominator → recovery undefined (`None`), never a
huge number. No other prompt may be dropped.

**Likely question.** "Isn't exclusion cherry-picking?" - The rule was fixed
and committed before results existed, is symmetric (no looking at which way
the recovery went), and every exclusion is counted and reported.
