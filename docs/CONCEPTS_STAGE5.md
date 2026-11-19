# Concepts for Stage 5: trust, checking and the tour

Plain-language notes for the three trust features. Each is about whether the
numbers on screen were computed correctly. None of them checks whether the
scientific explanation is right.

## 1. What a reproduction script does and does not prove

A "Reproduce in Python" download is a standalone script for one result. It
loads GPT-2 itself (through the raw HuggingFace `transformers` code, not the
app's code), recomputes the numbers the app displayed, and compares them with
the stored values. It prints a table and exits 0 only if every compared value
is within its tolerance.

It proves:
- The numbers shown for that result are reproducible by a second, independent
  implementation on this model revision, within the stated tolerance.
- The stored numbers were not edited afterwards (a changed number makes the
  script exit non-zero; see `V2_STAGE5_VERIFICATION.md` section 3).

It does NOT prove:
- That the explanation is right. "Head 9.6 carries the name signal" is a claim
  about cause; the script only confirms the arithmetic behind a number.
- That the chosen prompt is representative. One pair of sentences is one
  case.
- That a different GPT-2 revision or library version gives the same numbers. The
  script prints a warning if the local torch/transformers versions differ from
  the stored ones.

## 2. What the verification statuses mean

The Verify page runs the repository's reference scripts from the UI. Each card
shows one of these statuses (always in words and an icon, never colour alone):

| status | meaning |
|---|---|
| **Pass** | the reference script ran, printed a max difference, and that difference is within the tolerance. |
| **Fail** | the script exited non-zero, timed out, or the max difference is above the tolerance. The error text says which. |
| **Never run** | no stored result for this check on this machine yet. |
| **Stale result** | the stored pass/fail was for an earlier version of the script (the script's SHA-256 has changed since the run). The commit is shown but does not decide staleness. |
| **Manual check** | the check needs a browser (feedforward in the Neural Lab) and cannot run from the server. |

A pass means an independent implementation agrees with the app on these numbers,
within the tolerance. It does not validate the scientific interpretation, the
choice of prompt, or the report's wording.

## 3. AtP vs DLA vs exact patching, in plain language

Each method answers "how much does this component matter?" in a different way.
The question is the same for each of the 156 components (144 attention heads
and 12 MLP blocks).

- **Exact patching** (the reference). Run the model on the corrupted prompt, but
  restore one component's activation from the clean run, then measure how much
  of the clean answer comes back (the "recovery"). It is slow and it is what the
  circuit claims are based on.
- **Attribution patching (AtP).** Estimate the same recovery with one backward
  pass: the change in the activation times the gradient of the score. It is about
  as fast as one forward and backward pass for all components, but it is a
  linear approximation. It can be wrong where the effect is non-linear.
- **Direct logit attribution (DLA).** On the clean prompt only, add up how much
  each component writes directly to the final logits. It does not see effects
  that go through later components, so it is a "direct effect" measure, not
  the same quantity as patching.

What to expect:
- AtP and exact patching should agree closely on which components matter most
  when the effects are roughly linear. Where they disagree, the exact method is
  the reference.
- DLA and exact patching measure different things. Low agreement is expected
  for components whose effect is mostly indirect. The agreement panel says so.
- Agreement on one prompt pair says little about other prompts. The statistics
  describe these scores on this pair only.

The agreement panel reports Pearson and Spearman correlations against exact
values, the overlap of the top-10 lists against the random expectation, and the
correlation split into early (blocks 0 to 5) and late (blocks 6 to 11) groups. The
group sentence appears only when the two groups differ by more than 0.2.

## 4. The guided tour

The tour shows one real report in six scenes. It reuses a saved report only when
its provenance commit matches the current commit (ignoring a `+dirty` suffix).
Otherwise it generates a new report, which takes about 20 to 30 seconds of real
computation. Auto-advance runs about every 14 seconds (about 90 seconds in total).
It never auto-advances when reduced motion is requested. Every number on screen
comes from the report.
