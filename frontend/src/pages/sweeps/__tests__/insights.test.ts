import { describe, it, expect } from "vitest";
import { agreementCallouts, dlaCallouts, lensCallouts, patchingCallouts } from "../insights";
import type { AgreementResult, DlaResult, LensResult, SweepResult } from "../../../api/client";

function sweep(over: Partial<SweepResult>): SweepResult {
  return {
    job_type: "sweep",
    mode: "layer_position",
    task_id: "t",
    pair_id: "p",
    direction: "clean -> corrupted",
    tokens_clean: [],
    tokens_corrupted: [],
    candidate_a: " Mary",
    candidate_b: " John",
    baselines: { D_clean: 2.5, D_corrupted: -4.1, denominator: 6.6, warnings: [] },
    row_labels: ["block 0", "block 1", "block 2", "block 3"],
    col_labels: ["John", "and", "to"],
    recovery: [],
    d_patched: [],
    n_runs: 12,
    elapsed_seconds: 1,
    provenance: {},
    ...over,
  };
}

describe("patchingCallouts", () => {
  it("names the strongest word and the hand-over to the last word", () => {
    const r = sweep({
      recovery: [
        [0.9, 0, 0],
        [0.9, 0, 0],
        [0.2, 0, 0.6],
        [0.1, 0, 0.9],
      ],
    });
    const out = patchingCallouts(r);
    expect(out[0]).toEqual({ lead: "Strongest:", text: "“John”, block 0 to block 1 (peak 0.90)" });
    expect(out[1]).toEqual({ lead: "Hand-over:", text: "“to” carries it from block 2" });
  });

  it("says so when nothing matters", () => {
    const r = sweep({ recovery: [[0.01, 0.02, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] });
    expect(patchingCallouts(r)[0].lead).toBe("Little effect:");
  });

  it("lists the top heads in layer × head mode", () => {
    const r = sweep({
      mode: "layer_head",
      col_labels: ["head 0", "head 1", "head 2"],
      recovery: [[0.1, 0.8, 0], [0.6, 0, 0], [0, 0, 0.3], [0, 0, 0]],
    });
    const out = patchingCallouts(r);
    expect(out.map((c) => c.text)).toEqual(["block 0, head 1 (0.80)", "block 1, head 0 (0.60)", "block 2, head 2 (0.30)"]);
  });
});

describe("dla, lens and agreement callouts", () => {
  it("finds the top pusher and blocker", () => {
    const d = {
      components: [
        { name: "head 9.6", kind: "head", layer: 9, head: 6, value: 1.6 },
        { name: "head 10.7", kind: "head", layer: 10, head: 7, value: -1.0 },
      ],
      reconciliation: { sum_of_terms: 0.6, model_D: 0.6, abs_error: 0, tolerance: 0.002, ok: true },
    } as unknown as DlaResult;
    const out = dlaCallouts(d);
    expect(out[0].text).toBe("head 9.6 (+1.60)");
    expect(out[1].text).toBe("head 10.7 (-1.00)");
    expect(out[2].lead).toBe("Adds up:");
  });

  it("reports where the final answer settles", () => {
    const stage = (stage: string, token: string) => ({ stage, layer: null, D: null, prob_a: null, prob_b: null, top: [{ token, token_id: 1, logit: 0, prob: 0.5 }] });
    const l = {
      stages: [stage("embeddings", "x"), stage("block 0", "y"), stage("block 1", " Mary"), stage("block 2", " Mary")],
    } as unknown as LensResult;
    expect(lensCallouts(l)).toEqual([{ lead: "Settles:", text: "“Mary” is the top guess from block 1 onward" }]);
  });

  it("states both agreement numbers", () => {
    const r = { stats: { atp: { pearson_heads: 0.984 }, dla: { pearson_heads: 0.631 } } } as unknown as AgreementResult;
    expect(agreementCallouts(r).map((c) => c.text)).toEqual([
      "attribution patching tracks exact patching at 0.984",
      "DLA agrees at 0.631",
    ]);
  });
});
