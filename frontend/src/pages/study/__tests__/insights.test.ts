import { describe, expect, it } from "vitest";
import { agreementCallouts, stabilityCallouts, topHeadCallouts } from "../insights";
import type { KnownHeadsReference, Rq1Block, Rq2 } from "../../../api/client";

const group = (mean: number) => ({ signed: { mean, ci_low: mean - 0.01, ci_high: mean + 0.01, n: 4, n_invalid: 0 }, abs: { mean, ci_low: 0, ci_high: 1, n: 4, n_invalid: 0 }, n_components: 144, low_resolution: false });

describe("study callouts", () => {
  it("states both agreement numbers and whether each beats the control", () => {
    const rq1 = {
      n_prompts: 4,
      methods: {
        atp: { heads: group(0.976), mlps: group(0.7), all: group(0.9) },
        dla: { heads: group(0.17), mlps: group(0.1), all: group(0.17) },
      },
      beats_control: {
        atp: { heads: true, mlps: true, all: true },
        dla: { heads: false, mlps: false, all: false },
      },
    } as unknown as Rq1Block;
    expect(agreementCallouts(rq1)).toEqual([
      { lead: "Attribution patching:", text: "0.98 on heads, above the control" },
      { lead: "Direct logit attribution:", text: "0.17 on heads, not above the control" },
    ]);
  });

  it("names the leading head and the most common published group in the top ten", () => {
    const vec = new Array(144).fill(0);
    vec[5 * 12 + 5] = 0.32; // 5.5
    vec[8 * 12 + 10] = 0.29; // 8.10
    vec[8 * 12 + 6] = 0.27; // 8.6
    const known: KnownHeadsReference = {
      groups: [
        { group: "induction", heads: ["5.5"] },
        { group: "s_inhibition", heads: ["8.10", "8.6"] },
      ],
    };
    const out = topHeadCallouts(vec, known);
    expect(out[0]).toEqual({ lead: "Leader:", text: "head 5.5 (induction), 0.320" });
    expect(out[1]).toEqual({ lead: "S-inhibition heads:", text: "2 of the top 10" });
  });

  it("finds the most and least alike pairs of setups", () => {
    const rq2 = {
      condition_names: ["swap_D", "swap_P", "abl_zero_D"],
      topk: {
        k5: { matrix: [] },
        k10: {
          matrix: [
            [10, 9, 2],
            [9, 10, 4],
            [2, 4, 10],
          ],
        },
      },
    } as unknown as Rq2;
    expect(stabilityCallouts(rq2, "k10")).toEqual([
      { lead: "Most alike:", text: "swap D and swap P share 9 of 10" },
      { lead: "Least alike:", text: "swap D and abl zero D share 2 of 10" },
    ]);
  });
});
