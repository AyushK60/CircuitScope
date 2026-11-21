import { describe, expect, it } from "vitest";
import { describeHook, headline, recoveryPercent, recoveryVerb, residualBlock } from "../insights";

describe("playground insights", () => {
  it("describes hooks in words", () => {
    expect(describeHook("blocks.5.hook_out")).toBe("block 5 (residual stream)");
    expect(describeHook("blocks.7.attn.hook_out")).toBe("block 7 (attention output)");
    expect(describeHook("blocks.5.attn.hook_z")).toBe("block 5 (attention z)");
    expect(describeHook("blocks.9.mlp.hook_out")).toBe("block 9 (MLP output)");
  });

  it("only residual hooks map to a block button", () => {
    expect(residualBlock("blocks.11.hook_out")).toBe(11);
    expect(residualBlock("blocks.7.attn.hook_out")).toBeNull();
  });

  it("reads recovery from the number, including no effect and wrong direction", () => {
    expect(recoveryVerb(0.915)).toContain("most");
    expect(recoveryVerb(0.5)).toContain("part");
    expect(recoveryVerb(0.2)).toContain("a little");
    expect(recoveryVerb(0.01)).toContain("almost no");
    expect(recoveryVerb(-0.3)).toContain("further");
    expect(recoveryVerb(null)).toContain("no defined recovery");
    expect(recoveryPercent(0.9148)).toBe("91.5%");
    expect(recoveryPercent(null)).toBe("n/a");
  });

  it("builds a headline with the word and the block", () => {
    expect(headline("blocks.5.hook_out", " John", 0.915)).toBe(
      "Patching block 5 (residual stream) at “John” brings back most of the answer.",
    );
  });
});
