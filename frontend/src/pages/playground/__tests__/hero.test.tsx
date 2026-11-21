import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import Hero from "../Hero";
import { triedKey } from "../insights";

afterEach(cleanup);

const TOKENS = ["<|endoftext|>", "John", " and", " Mary", " went"];
const RESULT = {
  config: { clean_text: "", corrupted_text: "", hook: "blocks.5.hook_out", positions: [1], heads: null, candidate_a: " Mary", candidate_b: " John", top_k: 8 },
  provenance: { model_id: "gpt2", library: "x", device: "cpu", torch: "x", elapsed_seconds: 1 },
  tokens_clean: TOKENS,
  tokens_corrupted: ["<|endoftext|>", "Mary", " and", " Mary", " went"],
  runs: {
    clean: [{ token: " Mary", token_id: 1, logit: 0, prob: 0.66 }],
    corrupted: [{ token: " John", token_id: 2, logit: 0, prob: 0.74 }, { token: " Mary", token_id: 1, logit: 0, prob: 0.1 }],
    patched: [{ token: " Mary", token_id: 1, logit: 0, prob: 0.61 }],
  },
  metrics: { D_clean: 2.5, D_corrupted: -4.2, D_patched: 1.94, Delta: 6.1, Recovery: 0.915, sign_convention: "" },
  patch: { hook: "blocks.5.hook_out", positions: [1], heads: null, source: "" },
};

function setup(over: Partial<React.ComponentProps<typeof Hero>> = {}) {
  const setPosition = vi.fn();
  const setHook = vi.fn();
  render(
    <Hero
      prompt="John and Mary went"
      tokens={TOKENS}
      position={1}
      setPosition={setPosition}
      hook="blocks.5.hook_out"
      setHook={setHook}
      tried={{}}
      runState="idle"
      result={null}
      {...over}
    />,
  );
  return { setPosition, setHook };
}

describe("Playground hero", () => {
  it("clicking a word chooses where to patch, clicking a block chooses which", () => {
    const { setPosition, setHook } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Patch at position 3: Mary" }));
    expect(setPosition).toHaveBeenCalledWith(3);
    fireEvent.click(screen.getByRole("button", { name: "Patch block 7" }));
    expect(setHook).toHaveBeenCalledWith("blocks.7.hook_out");
    expect(screen.getByRole("button", { name: "Patch block 5" }).getAttribute("aria-pressed")).toBe("true");
  });

  it("hides the start-of-text token and shows a prompt to run before there is a result", () => {
    setup();
    expect(screen.queryByText("<|endoftext|>")).toBeNull();
    expect(screen.getByText(/Press Run experiment/)).toBeTruthy();
  });

  it("shades only blocks that were really run, with their measured percentage", () => {
    setup({ tried: { [triedKey("blocks.5.hook_out", 1)]: 0.915 } });
    expect(screen.getByRole("button", { name: "Patch block 5" }).textContent).toContain("92");
    expect(screen.getByRole("button", { name: "Patch block 4" }).textContent).toBe("4");
  });

  it("with a result, reads the measured recovery and shows before and after", () => {
    setup({ runState: "complete", result: RESULT });
    expect(screen.getByRole("img", { name: "Recovery 91.5%" })).toBeTruthy();
    expect(screen.getByText(/brings back most of the answer/)).toBeTruthy();
    expect(screen.getByText("Before: corrupted prompt")).toBeTruthy();
    expect(screen.getByText("After: patched")).toBeTruthy();
    // the word that differs between the two prompts is marked
    expect(screen.getByRole("button", { name: "Patch at position 1: John" }).className).toContain("swap");
  });
});
