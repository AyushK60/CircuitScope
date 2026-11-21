import { describe, it, expect, afterEach, vi, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import HeroSentence from "../HeroSentence";
import { api, type FlowData } from "../../../api/client";

vi.mock("../../../api/client", async (orig) => {
  const mod = await orig<typeof import("../../../api/client")>();
  return { ...mod, api: { ...mod.api, flow: vi.fn(), predict: vi.fn() } };
});

const cand = (token: string, prob: number) => ({ token, token_id: token.length, logit: 0, prob });

function flowData(): FlowData {
  const tokens = ["<|endoftext|>", "John", " and", " Mary", " to"];
  const n = tokens.length;
  // every head puts all its attention on token 3 (" Mary") from the last position
  const head = Array.from({ length: n }, (_, q) =>
    Array.from({ length: n }, (_, k) => (q === n - 1 && k === 3 ? 1 : 0)),
  );
  return {
    hook_basis: "test",
    tokens,
    token_ids: tokens.map((_, i) => i),
    n_layers: 12,
    blocks: Array.from({ length: 12 }, (_, layer) => ({
      layer,
      resid_norm_per_token: [],
      resid_norm_mean: 10 + layer,
      attention: [head, head],
    })),
    top_predictions: [cand(" Mary", 0.662), cand(" John", 0.054)],
  };
}

/* jsdom has no layout: give every element its own horizontal slot so the arcs have anchors */
const slots = new WeakMap<Element, number>();
beforeEach(() => {
  vi.clearAllMocks();
  let next = 0;
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
    if (!slots.has(this)) slots.set(this, next++ * 60);
    const left = slots.get(this)!;
    return { left, right: left + 40, width: 40, top: 0, bottom: 20, height: 20, x: left, y: 0, toJSON: () => ({}) };
  });
});
afterEach(cleanup);

describe("landing hero", () => {
  it("offline: shows the labelled reference result and keeps words inert", () => {
    render(<HeroSentence live={false} navigate={() => {}} />);
    expect(screen.getByText("REFERENCE")).toBeInTheDocument();
    expect(screen.getByText("66.2%")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Hide the word/ })).toBeNull();
    expect(api.flow).not.toHaveBeenCalled();
  });

  it("live: shows measured data and re-runs the model without a clicked word", async () => {
    vi.mocked(api.flow).mockResolvedValue({ model: {} as never, ...flowData() });
    vi.mocked(api.predict).mockResolvedValue({
      model: {} as never,
      tokenization: { tokens: [], token_ids: [], n_tokens: 0 },
      prediction: {
        tokens: [],
        token_ids: [],
        top_predictions: [cand(" John", 0.4), cand(" Mary", 0.2)],
        elapsed_seconds: 0.1,
        device: "cpu",
      },
    });
    render(<HeroSentence live navigate={() => {}} />);
    await screen.findByText("MEASURED · LIVE");
    expect(screen.getByText("66.2%")).toBeInTheDocument();
    const mary = screen.getByRole("button", { name: "Hide the word Mary and re-run the model" });
    expect(mary).toHaveClass("best");

    fireEvent.click(mary);
    await waitFor(() => expect(api.predict).toHaveBeenCalledWith("John and to", 5));
    expect(await screen.findByText(/Without “Mary” the model's top word is “John”/)).toBeInTheDocument();
    expect(screen.getByText(/The answer changed\./)).toBeInTheDocument();
  });
});
