import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import GuidedTour from "../GuidedTour";
import type { FlowData, ModelMeta } from "../../../api/client";

afterEach(cleanup);

const meta = {
  model_id: "gpt2",
  revision: "gpt2",
  device: "cpu",
  dtype: "float32",
  n_layers: 12,
  n_heads: 12,
  d_model: 768,
  d_head: 64,
  d_mlp: 3072,
  n_ctx: 1024,
  d_vocab: 50257,
  n_params_approx: 124,
  tokenizer_class: "GPT2TokenizerFast",
} as ModelMeta;

function flow(): FlowData {
  const tokens = ["<|endoftext|>", "The", " cat", " sat", " on"];
  const n = tokens.length;
  // each head: row q puts 80% on token 0 (the resting place), the rest on token q-1 (or 0)
  const head = Array.from({ length: n }, (_, q) =>
    Array.from({ length: n }, (_, k) => (k > q ? 0 : k === 0 ? (q === 0 ? 1 : 0.8) : k === q - 1 ? 0.2 : 0)),
  );
  return {
    hook_basis: "test",
    tokens,
    token_ids: tokens.map((_, i) => i),
    n_layers: 12,
    blocks: Array.from({ length: 12 }, (_, layer) => ({
      layer,
      resid_norm_per_token: [3000, 70, 80, 75, 72],
      resid_norm_mean: 100 + layer * 10,
      attention: Array.from({ length: 12 }, () => head),
    })),
    top_predictions: [
      { token: " the", token_id: 1, logit: 5, prob: 0.35 },
      { token: " a", token_id: 2, logit: 4, prob: 0.12 },
    ],
  };
}

describe("guided tour", () => {
  it("shows a skeleton until the run is captured", () => {
    const { container } = render(<GuidedTour flow={null} meta={meta} />);
    expect(container.querySelector(".skeleton")).not.toBeNull();
  });

  it("walks the stages: tokens, embeddings, a block, the answer", () => {
    render(<GuidedTour flow={flow()} meta={meta} />);
    expect(screen.getByRole("heading", { name: "Your sentence becomes tokens" })).toBeInTheDocument();
    expect(screen.getByText(/Stage 1 of 16/, { selector: ".ex-where" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Next stage" }));
    expect(screen.getByRole("heading", { name: "Each token becomes a list of numbers" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Block 3" }));
    expect(screen.getByRole("heading", { name: "Block 3: read, think, write back" })).toBeInTheDocument();
    // measured notice for the default head: 80% of attention on the start-of-text token
    expect(screen.getByText(/puts 80% of its attention/)).toBeInTheDocument();
    expect(screen.getAllByText(/blocks\.3\.attn\.hook_pattern/).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("tab", { name: "Answer" }));
    expect(screen.getByRole("heading", { name: "The answer: a score for every word" })).toBeInTheDocument();
    expect(screen.getByText("35.0%")).toBeInTheDocument();
  });

  it("picking another head keeps the block and updates the explanation", () => {
    render(<GuidedTour flow={flow()} meta={meta} />);
    fireEvent.click(screen.getByRole("tab", { name: "Block 0" }));
    fireEvent.click(screen.getByRole("button", { name: "Head 2" }));
    expect(screen.getByText(/Head 2, standing on the last word/)).toBeInTheDocument();
  });

  it("arrow keys move between stages", () => {
    render(<GuidedTour flow={flow()} meta={meta} />);
    const rail = screen.getByRole("tablist", { name: "Stages of one forward pass" });
    fireEvent.keyDown(rail, { key: "ArrowRight" });
    fireEvent.keyDown(rail, { key: "ArrowRight" });
    expect(screen.getByRole("heading", { name: "Block 0: read, think, write back" })).toBeInTheDocument();
    fireEvent.keyDown(rail, { key: "ArrowLeft" });
    expect(screen.getByRole("heading", { name: "Each token becomes a list of numbers" })).toBeInTheDocument();
  });
});
