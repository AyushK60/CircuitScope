import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import TopPredictions from "../TopPredictions";
import TokenRow from "../TokenRow";
import AttentionHeatmap from "../AttentionHeatmap";
import ArchitectureCanvas from "../ArchitectureCanvas";
import type { AttentionPattern, Candidate, ModelMeta } from "../../api/client";

afterEach(cleanup);

const candidates: Candidate[] = [
  { token: " Mary", token_id: 5335, logit: 20.1, prob: 0.662 },
  { token: " John", token_id: 1757, logit: 17.6, prob: 0.054 },
];

const meta: ModelMeta = {
  model_id: "gpt2",
  revision: "gpt2",
  device: "cpu",
  dtype: "torch.float32",
  n_layers: 12,
  n_heads: 12,
  d_model: 768,
  d_head: 64,
  d_mlp: 3072,
  n_ctx: 1024,
  d_vocab: 50257,
  n_params_approx: 124.4,
  tokenizer_class: "GPT2Tokenizer",
};

describe("TopPredictions", () => {
  it("renders measured candidates with probability scale", () => {
    render(<TopPredictions candidates={candidates} metric="prob" />);
    expect(screen.getByText("66.20%")).toBeTruthy();
    expect(screen.getByText("Mary")).toBeTruthy();
  });
  it("shows skeleton when no data", () => {
    const { container } = render(<TopPredictions candidates={null} />);
    expect(container.querySelector(".skeleton")).toBeTruthy();
  });
});

describe("TokenRow", () => {
  it("renders exact tokens with position indices", () => {
    render(
      <TokenRow
        tok={{ tokens: ["<|endoftext|>", "Hello", " world"], token_ids: [50256, 15496, 995], n_tokens: 3 }}
      />,
    );
    expect(screen.getByRole("list", { name: "Tokenized input" }).children.length).toBe(3);
    const items = screen.getAllByRole("listitem");
    expect(items[2].textContent).toContain(" world");
    expect(items[2].textContent).toContain("2");
    expect(items[2].getAttribute("title")).toContain("id 995");
  });
  it("highlights the patched position", () => {
    render(
      <TokenRow
        tok={{ tokens: [" a", " b"], token_ids: [1, 2], n_tokens: 2 }}
        highlightPositions={[1]}
      />,
    );
    expect(screen.getAllByRole("listitem")[1].className).toContain("hl");
  });
});

describe("ArchitectureCanvas", () => {
  it("renders one block per real model layer, in correct order", () => {
    render(
      <ArchitectureCanvas
        meta={meta}
        flow={null}
        level="model"
        layer={5}
        keyboardFocusLayer={5}
        onLevel={() => {}}
        onLayer={() => {}}
        onHead={() => {}}
      />,
    );
    for (let l = 0; l < 12; l++) {
      expect(screen.getByRole("button", { name: `Open transformer block ${l}` })).toBeTruthy();
    }
  });
});

describe("AttentionHeatmap", () => {
  it("renders the real pattern matrix with hook provenance", () => {
    const att: AttentionPattern = {
      hook: "blocks.5.attn.hook_pattern",
      shape: [2, 3, 3],
      tokens: ["<|endoftext|>", "Hello", " world"],
      pattern: [
        [
          [1.0, 0.0, 0.0],
          [0.5, 0.5, 0.0],
          [0.2, 0.3, 0.5],
        ],
        [
          [0.9, 0.1, 0.0],
          [0.4, 0.4, 0.2],
          [0.1, 0.1, 0.8],
        ],
      ],
    };
    render(<AttentionHeatmap att={att} head={1} />);
    const svg = screen.getByRole("img");
    expect(svg).toBeTruthy();
    // 9 cells for head 1
    expect(svg.querySelectorAll("rect").length).toBe(9);
    expect(screen.getByText(/blocks\.5\.attn\.hook_pattern/)).toBeTruthy();
  });
});
