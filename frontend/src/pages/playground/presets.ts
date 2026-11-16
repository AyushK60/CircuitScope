export const PRESET = {
  id: "ioi_john_mary",
  name: "IOI · John and Mary",
  clean: "John and Mary went to the shop. John gave a book to",
  corrupted: "John and Mary went to the shop. Mary gave a book to",
  candidate_a: " Mary",
  candidate_b: " John",
  s2_position: 9,
};

/** Patchable hooks offered — all verified in the backend allowlist
 *  (PATCHABLE_HOOKS). Generated so the UI can never desync from state. */
export const HOOK_OPTIONS = [
  ...Array.from({ length: 12 }, (_, l) => ({
    value: `blocks.${l}.hook_out`,
    label: `Block ${l} · residual stream${l === 5 ? " (recommended)" : ""}`,
  })),
  ...Array.from({ length: 12 }, (_, l) => ({
    value: `blocks.${l}.attn.hook_out`,
    label: `Block ${l} · attention output`,
  })),
  { value: "blocks.5.attn.hook_z", label: "Block 5 · attention z (per-head)" },
  ...[5, 9].map((l) => ({
    value: `blocks.${l}.mlp.hook_out`,
    label: `Block ${l} · MLP output`,
  })),
];
