"""Static configuration: resource limits and the verified hook allowlist.

Every hook name in SUPPORTED_HOOKS was verified against the installed
TransformerLens 4.x TransformerBridge on 2026-10-30: it is present in
``bridge.hook_dict`` AND captured by ``run_with_cache``. Note that
``blocks.{i}.attn.hook_result`` exists in hook_dict but is NOT captured by
run_with_cache in this version, so it is excluded.
"""

MAX_PROMPT_CHARS = 512
MAX_TOP_K = 20
MAX_CAPTURE_HOOKS = 12
MAX_CAPTURE_TOKENS = 512  # activation captures bounded to short sequences
MAX_HEADS = 12
MAX_LAYERS = 12
ONE_HEAVY_JOB = True
DB_PATH_DEFAULT = "outputs/experiments.db"
MODEL_ID = "gpt2"

# Hooks whose output can be replaced during a patched run.
# Shapes (batch=1):
#   blocks.{i}.hook_out            [1, seq, 768]        residual stream after block i
#   blocks.{i}.attn.hook_out       [1, seq, 768]        attention sublayer output
#   blocks.{i}.attn.hook_z         [1, seq, 12, 64]     per-head pre-o-projection values
#   blocks.{i}.mlp.hook_out        [1, seq, 768]        MLP sublayer output
PATCHABLE_HOOKS = tuple(
    h.format(i=i)
    for i in range(MAX_LAYERS)
    for h in (
        "blocks.{i}.hook_out",
        "blocks.{i}.attn.hook_out",
        "blocks.{i}.attn.hook_z",
        "blocks.{i}.mlp.hook_out",
    )
)

# Hooks that can be read (captured) for inspection; includes patchables plus
# structural measurements.
CAPTURABLE_HOOKS = set(PATCHABLE_HOOKS) | {"hook_embed", "hook_pos_embed", "ln_final.hook_normalized"} | {
    f"blocks.{i}.attn.hook_pattern" for i in range(MAX_LAYERS)
}

# Prepared IOI example: token positions in the clean prompt
# "John and Mary went to the shop. John gave a book to"
# index 0 is <|endoftext|> (BOS).
S2_POSITION = 9  # second subject " John"
END_POSITION = 13  # final token " to" (where next-token logits are read)


def validate_hook_name(name: str, *, patchable: bool) -> bool:
    if patchable:
        return name in PATCHABLE_HOOKS
    return name in CAPTURABLE_HOOKS
