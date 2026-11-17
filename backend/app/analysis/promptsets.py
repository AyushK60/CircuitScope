"""Stage 3 Task 1: deterministic IOI prompt-set generator.

``generate(n, seed)`` builds n token-aligned IOI prompts from three fixed
templates. Every word is filtered through the REAL GPT-2 tokenizer on first
use: a word survives only if `` word`` (leading space) is exactly one token.
All three corruptions of each prompt (``swap``, ``third``, ``resample``) are
asserted token-aligned with the clean prompt; misaligned draws are rejected and
counted, never silently accepted.

Deterministic given (n, seed) via ``random.Random(seed)`` — no global state,
no numpy RNG here.
"""

from __future__ import annotations

import random

from ..model_runtime.loader import ADAPTER

MAX_DRAWS_PER_PROMPT = 1000  # safety guard against a pathological filter pass

# Word pools in code (supersets; filtered through the tokenizer at first use).
NAME_POOL = [
    "John", "Mary", "Tom", "Anna", "James", "Sarah", "David", "Emma",
    "Peter", "Laura", "Michael", "Kate", "Paul", "Alice", "Robert", "Grace",
    "Simon", "Clara", "Henry", "Jane", "Ben", "Nora", "Adam", "Elsie",
    "Victor", "Iris", "Martin", "Faye",
]
PLACE_POOL = [
    "shop", "park", "cafe", "library", "garden", "market", "station",
    "office", "school", "museum", "theatre", "bakery", "harbour", "studio",
]
OBJ_POOL = [
    "book", "ball", "gift", "flower", "letter", "toy", "coat", "camera",
    "guitar", "picture", "blanket", "teapot", "pen", "shell",
]

TEMPLATES = {
    1: "{A} and {B} went to the {place}. {S2} gave a {obj} to",
    2: "Then, {A} and {B} went to the {place}. {S2} gave a {obj} to",
    3: "After lunch, {A} and {B} had a long talk at the {place}. {S2} gave a {obj} to",
}

_filtered_cache: dict[str, list[str]] | None = None


def _single_token_words() -> dict[str, list[str]]:
    """Filter every pool word through the real tokenizer (cached at first use).

    A word is kept only if `` word`` (leading space) is exactly one GPT-2 token.
    Names must ALSO be one token without the leading space, because the first
    subject stands at the start of the sentence: e.g. ``Iris`` is single-token
    as `` Iris`` but multi-token as ``Iris``, which would break the
    prompt-for-prompt token alignment that ``resample`` depends on.
    """
    global _filtered_cache
    if _filtered_cache is not None:
        return _filtered_cache
    kept: dict[str, list[str]] = {"names": [], "places": [], "objs": []}
    for word in NAME_POOL:
        try:
            ADAPTER.token_id_for_str(f" {word}")
            ADAPTER.token_id_for_str(word)
        except ValueError:
            continue
        kept["names"].append(word)
    for word in PLACE_POOL:
        try:
            ADAPTER.token_id_for_str(f" {word}")
        except ValueError:
            continue
        kept["places"].append(word)
    for word in OBJ_POOL:
        try:
            ADAPTER.token_id_for_str(f" {word}")
        except ValueError:
            continue
        kept["objs"].append(word)
    _filtered_cache = kept
    return kept


def _s2_position(tokens: list[str], s2_name: str) -> int:
    """Token index (BOS included) of the second-subject name.

    In every template the S2 token is followed by exactly 4 tokens
    (" gave", " a", obj, " to"), so its position is n_tokens - 5; the
    returned index must also carry the S2 name itself.
    """
    idx = len(tokens) - 5
    # Token strings carry their leading space (" Jane"); compare stripped.
    if idx < 0 or tokens[idx].strip() != s2_name:
        raise ValueError(
            f"could not locate second subject {s2_name!r} at position {idx} in {tokens}"
        )
    return idx


def _token_count(text: str) -> int:
    bridge = ADAPTER.bridge()
    return int(bridge.to_tokens(text).shape[1])


def _draw(rng: random.Random, words: dict[str, list[str]]) -> dict:
    names = rng.sample(words["names"], 3)
    return {
        "A": names[0],
        "B": names[1],
        "C": names[2],
        "place": rng.choice(words["places"]),
        "obj": rng.choice(words["objs"]),
    }


def generate(n: int, seed: int) -> dict:
    """Generate n IOI prompts with their three corruptions.

    Returns {"prompts": [...], "rejected_draws": int, "template_counts": {...}}.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    words = _single_token_words()
    if len(words["names"]) < 3 or not words["places"] or not words["objs"]:
        raise ValueError(
            f"tokenizer filtering left too few words: {len(words['names'])} names, "
            f"{len(words['places'])} places, {len(words['objs'])} objects"
        )
    rng = random.Random(seed)
    bridge = ADAPTER.bridge()

    prompts: list[dict] = []
    rejected_draws = 0

    # Template x S2-role schedule: template cycles 1,2,3; the S2 role (which of
    # A/B is the repeated subject) alternates every full template cycle, so each
    # template x role cell appears equally often as n allows.
    schedule = []
    i = 0
    while len(schedule) < n:
        for t in (1, 2, 3):
            if len(schedule) >= n:
                break
            schedule.append((t, "A" if (i // 3) % 2 == 0 else "B"))
            i += 1

    # First pass: draw prompts (no corruptions yet) — resample needs the full set.
    for idx, (template_id, s2_role) in enumerate(schedule):
        for attempt in range(MAX_DRAWS_PER_PROMPT):
            d = _draw(rng, words)
            A, B, C = d["A"], d["B"], d["C"]
            # Names must be pairwise distinct within a prompt.
            if not (A != B and A != C and B != C):
                rejected_draws += 1
                continue
            s2, io = (A, B) if s2_role == "A" else (B, A)
            clean_text = TEMPLATES[template_id].format(A=A, B=B, place=d["place"], S2=s2, obj=d["obj"])
            tokens = bridge.to_str_tokens(clean_text)
            try:
                s2_pos = _s2_position(tokens, s2)
            except ValueError:
                rejected_draws += 1
                continue
            prompts.append(
                {
                    "id": f"p{idx:03d}",
                    "template_id": template_id,
                    "names": {"A": A, "B": B, "C": C},
                    "place": d["place"],
                    "obj": d["obj"],
                    "s2_role": s2_role,
                    "clean_text": clean_text,
                    "candidate_a": f" {io}",  # IO name (the answer)
                    "candidate_b": f" {s2}",  # repeated subject S
                    "s2_position": s2_pos,
                    "end_position": len(tokens) - 1,
                    "n_tokens": len(tokens),
                }
            )
            break
        else:
            raise RuntimeError(
                f"could not draw a valid prompt after {MAX_DRAWS_PER_PROMPT} attempts"
            )

    # Second pass: build the three corruptions and assert token alignment.
    for prompt in prompts:
        A, B, C = prompt["names"]["A"], prompt["names"]["B"], prompt["names"]["C"]
        s2 = prompt["candidate_b"].strip()
        io = prompt["candidate_a"].strip()
        swap_s2, third_s2 = io, C  # swap: S2 replaced by the other name; third: by C
        swap_text = TEMPLATES[prompt["template_id"]].format(
            A=A, B=B, place=prompt["place"], S2=swap_s2, obj=prompt["obj"]
        )
        third_text = TEMPLATES[prompt["template_id"]].format(
            A=A, B=B, place=prompt["place"], S2=third_s2, obj=prompt["obj"]
        )
        corruptions: dict[str, str | None] = {"swap": swap_text, "third": third_text, "resample": None}
        for kind in ("swap", "third"):
            if _token_count(corruptions[kind]) != prompt["n_tokens"]:  # type: ignore[arg-type]
                rejected_draws += 1
                corruptions[kind] = None  # never silently accept a misaligned pair
        prompt["corruptions"] = corruptions

    # resample: the NEXT prompt of the same template in the generated set (cyclic).
    for i, prompt in enumerate(prompts):
        for j in range(1, len(prompts) + 1):
            other = prompts[(i + j) % len(prompts)]
            if other["template_id"] == prompt["template_id"]:
                prompt["corruptions"]["resample"] = other["clean_text"]
                break
        if prompt["corruptions"]["resample"] is None:
            # Cannot happen while len(prompts) >= 1 (the loop always finds the
            # prompt itself as a last resort), but never leave it unset.
            prompt["corruptions"]["resample"] = prompt["clean_text"]
        if _token_count(prompt["corruptions"]["resample"]) != prompt["n_tokens"]:
            rejected_draws += 1
            prompt["corruptions"]["resample"] = None

    template_counts: dict[str, int] = {}
    for t in (1, 2, 3):
        template_counts[f"template_{t}"] = sum(1 for p in prompts if p["template_id"] == t)
    s2_role_counts = {r: sum(1 for p in prompts if p["s2_role"] == r) for r in ("A", "B")}

    return {
        "prompts": prompts,
        "rejected_draws": rejected_draws,
        "template_counts": template_counts,
        "s2_role_counts": s2_role_counts,
    }
