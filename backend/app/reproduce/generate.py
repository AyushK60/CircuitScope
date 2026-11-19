"""Build reproduction-script data and text.

Every script is plain text. User text only ever appears inside one JSON literal
(``json.dumps`` + ``repr``), never as Python code, a path, or an argument.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from .runtime import BODY

KINDS = ("intervention", "sweep", "study_subset", "report", "dla", "lens")
SAFE_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")

TOLERANCES = {
    "d": 5e-3,
    "recovery": 5e-3,
    "dla_term": 2e-3,
    "lens_d": 2e-3,
    "lens_identity": 1e-3,
    "atp": 2e-3,
}

DESCRIPTIONS = {
    "intervention": (
        "the clean, corrupted and patched score gaps and the recovery of the "
        "stored activation patch"
    ),
    "sweep": "the stored patching-sweep cells (Recovery and D_patched) for one sweep mode",
    "dla": "all 171 direct-logit-attribution terms and the reconciliation identity",
    "lens": "every logit-lens stage's score gap and the final-stage identity",
    "study_subset": (
        "exact denoising Rec_D for all 156 components and attribution patching "
        "(Rec_AtP) for the first study prompts"
    ),
    "report": (
        "the report's DLA terms, lens stages, layer x position sweep and layer x "
        "head sweep for the report's pair"
    ),
}

RUNTIMES = {
    "intervention": ("about 10 seconds", "about 10 seconds (one patch)"),
    "sweep": ("about 1 minute for the fixed subset", "about 3-6 minutes for every cell"),
    "dla": ("about 10 seconds (one forward pass)", "about 10 seconds (one forward pass)"),
    "lens": ("about 10 seconds (one forward pass)", "about 10 seconds (one forward pass)"),
    "study_subset": ("about 1-2 minutes for the first 2 prompts", "about 1 minute per prompt"),
    "report": ("about 2-4 minutes for the fixed cells", "about 10-20 minutes for every cell"),
}


def safe_id(raw: str | None) -> str:
    """A filename-safe id. Raw user text is hashed, never used as a path."""
    text = str(raw or "live")
    if SAFE_ID_RE.fullmatch(text):
        return text
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def filename(kind: str, raw_id: str | None) -> str:
    return f"circuitscope_repro_{kind}_{safe_id(raw_id)}.py"


def _provenance_defaults(provenance: dict | None) -> dict:
    prov = provenance or {}
    versions = prov.get("versions") or {}
    return {
        "model_revision": prov.get("model_revision", "unknown"),
        "versions": {
            "torch": versions.get("torch", "unknown"),
            "transformers": versions.get("transformers", "unknown"),
        },
    }


def _header(kind: str, data: dict, requested_mode: str) -> str:
    prov = _provenance_defaults(
        {
            "model_revision": data.get("model_revision"),
            "versions": data.get("versions"),
        }
    )
    sample_rt, full_rt = RUNTIMES[kind]
    lines = [
        f"# CircuitScope reproduction script - {kind}",
        "#",
        f"# Reproduces: {DESCRIPTIONS[kind]}.",
        "# It checks reproduction of the numbers by independent code (raw",
        "# HuggingFace transformers forward hooks, no TransformerLens, no backend",
        "# imports). It does NOT check the scientific interpretation.",
        "#",
        "# Install: pip install torch transformers",
        f"# Expected runtime: --mode sample {sample_rt}; --mode full {full_rt}.",
        f"# Stored model revision: {prov['model_revision']}",
        f"# Stored library versions: torch {prov['versions']['torch']}, "
        f"transformers {prov['versions']['transformers']}",
        f"# Tolerances: {json.dumps(data['tolerances'], sort_keys=True)}",
        f"# Embedded default mode: {requested_mode}",
        "#",
        "# All inputs are embedded below as ONE JSON literal loaded with",
        "# json.loads. No user text is interpolated into this code.",
        "#",
        "# usage: python <this file> [--mode sample|full] [--n-prompts N]",
        "",
        "",
    ]
    return "\n".join(lines)


def generate(kind: str, data: dict, requested_mode: str = "sample") -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown reproduce kind: {kind!r}")
    payload = dict(data)
    payload["tolerances"] = dict(TOLERANCES)
    payload["mode_default"] = requested_mode
    if kind == "report":
        # The combined script calls the same runners on nested data, which each
        # read their own tolerances; give every nested block the same table.
        for key in ("dla", "lens", "position_sweep", "head_sweep"):
            if isinstance(payload.get(key), dict):
                payload[key] = {**payload[key], "tolerances": dict(TOLERANCES)}
    literal = repr(json.dumps(payload, ensure_ascii=True, allow_nan=False))
    body = BODY.replace("__DATA_LITERAL__", literal).replace("__KIND_LITERAL__", repr(kind))
    return _header(kind, payload, requested_mode) + body


# ---------------------------------------------------------------------------
# data builders
# ---------------------------------------------------------------------------


def _sweep_cells(recovery: list[list[float]], mode: str) -> list[list[int]]:
    """The documented fixed subset: 5 largest-|value| cells, 5 fixed-seed
    cells, and the invariant cells (last block's non-final positions for
    layer x position; up to 3 stored near-zero cells for layer x head)."""
    all_cells = [
        [layer, col]
        for layer, row in enumerate(recovery)
        for col in range(len(row))
    ]
    if not all_cells:
        return []
    ranked = sorted(all_cells, key=lambda c: (-abs(recovery[c[0]][c[1]]), c[0], c[1]))
    chosen: list[list[int]] = ranked[:5]
    rng = random.Random(12345)
    for cell in rng.sample(all_cells, min(5, len(all_cells))):
        if cell not in chosen:
            chosen.append(cell)
    if mode == "layer_position":
        last = len(recovery) - 1
        for col in range(len(recovery[last]) - 1):
            cell = [last, col]
            if cell not in chosen:
                chosen.append(cell)
    else:
        near_zero = sorted(
            all_cells, key=lambda c: (abs(recovery[c[0]][c[1]]), c[0], c[1])
        )[:3]
        for cell in near_zero:
            if cell not in chosen:
                chosen.append(cell)
    return chosen


def sweep_data(
    *,
    mode: str,
    clean_text: str,
    corrupted_text: str,
    candidate_a: str,
    candidate_b: str,
    baselines: dict,
    recovery: list[list[float]],
    d_patched: list[list[float]],
    provenance: dict | None,
) -> dict:
    return {
        "mode": mode,
        "clean_text": clean_text,
        "corrupted_text": corrupted_text,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "stored": {
            "baselines": {
                "D_clean": baselines["D_clean"],
                "D_corrupted": baselines["D_corrupted"],
            },
            "recovery": recovery,
            "d_patched": d_patched,
        },
        "cells_sample": _sweep_cells(recovery, mode),
        **_provenance_defaults(provenance),
    }


def intervention_data(config: dict, results: dict) -> dict:
    patch = results.get("patch") or config
    heads = patch.get("heads")
    data = {
        "clean_text": config["clean_text"],
        "corrupted_text": config["corrupted_text"],
        "candidate_a": config["candidate_a"],
        "candidate_b": config["candidate_b"],
        "hook": patch.get("hook", config.get("hook")),
        "positions": list(patch.get("positions") or config.get("positions") or []),
        "heads": list(heads) if heads is not None else list(range(12)),
        "stored": {"metrics": results["metrics"]},
        **_provenance_defaults(results.get("provenance")),
    }
    return data


def dla_data(
    *,
    text: str,
    candidate_a: str,
    candidate_b: str,
    dla_result: dict,
    provenance: dict | None,
) -> dict:
    return {
        "text": text,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "stored": {
            "D": dla_result["D"],
            "components": {c["name"]: c["value"] for c in dla_result["components"]},
            "reconciliation": {
                "abs_error": dla_result["reconciliation"]["abs_error"],
            },
        },
        **_provenance_defaults(provenance),
    }


def lens_data(
    *,
    text: str,
    candidate_a: str | None,
    candidate_b: str | None,
    lens_result: dict,
    provenance: dict | None,
) -> dict:
    stages = []
    for s in lens_result["stages"]:
        entry: dict[str, Any] = {"stage": s["stage"], "D": s.get("D")}
        top = s.get("top") or []
        entry["top1_id"] = top[0]["token_id"] if top else None
        stages.append(entry)
    return {
        "text": text,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "stored": {
            "stages": stages,
            "final_D": lens_result.get("model_D"),
            "final_matches_model": lens_result.get("final_matches_model") or {},
        },
        **_provenance_defaults(provenance),
    }


def study_subset_data(results: dict) -> dict:
    prompts = results.get("prompt_set", {}).get("prompts") or []
    per_prompt = results.get("per_prompt") or []
    records = []
    for p, pp in zip(prompts, per_prompt):
        swap = (p.get("corruptions") or {}).get("swap")
        rec = (pp.get("rec_d") or {}).get("swap")
        atp = pp.get("atp")
        if not swap or rec is None or atp is None:
            continue
        if any(v is None for v in rec) or any(v is None for v in atp):
            continue
        records.append(
            {
                "id": p["id"],
                "clean_text": p["clean_text"],
                "corrupted_text": swap,
                "candidate_a": p["candidate_a"],
                "candidate_b": p["candidate_b"],
                "rec_d": list(rec),
                "atp": list(atp),
            }
        )
    return {
        "prompts": records,
        "n_prompts_default": 2,
        **_provenance_defaults(results.get("provenance")),
    }


def report_data(results: dict, config: dict) -> dict:
    sections = {s["id"]: s for s in results.get("sections") or []}
    need = ("where_formed", "which_words", "which_parts")
    missing = [k for k in need if k not in sections]
    if missing:
        raise ValueError(f"report is missing sections needed for reproduction: {missing}")
    head_sweep = sections["which_parts"]["data"]["sweep"]
    pos_sweep = sections["which_words"]["data"]["sweep"]
    dla = sections["which_parts"]["data"].get("dla") or {}
    components = dla.get("components")
    if not components:
        raise ValueError(
            "this saved report predates the reproduce data; generate a new report first"
        )
    if isinstance(components, dict):
        comp_map = components
    else:
        comp_map = {c["name"]: c["value"] for c in components}
    lens = sections["where_formed"]["data"]["lens"]
    clean = config["clean_text"]
    cand_a, cand_b = config["candidate_a"], config["candidate_b"]
    prov = results.get("provenance")

    def _sweep(sweep: dict) -> dict:
        return sweep_data(
            mode=sweep["mode"],
            clean_text=clean,
            corrupted_text=config["corrupted_text"],
            candidate_a=cand_a,
            candidate_b=cand_b,
            baselines=sweep["baselines"],
            recovery=sweep["recovery"],
            d_patched=sweep["d_patched"],
            provenance=prov,
        )

    return {
        "dla": {
            "text": clean,
            "candidate_a": cand_a,
            "candidate_b": cand_b,
            "stored": {
                "D": dla.get("D"),
                "components": comp_map,
                "reconciliation": {
                    "abs_error": (dla.get("reconciliation") or {}).get("abs_error"),
                },
            },
            **_provenance_defaults(prov),
        },
        "lens": {
            "text": clean,
            "candidate_a": cand_a,
            "candidate_b": cand_b,
            "stored": {
                "stages": [
                    {"stage": s["stage"], "D": s.get("D")} for s in lens.get("stages") or []
                ],
                "final_D": lens.get("final_d"),
                "final_matches_model": lens.get("final_check") or {},
            },
            **_provenance_defaults(prov),
        },
        "position_sweep": _sweep(pos_sweep),
        "head_sweep": _sweep(head_sweep),
        **_provenance_defaults(prov),
    }
