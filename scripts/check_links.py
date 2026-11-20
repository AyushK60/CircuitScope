"""Check that every relative Markdown link resolves to a file.

Scope: README.md at the repo root and every docs/**/*.md (recursive). For each
file, every inline link or reference definition of the form  [text](target)  /
![alt](target)  /  [ref]: target  is resolved relative to the file's own
directory. Absolute URLs (http/https/mailto), pure anchors (#fragment) and
empty targets are ignored; a #fragment suffix on a file link is stripped.
Links to known generated locations (outputs/, release/) are also ignored:
those directories are git-ignored and do not exist in a fresh clone.

Exits 0 when every checked link resolves, otherwise prints each broken link
and exits 1. Standard library only.

Usage: .venv\\Scripts\\python scripts\\check_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]

# Generated, git-ignored directories: links into them are allowed to dangle in
# a fresh clone, so they are not checked.
IGNORED_PREFIXES = ("outputs/", "outputs\\", "release/", "release\\")

INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*([^)]+?)\s*\)")
REFERENCE_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+(\S+)\s*$", re.MULTILINE)


def _targets(text: str) -> list[str]:
    found = [m.group(1) for m in INLINE_LINK.finditer(text)]
    found += REFERENCE_DEF.findall(text)
    return found


def broken_links_in(path: Path) -> list[str]:
    """Return the broken link targets (with context) for one Markdown file."""
    problems: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in _targets(text):
        target = raw.strip().strip("<>").strip()
        if not target or target.startswith("#"):
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):  # http:, https:, mailto:, ...
            continue
        target = unquote(target)
        # strip any anchor/query part after the file path
        target = re.split(r"[#?]", target, maxsplit=1)[0]
        if not target:
            continue
        normalized = target.replace("\\", "/")
        if normalized.startswith(IGNORED_PREFIXES):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_file():
            problems.append(f"{path.relative_to(REPO_ROOT).as_posix()}: broken link -> {raw}")
    return problems


def main() -> int:
    files = [REPO_ROOT / "README.md"]
    files += sorted((REPO_ROOT / "docs").rglob("*.md"))
    files = [f for f in files if f.exists()]
    if not files:
        print("check_links: no files found (unexpected)")
        return 1

    all_problems: list[str] = []
    checked = 0
    for f in files:
        targets = _targets(f.read_text(encoding="utf-8", errors="replace"))
        checked += len(targets)
        all_problems += broken_links_in(f)

    print(f"check_links: checked {checked} links in {len(files)} files")
    if all_problems:
        for p in all_problems:
            print(f"  BROKEN: {p}")
        print(f"check_links: FAILED ({len(all_problems)} broken link(s))")
        return 1
    print("check_links: all links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
