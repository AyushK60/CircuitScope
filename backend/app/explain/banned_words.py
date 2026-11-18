"""Banned words for generated claims (Stage 4 principles 4 and 7).

Fixed lists, kept in the repo so the lint is auditable. Checked
case-insensitively; single words are matched as whole words only.

- BANNED_WORDS / BANNED_PHRASES: causal or mind-ascribing language that no
  generated claim may use, at any level. Generated causal statements must name
  the intervention ("in this test, copying X changed Y by Z").
- BANNED_IDIOMS: idioms banned at the `curious` level only (plain English for
  readers of any background).
"""

BANNED_WORDS = [
    "proves",
    "proof",
    "understands",
    "knows",
    "thinks",
    "reasons",
    "decides",
    "causes",
]

BANNED_PHRASES = [
    "responsible for",
    "is the reason",
    "explains why",
]

BANNED_IDIOMS = [
    "black box",
    "in a nutshell",
    "the bottom line",
    "at the end of the day",
    "shed light on",
    "deep dive",
    "rule of thumb",
    "behind the scenes",
    "piece of the puzzle",
    "grain of salt",
    "the big picture",
    "food for thought",
    "on the same page",
    "reads between the lines",
    "the whole nine yards",
    "pull back the curtain",
]
