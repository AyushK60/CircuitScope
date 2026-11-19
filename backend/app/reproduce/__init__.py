"""Reproduction scripts as text (Stage 5 Task 1).

The server only ever generates text; tests may execute it in a subprocess.
"""

from .generate import KINDS, filename, generate

__all__ = ["KINDS", "filename", "generate"]
