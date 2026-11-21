"""Reproduction scripts as text.

The server only ever generates text; tests may execute it in a subprocess.
"""

from .generate import KINDS, filename, generate

__all__ = ["KINDS", "filename", "generate"]
