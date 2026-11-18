"""Explain package (Stage 4): template-based claim generation, claim lint,
thresholds, and the shared glossary loader.

No LLM is involved anywhere: every sentence is a template from
``templates.py`` filled with measured values, and every claim passes
``lint.py`` before it ships (a report job that lints badly fails honestly).
"""
