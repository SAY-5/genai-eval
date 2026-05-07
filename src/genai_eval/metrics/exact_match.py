"""Exact-match metric (case-insensitive, whitespace-normalised)."""

from __future__ import annotations


def _normalise(s: str) -> str:
    return " ".join(s.lower().split()).strip()


def exact_match(reference: str, hypothesis: str) -> float:
    """Return 1.0 if normalised strings match exactly, else 0.0."""
    return 1.0 if _normalise(reference) == _normalise(hypothesis) else 0.0
