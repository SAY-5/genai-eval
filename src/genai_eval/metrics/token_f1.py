"""Token-overlap F1 (SQuAD-style)."""

from __future__ import annotations

from collections import Counter


def _tokenize(s: str) -> list[str]:
    return s.lower().split()


def token_f1(reference: str, hypothesis: str) -> float:
    """Return F1 over token multisets."""
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    common = Counter(ref) & Counter(hyp)
    matches = sum(common.values())
    if matches == 0:
        return 0.0
    precision = matches / len(hyp)
    recall = matches / len(ref)
    return 2 * precision * recall / (precision + recall)
