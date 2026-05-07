"""chrF — character n-gram F-score, language-agnostic.

Reference: Popović (2015), "chrF: character n-gram F-score for automatic MT evaluation".
Defaults: n_min=1, n_max=6, beta=2 (recall-weighted, the standard chrF setting).
"""

from __future__ import annotations

from collections import Counter


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """Return the multiset of character n-grams (whitespace included)."""
    if n <= 0 or len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _ngram_f(ref: str, hyp: str, n: int, beta: float) -> float:
    ref_ng = _char_ngrams(ref, n)
    hyp_ng = _char_ngrams(hyp, n)
    if not ref_ng or not hyp_ng:
        return 0.0
    matches = sum((ref_ng & hyp_ng).values())
    if matches == 0:
        return 0.0
    precision = matches / sum(hyp_ng.values())
    recall = matches / sum(ref_ng.values())
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def chrf_score(
    reference: str,
    hypothesis: str,
    n_min: int = 1,
    n_max: int = 6,
    beta: float = 2.0,
) -> float:
    """Average chrF over n in [n_min, n_max], in [0, 1]."""
    if not reference or not hypothesis:
        return 0.0
    scores = [_ngram_f(reference, hypothesis, n, beta) for n in range(n_min, n_max + 1)]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
