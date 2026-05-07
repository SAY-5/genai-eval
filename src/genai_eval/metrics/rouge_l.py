"""ROUGE-L (longest common subsequence) F1, pure Python."""

from __future__ import annotations


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    # space-optimised DP: keep two rows
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        # reset curr for next iteration
        for k in range(m + 1):
            curr[k] = 0
    return prev[m]


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenizer with lowercase. Adequate for ROUGE-L."""
    return text.lower().split()


def rouge_l_f1(reference: str, hypothesis: str, beta: float = 1.0) -> float:
    """Return ROUGE-L F-beta (default F1) in [0, 1].

    Formula:
        P = LCS / |hyp|
        R = LCS / |ref|
        F = (1 + beta^2) * P * R / (R + beta^2 * P)
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    beta_sq = beta * beta
    denom = recall + beta_sq * precision
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom
