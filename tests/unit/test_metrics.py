"""Metric correctness tests."""

from __future__ import annotations

import math

from genai_eval.metrics.chrf import chrf_score
from genai_eval.metrics.exact_match import exact_match
from genai_eval.metrics.rouge_l import rouge_l_f1
from genai_eval.metrics.token_f1 import token_f1

# ---- ROUGE-L ----


def test_rouge_l_identical_is_one() -> None:
    s = "the quick brown fox jumps over the lazy dog"
    assert rouge_l_f1(s, s) == 1.0


def test_rouge_l_disjoint_is_zero() -> None:
    assert rouge_l_f1("abc def", "xyz qrs") == 0.0


def test_rouge_l_partial_match() -> None:
    ref = "the cat sat on the mat"
    hyp = "the cat is on the mat"
    score = rouge_l_f1(ref, hyp)
    # LCS = ["the", "cat", "on", "the", "mat"] = 5; ref=6 hyp=6 -> P=R=5/6, F1=5/6
    assert math.isclose(score, 5 / 6, rel_tol=1e-6)


def test_rouge_l_empty_inputs() -> None:
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0


# ---- chrF ----


def test_chrf_identical_is_one() -> None:
    s = "hello world"
    assert math.isclose(chrf_score(s, s), 1.0, rel_tol=1e-6)


def test_chrf_disjoint_is_low() -> None:
    score = chrf_score("aaaaaa", "zzzzzz")
    assert score == 0.0


def test_chrf_partial_overlap() -> None:
    # Some overlap: shared 'cat' substring
    score = chrf_score("the cat", "a cat")
    assert 0.0 < score < 1.0


def test_chrf_japanese_input() -> None:
    """chrF must handle non-Latin scripts (no crash, plausible score)."""
    score = chrf_score("猫はマットの上にいます。", "猫はマットの上にいます。")
    assert math.isclose(score, 1.0, rel_tol=1e-6)


# ---- exact_match ----


def test_exact_match_normalisation() -> None:
    assert exact_match("Paris", "  paris  ") == 1.0
    assert exact_match("Paris", "London") == 0.0


# ---- token_f1 ----


def test_token_f1_perfect() -> None:
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_subset() -> None:
    # ref=2 tokens, hyp=2 tokens, 1 match -> P=R=1/2, F1=0.5
    assert math.isclose(token_f1("hello world", "hello there"), 0.5, rel_tol=1e-6)


def test_token_f1_disjoint() -> None:
    assert token_f1("hello world", "foo bar") == 0.0
