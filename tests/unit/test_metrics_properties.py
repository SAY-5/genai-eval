"""Property-based tests for metric implementations.

These cover algebraic properties (idempotence, range, symmetry where applicable)
across a wide input space using Hypothesis. They complement the example-based
tests in ``test_metrics.py``.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from genai_eval.metrics.chrf import chrf_score
from genai_eval.metrics.exact_match import exact_match
from genai_eval.metrics.rouge_l import rouge_l_f1
from genai_eval.metrics.token_f1 import token_f1

# Token strategy: small alphabet so we exercise overlap. Hypothesis text() with
# alphabet=letters and a min size keeps values meaningful.
_token = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=8,
)
# A non-empty whitespace-tokenized "sentence".
_sentence = st.lists(_token, min_size=1, max_size=10).map(lambda toks: " ".join(toks))
# A non-empty raw character string for chrF.
_chars = st.text(min_size=1, max_size=40)


# ---- ROUGE-L properties ----


@given(s=_sentence)
@settings(max_examples=200, deadline=None)
def test_rouge_l_self_is_one(s: str) -> None:
    assert math.isclose(rouge_l_f1(s, s), 1.0, abs_tol=1e-9)


@given(s=_sentence)
@settings(max_examples=100, deadline=None)
def test_rouge_l_with_empty_is_zero(s: str) -> None:
    assert rouge_l_f1(s, "") == 0.0
    assert rouge_l_f1("", s) == 0.0


@given(a=_sentence, b=_sentence)
@settings(max_examples=200, deadline=None)
def test_rouge_l_in_unit_interval(a: str, b: str) -> None:
    score = rouge_l_f1(a, b)
    assert 0.0 <= score <= 1.0


@given(a=_sentence, b=_sentence)
@settings(max_examples=200, deadline=None)
def test_rouge_l_symmetric_with_default_beta(a: str, b: str) -> None:
    # With beta=1 (F1), swapping precision/recall must yield the same value.
    assert math.isclose(rouge_l_f1(a, b), rouge_l_f1(b, a), abs_tol=1e-9)


# ---- chrF properties ----


@given(s=st.text(min_size=6, max_size=40))
@settings(max_examples=200, deadline=None)
def test_chrf_self_is_one(s: str) -> None:
    # chrF averages F over n=1..6 by default, so the equality only holds when
    # the input is long enough that every n-gram order is non-empty.
    assert math.isclose(chrf_score(s, s), 1.0, abs_tol=1e-9)


@given(s=_chars)
@settings(max_examples=100, deadline=None)
def test_chrf_with_empty_is_zero(s: str) -> None:
    assert chrf_score(s, "") == 0.0
    assert chrf_score("", s) == 0.0


@given(a=_chars, b=_chars)
@settings(max_examples=200, deadline=None)
def test_chrf_in_unit_interval(a: str, b: str) -> None:
    score = chrf_score(a, b)
    assert 0.0 <= score <= 1.0


# ---- exact_match properties ----


@given(s=_sentence)
@settings(max_examples=100, deadline=None)
def test_exact_match_self_is_one(s: str) -> None:
    assert exact_match(s, s) == 1.0


@given(s=_sentence)
@settings(max_examples=100, deadline=None)
def test_exact_match_whitespace_invariant(s: str) -> None:
    # exact_match normalises whitespace, so collapsing or padding spaces
    # should not affect the result.
    padded = "   " + "  ".join(s.split()) + "   "
    assert exact_match(s, padded) == 1.0


@given(a=_sentence, b=_sentence)
@settings(max_examples=200, deadline=None)
def test_exact_match_is_zero_or_one(a: str, b: str) -> None:
    score = exact_match(a, b)
    assert score in (0.0, 1.0)


# ---- token_f1 properties ----


@given(s=_sentence)
@settings(max_examples=200, deadline=None)
def test_token_f1_self_is_one(s: str) -> None:
    assert math.isclose(token_f1(s, s), 1.0, abs_tol=1e-9)


@given(s=_sentence)
@settings(max_examples=100, deadline=None)
def test_token_f1_with_empty_is_zero(s: str) -> None:
    assert token_f1(s, "") == 0.0
    assert token_f1("", s) == 0.0


@given(a=_sentence, b=_sentence)
@settings(max_examples=200, deadline=None)
def test_token_f1_in_unit_interval(a: str, b: str) -> None:
    score = token_f1(a, b)
    assert 0.0 <= score <= 1.0


@given(a=_sentence, b=_sentence)
@settings(max_examples=200, deadline=None)
def test_token_f1_symmetric(a: str, b: str) -> None:
    # F1 over multiset intersection is symmetric.
    assert math.isclose(token_f1(a, b), token_f1(b, a), abs_tol=1e-9)


@given(toks=st.lists(_token, min_size=1, max_size=10))
@settings(max_examples=100, deadline=None)
def test_token_f1_permutation_invariant(toks: list[str]) -> None:
    # token_f1 operates on multisets, so permuting the hypothesis tokens
    # must yield the same score against the reference.
    ref = " ".join(toks)
    rev = " ".join(reversed(toks))
    assert math.isclose(token_f1(ref, rev), 1.0, abs_tol=1e-9)
