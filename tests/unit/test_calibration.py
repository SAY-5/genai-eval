"""Unit tests for the judge-vs-human calibration helpers."""

from __future__ import annotations

import math

from genai_eval.calibration import aggregate_human_means, calibration_report


def test_aggregate_human_means_takes_simple_mean_per_item() -> None:
    triples = [
        (1, "pass", 1.0),
        (1, "pass", 0.0),  # mean: 0.5
        (2, "pass", 1.0),
        (3, "factuality", 0.6),
        (3, "factuality", 0.8),  # mean: 0.7
    ]
    means = aggregate_human_means(triples)
    assert means["pass"] == {1: 0.5, 2: 1.0}
    assert means["factuality"] == {3: 0.7}


def test_calibration_report_perfect_agreement_yields_rate_one_mae_zero() -> None:
    judge = {1: {"pass": 1.0}, 2: {"pass": 0.0}, 3: {"pass": 1.0}}
    triples = [(1, "pass", 1.0), (2, "pass", 0.0), (3, "pass", 1.0)]
    reports = calibration_report(judge_scores=judge, human_scores=triples, threshold=0.5)
    assert len(reports) == 1
    r = reports[0]
    assert r.category == "pass"
    assert r.n == 3
    assert r.agreement_rate == 1.0
    assert r.mae == 0.0
    assert math.isclose(r.judge_mean, 2 / 3, abs_tol=1e-9)
    assert math.isclose(r.human_mean, 2 / 3, abs_tol=1e-9)


def test_calibration_report_ten_agree_one_disagree_matches_expectation() -> None:
    """The integration-test expectation: 10 of 11 agree at threshold=0.5."""
    judge = {i: {"pass": 1.0} for i in range(1, 12)}
    triples = [(i, "pass", 1.0) for i in range(1, 11)]  # 10 perfect agreements
    triples.append((11, "pass", 0.0))  # 1 sharp disagreement
    reports = calibration_report(judge_scores=judge, human_scores=triples, threshold=0.5)
    assert len(reports) == 1
    r = reports[0]
    assert r.n == 11
    assert r.agreement_rate == 10 / 11
    # MAE = (10*0 + 1*1) / 11
    assert r.mae == 1 / 11


def test_calibration_report_threshold_controls_agreement() -> None:
    judge = {1: {"pass": 0.0}, 2: {"pass": 0.0}}
    triples = [(1, "pass", 0.4), (2, "pass", 0.6)]
    # threshold=0.5: both items agree (|0-0.4|=0.4 <= 0.5; |0-0.6|=0.6 > 0.5)
    reports = calibration_report(judge_scores=judge, human_scores=triples, threshold=0.5)
    assert reports[0].agreement_rate == 0.5
    # threshold=0.7: both agree.
    reports = calibration_report(judge_scores=judge, human_scores=triples, threshold=0.7)
    assert reports[0].agreement_rate == 1.0
    # threshold=0.3: neither agrees.
    reports = calibration_report(judge_scores=judge, human_scores=triples, threshold=0.3)
    assert reports[0].agreement_rate == 0.0


def test_calibration_report_falls_back_to_pass_when_category_missing() -> None:
    judge = {1: {"pass": 0.7}}
    triples = [(1, "factuality", 0.7)]  # category not in judge -> uses "pass"
    reports = calibration_report(judge_scores=judge, human_scores=triples, threshold=0.0)
    assert reports[0].agreement_rate == 1.0
    assert reports[0].category == "factuality"


def test_calibration_report_returns_categories_sorted() -> None:
    judge = {1: {"a": 0.5, "b": 0.5}}
    triples = [(1, "b", 0.5), (1, "a", 0.5)]
    reports = calibration_report(judge_scores=judge, human_scores=triples)
    assert [r.category for r in reports] == ["a", "b"]
