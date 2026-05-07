"""Judge-vs-human calibration helpers.

Pure functions used by the ``genai-eval calibration`` CLI and the integration
tests. Splitting these from the API layer keeps the math unit-testable
without spinning up the ASGI app or the database.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryReport:
    """Summary stats for a single rubric category."""

    category: str
    n: int
    agreement_rate: float
    mae: float
    judge_mean: float
    human_mean: float


def aggregate_human_means(
    scores: Sequence[tuple[int, str, float]],
) -> dict[str, dict[int, float]]:
    """Return ``{category: {run_item_id: mean_human_score}}``.

    Multi-rater means are computed as the simple arithmetic mean of all human
    scores recorded against the same (item, category) pair.
    """
    bucket: dict[str, dict[int, list[float]]] = {}
    for run_item_id, category, score in scores:
        bucket.setdefault(category, {}).setdefault(run_item_id, []).append(score)
    return {
        cat: {item_id: sum(vals) / len(vals) for item_id, vals in items.items()}
        for cat, items in bucket.items()
    }


def calibration_report(
    *,
    judge_scores: Mapping[int, Mapping[str, float]],
    human_scores: Sequence[tuple[int, str, float]],
    threshold: float = 0.5,
) -> list[CategoryReport]:
    """Compute per-category agreement-rate and MAE.

    Args:
        judge_scores: ``{run_item_id: {category: judge_score}}``. Items
            without a record for a category fall back to the ``"pass"`` key.
        human_scores: list of ``(run_item_id, category, score)`` triples,
            potentially containing multiple entries per (item, category) for
            multi-rater workflows.
        threshold: ``|judge - human|`` at or below this counts as agreement.

    Returns:
        One ``CategoryReport`` per category, sorted by category name.
    """
    means = aggregate_human_means(human_scores)
    out: list[CategoryReport] = []
    for category in sorted(means):
        per_item = means[category]
        agreements = 0
        abs_errors: list[float] = []
        judge_vals: list[float] = []
        human_vals: list[float] = []
        for item_id, human_mean in per_item.items():
            item_scores = judge_scores.get(item_id, {})
            judge = float(item_scores.get(category, item_scores.get("pass", 0.0)))
            judge_vals.append(judge)
            human_vals.append(human_mean)
            err = abs(judge - human_mean)
            abs_errors.append(err)
            if err <= threshold:
                agreements += 1
        n = len(abs_errors)
        if n == 0:
            continue
        out.append(
            CategoryReport(
                category=category,
                n=n,
                agreement_rate=agreements / n,
                mae=sum(abs_errors) / n,
                judge_mean=sum(judge_vals) / n,
                human_mean=sum(human_vals) / n,
            )
        )
    return out


__all__ = ["CategoryReport", "aggregate_human_means", "calibration_report"]
