"""Tests for the A/B comparison harness."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

from genai_eval.comparisons import (
    CellCompare,
    compare_cells,
    render_markdown,
    two_proportion_pvalue,
)
from genai_eval.orchestrator import RunConfig, run_suite

SUITES_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "suites"


# ---- statistical helper ----


def test_two_proportion_pvalue_identical_returns_one() -> None:
    assert two_proportion_pvalue(20, 30, 20, 30) == 1.0


def test_two_proportion_pvalue_clear_difference_below_005() -> None:
    # 100% vs 0% with reasonable n must be highly significant.
    p = two_proportion_pvalue(50, 50, 1, 50)
    assert 0.0 <= p < 0.05


def test_two_proportion_pvalue_handles_empty_samples() -> None:
    assert two_proportion_pvalue(0, 0, 1, 1) == 1.0
    assert two_proportion_pvalue(1, 1, 0, 0) == 1.0


def test_two_proportion_pvalue_in_unit_interval() -> None:
    for sa, na, sb, nb in [(5, 10, 6, 10), (20, 100, 30, 100), (0, 5, 5, 5)]:
        p = two_proportion_pvalue(sa, na, sb, nb)
        assert 0.0 <= p <= 1.0


# ---- cell-level diff ----


def test_compare_cells_aligns_by_task_language() -> None:
    summary_a = {
        "cells": [
            {"task": "qa", "language": "en", "n": 10, "pass_rate": 0.5},
            {"task": "qa", "language": "es", "n": 10, "pass_rate": 1.0},
        ]
    }
    summary_b = {
        "cells": [
            {"task": "qa", "language": "en", "n": 10, "pass_rate": 0.8},
            {"task": "qa", "language": "es", "n": 10, "pass_rate": 1.0},
        ]
    }
    diffs = compare_cells(summary_a, summary_b)
    assert len(diffs) == 2
    by_key = {(d.task, d.language): d for d in diffs}
    en = by_key[("qa", "en")]
    assert math.isclose(en.delta, 0.3, abs_tol=1e-9)
    es = by_key[("qa", "es")]
    assert es.delta == 0.0
    assert es.p_value == 1.0


# ---- end-to-end determinism: byte-identical Markdown across runs ----


def _render_for_models(model_a: str, model_b: str) -> str:
    cfg_a = RunConfig(
        provider_name="fake",
        model=model_a,
        suite_filter={"tasks": ["qa"]},
        suites_dir=SUITES_DIR,
    )
    cfg_b = RunConfig(
        provider_name="fake",
        model=model_b,
        suite_filter={"tasks": ["qa"]},
        suites_dir=SUITES_DIR,
    )

    async def _exec() -> tuple[dict, dict]:
        a = await run_suite(cfg_a)
        b = await run_suite(cfg_b)
        return a, b

    a, b = asyncio.run(_exec())
    cells = compare_cells(a["summary"], b["summary"])
    return render_markdown(
        model_a=model_a,
        model_b=model_b,
        suite="qa",
        cells=cells,
        overall_a=a["summary"]["overall_pass_rate"],
        overall_b=b["summary"]["overall_pass_rate"],
    )


def test_compare_markdown_is_byte_identical_across_runs() -> None:
    one = _render_for_models("fake-small", "fake-large")
    two = _render_for_models("fake-small", "fake-large")
    assert one == two
    # Sanity: B (fake-large) should outperform A (fake-small) overall.
    overall_a_pct = float(_extract_overall(one, "A"))
    overall_b_pct = float(_extract_overall(one, "B"))
    assert overall_b_pct > overall_a_pct


def _extract_overall(md: str, arm: str) -> str:
    """Pull the percentage value from a 'overall A: 12.34%' style line."""
    needle = f"overall {arm}:"
    for line in md.splitlines():
        if needle in line:
            tail = line.split(needle, 1)[1].strip()
            return tail.rstrip("%").strip()
    return "0"


# ---- markdown shape ----


def test_render_markdown_lists_each_cell_once() -> None:
    cells = [
        CellCompare("qa", "en", 10, 10, 0.5, 0.8, 0.3, 0.05),
        CellCompare("qa", "es", 10, 10, 0.7, 0.7, 0.0, 1.0),
    ]
    md = render_markdown(
        model_a="alpha",
        model_b="beta",
        suite="qa",
        cells=cells,
        overall_a=0.6,
        overall_b=0.75,
    )
    assert "| qa | en |" in md
    assert "| qa | es |" in md
    assert md.count("| qa |") == 2
    assert "alpha" in md and "beta" in md
