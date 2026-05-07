"""eval-smoke: full pipeline end-to-end against FakeProvider, asserts numbers.

This test runs the orchestrator against the committed YAML suites and the
FakeProvider, then asserts on real measured numbers. CI uses this as the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_eval.orchestrator import RunConfig, run_suite

SUITES_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "suites"


@pytest.mark.asyncio
async def test_full_eval_against_fake_provider(fresh_db: None) -> None:  # noqa: ARG001
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={},
        suites_dir=SUITES_DIR,
    )
    result = await run_suite(cfg)
    summary = result["summary"]

    # Pipeline correctness: every cell must report n_total > 0.
    assert summary["n_total"] > 0
    cells = summary["cells"]
    # 5 tasks × 3 langs minus translation langs (1) plus translation pairs (3)
    # plus code_repair (1 lang). Concrete: summarization 3, translation 1,
    # qa 3, classification 3, code_repair 1 = 11 cells (translation reports per pair).
    assert len(cells) >= 5  # at least one cell per task type

    # FakeProvider is intentionally imperfect: overall pass rate must be < 1.0
    # (otherwise the failure path is never exercised).
    assert summary["overall_pass_rate"] < 1.0

    # All committed runs must have zero infrastructure errors against FakeProvider.
    assert summary["n_errors"] == 0

    # Spot-check that pass rates are sensible (between 0 and 1).
    for cell in cells:
        assert 0.0 <= cell["pass_rate"] <= 1.0
        assert cell["n"] > 0
