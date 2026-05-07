"""Orchestrator unit tests: example collection, summary aggregation, error handling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from genai_eval.orchestrator import (
    ItemResult,
    RunConfig,
    collect_examples,
    summarise,
)
from genai_eval.providers import ChatMessage, ChatResult

SUITES_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "suites"


def test_collect_examples_all_tasks() -> None:
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={},
        suites_dir=SUITES_DIR,
    )
    examples = collect_examples(cfg)
    task_types = {e.task_type for e in examples}
    assert task_types == {"summarization", "translation", "qa", "classification", "code_repair"}


def test_collect_examples_filter_tasks() -> None:
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={"tasks": ["qa"]},
        suites_dir=SUITES_DIR,
    )
    examples = collect_examples(cfg)
    assert all(e.task_type == "qa" for e in examples)
    assert len(examples) == 9  # 3 langs × 3 examples


def test_collect_examples_smoke_caps_at_two() -> None:
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={"tasks": ["qa"]},
        suites_dir=SUITES_DIR,
        smoke=True,
    )
    examples = collect_examples(cfg)
    # 3 languages × 2 = 6
    assert len(examples) == 6


def test_summarise_empty() -> None:
    out = summarise([])
    assert out["overall_pass_rate"] == 0.0
    assert out["n_total"] == 0


def test_summarise_aggregates_per_cell() -> None:
    items = [
        ItemResult("qa", "en", "qa-001", "Paris", {"pass": 1.0}, 10.0, 0.0, "ok", None),
        ItemResult("qa", "en", "qa-002", "London", {"pass": 0.0}, 12.0, 0.0, "ok", None),
        ItemResult("qa", "es", "qa-001", "París", {"pass": 1.0}, 11.0, 0.0, "ok", None),
    ]
    summary = summarise(items)
    assert summary["overall_pass_rate"] == pytest.approx(2 / 3)
    cells = {(c["task"], c["language"]): c for c in summary["cells"]}
    assert cells[("qa", "en")]["pass_rate"] == 0.5
    assert cells[("qa", "es")]["pass_rate"] == 1.0


def test_summarise_counts_errors() -> None:
    items = [
        ItemResult("qa", "en", "x", "", {"pass": 0.0}, 0.0, 0.0, "error", "boom"),
        ItemResult("qa", "en", "y", "Paris", {"pass": 1.0}, 5.0, 0.0, "ok", None),
    ]
    summary = summarise(items)
    assert summary["n_errors"] == 1


# ---- error-path: a provider that raises shouldn't abort the whole run ----


@dataclass
class _FlakyProvider:
    name: str = "flaky"

    async def chat(
        self,
        messages: Sequence[ChatMessage],  # noqa: ARG002
        model: str,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
    ) -> ChatResult:
        raise RuntimeError("simulated failure")


@pytest.mark.asyncio
async def test_run_one_swallows_exceptions(fresh_db: None) -> None:  # noqa: ARG001
    """A provider that raises results in status='error', not a propagated exception."""
    import asyncio

    from genai_eval.orchestrator import _run_one
    from genai_eval.tasks import Example

    ex = Example(
        id="qa-001",
        task_type="qa",
        language="en",
        input={"passage": "x", "question": "y"},
        gold={"answer": "z"},
    )
    sem = asyncio.Semaphore(1)
    result = await _run_one(ex, _FlakyProvider(), "any-model", sem)
    assert result.status == "error"
    assert result.error_text is not None
    assert "simulated" in result.error_text
