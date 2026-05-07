"""Per-task scoring decision tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_eval.tasks import (
    Example,
    classification,
    code_repair,
    qa,
    summarization,
    translation,
)

SUITES_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "suites"


# ---- summarization ----


def test_summarization_load() -> None:
    examples = summarization.load_suite("en", SUITES_DIR)
    # The hand-curated baseline carries three examples; the synthetic suite
    # extension adds more on top. Either way the task type must be set.
    assert len(examples) >= 3
    assert all(e.task_type == "summarization" for e in examples)


def test_summarization_score_decision_table() -> None:
    ex = Example(
        id="x",
        task_type="summarization",
        language="en",
        input={"passage": ""},
        gold={"summary": "the cat sat on the mat"},
    )
    # perfect overlap -> pass
    perfect = summarization.score(ex, "the cat sat on the mat")
    assert perfect["pass"] == 1.0
    # disjoint -> fail
    bad = summarization.score(ex, "completely unrelated content")
    assert bad["pass"] == 0.0


# ---- translation ----


def test_translation_load() -> None:
    examples = translation.load_suite("all", SUITES_DIR)
    assert len(examples) >= 6
    pairs = {e.language for e in examples}
    assert "en-es" in pairs
    assert "en-ja" in pairs


def test_translation_score_decision_table() -> None:
    ex = Example(
        id="x",
        task_type="translation",
        language="en-es",
        input={"text": "hi", "source": "en", "target": "es"},
        gold={"text": "El gato está sobre la alfombra."},
    )
    perfect = translation.score(ex, "El gato está sobre la alfombra.")
    assert perfect["pass"] == 1.0
    bad = translation.score(ex, "Banana telephone xyz qrs.")
    assert bad["pass"] == 0.0


# ---- qa ----


def test_qa_score_decision_table() -> None:
    ex = Example(
        id="x",
        task_type="qa",
        language="en",
        input={"passage": "", "question": ""},
        gold={"answer": "Paris"},
    )
    assert qa.score(ex, "Paris")["pass"] == 1.0
    assert qa.score(ex, "London")["pass"] == 0.0
    assert qa.score(ex, "Paris")["exact_match"] == 1.0


# ---- classification ----


def test_classification_score_decision_table() -> None:
    ex = Example(
        id="x",
        task_type="classification",
        language="en",
        input={"text": ""},
        gold={"label": "positive"},
    )
    assert classification.score(ex, "positive")["pass"] == 1.0
    assert classification.score(ex, "negative")["pass"] == 0.0


# ---- code_repair ----


def test_code_repair_passes_with_correct_fix() -> None:
    ex = Example(
        id="x",
        task_type="code_repair",
        language="py",
        input={
            "buggy_code": "def add(a,b): return a-b\n",
            "test_code": "assert add(2,3)==5\nassert add(0,0)==0\n",
        },
        gold={"reference_fix": ""},
    )
    out = code_repair.score(ex, "def add(a, b):\n    return a + b\n")
    assert out["pass"] == 1.0


def test_code_repair_fails_with_buggy_fix() -> None:
    ex = Example(
        id="x",
        task_type="code_repair",
        language="py",
        input={
            "buggy_code": "",
            "test_code": "assert add(2,3)==5\n",
        },
        gold={"reference_fix": ""},
    )
    out = code_repair.score(ex, "def add(a, b):\n    return a - b\n")
    assert out["pass"] == 0.0


def test_code_repair_handles_runtime_error() -> None:
    """Code that raises during the test must score 0, not crash the harness."""
    ex = Example(
        id="x",
        task_type="code_repair",
        language="py",
        input={
            "buggy_code": "",
            "test_code": "assert add(2,3)==5\n",
        },
        gold={"reference_fix": ""},
    )
    out = code_repair.score(ex, "raise RuntimeError('boom')\n")
    assert out["pass"] == 0.0


@pytest.mark.parametrize("lang", ["en", "es", "ja"])
def test_qa_load_all_languages(lang: str) -> None:
    examples = qa.load_suite(lang, SUITES_DIR)
    assert len(examples) >= 3
