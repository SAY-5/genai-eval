"""Summarization task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from genai_eval.metrics.rouge_l import rouge_l_f1
from genai_eval.providers import ChatMessage
from genai_eval.tasks import Example

PASS_THRESHOLD = 0.30  # ROUGE-L F1 >= 0.30 counts as a pass


def load_suite(language: str, suites_dir: Path) -> list[Example]:
    """Load summarization examples for a language."""
    path = suites_dir / "summarization" / f"{language}.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Example(
            id=item["id"],
            task_type="summarization",
            language=language,
            input=item["input"],
            gold=item["gold"],
            metadata=item.get("metadata", {}),
        )
        for item in raw["examples"]
    ]


def build_messages(ex: Example) -> list[ChatMessage]:
    """Build chat messages, embedding a TAG so FakeProvider can look up output."""
    tag = f"<<TAG kind=task|task_type=summarization|example_id={ex.id}|lang_key={ex.language}>>"
    sys = (
        "You are a careful summarizer. Produce a 1-2 sentence summary of the input "
        "passage in the same language as the input. Do not add information."
    )
    user = f"{tag}\nLanguage: {ex.language}\nPassage:\n{ex.input['passage']}"
    return [ChatMessage("system", sys), ChatMessage("user", user)]


def score(ex: Example, output: str) -> dict[str, float]:
    """Compute ROUGE-L F1 against the gold summary."""
    rouge = rouge_l_f1(ex.gold["summary"], output)
    return {"rouge_l": rouge, "pass": 1.0 if rouge >= PASS_THRESHOLD else 0.0}


__all__ = ["load_suite", "build_messages", "score", "PASS_THRESHOLD"]


# Re-export Any so tests can import the type hint without warnings.
_ = Any
