"""QA task — short-answer question answering."""

from __future__ import annotations

from pathlib import Path

import yaml

from genai_eval.metrics.exact_match import exact_match
from genai_eval.metrics.token_f1 import token_f1
from genai_eval.providers import ChatMessage
from genai_eval.tasks import Example

PASS_THRESHOLD_F1 = 0.50


def load_suite(language: str, suites_dir: Path) -> list[Example]:
    path = suites_dir / "qa" / f"{language}.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Example(
            id=item["id"],
            task_type="qa",
            language=language,
            input=item["input"],
            gold=item["gold"],
            metadata=item.get("metadata", {}),
        )
        for item in raw["examples"]
    ]


def build_messages(ex: Example) -> list[ChatMessage]:
    tag = f"<<TAG kind=task|task_type=qa|example_id={ex.id}|lang_key={ex.language}>>"
    sys = (
        "You are a precise question-answering system. Answer the question using only "
        "information present in the reference passage. Reply with the shortest correct "
        "answer; no extra words."
    )
    user = f"{tag}\n" f"Passage:\n{ex.input['passage']}\n\n" f"Question: {ex.input['question']}"
    return [ChatMessage("system", sys), ChatMessage("user", user)]


def score(ex: Example, output: str) -> dict[str, float]:
    em = exact_match(ex.gold["answer"], output)
    f1 = token_f1(ex.gold["answer"], output)
    return {"exact_match": em, "token_f1": f1, "pass": 1.0 if f1 >= PASS_THRESHOLD_F1 else 0.0}


__all__ = ["load_suite", "build_messages", "score", "PASS_THRESHOLD_F1"]
