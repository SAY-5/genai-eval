"""Classification task — single-label sentiment."""

from __future__ import annotations

from pathlib import Path

import yaml

from genai_eval.metrics.exact_match import exact_match
from genai_eval.providers import ChatMessage
from genai_eval.tasks import Example


def load_suite(language: str, suites_dir: Path) -> list[Example]:
    path = suites_dir / "classification" / f"{language}.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Example(
            id=item["id"],
            task_type="classification",
            language=language,
            input=item["input"],
            gold=item["gold"],
            metadata={"labels": raw.get("labels", []), **item.get("metadata", {})},
        )
        for item in raw["examples"]
    ]


def build_messages(ex: Example) -> list[ChatMessage]:
    tag = f"<<TAG kind=task|task_type=classification|example_id={ex.id}|lang_key={ex.language}>>"
    labels = ex.metadata.get("labels", [])
    sys = (
        "You are a text classifier. Output exactly one label from the allowed list, "
        "no quotes, no punctuation, no extra words."
    )
    user = f"{tag}\n" f"Allowed labels: {labels}\n" f"Text: {ex.input['text']}"
    return [ChatMessage("system", sys), ChatMessage("user", user)]


def score(ex: Example, output: str) -> dict[str, float]:
    em = exact_match(ex.gold["label"], output)
    return {"accuracy": em, "pass": em}


__all__ = ["load_suite", "build_messages", "score"]
