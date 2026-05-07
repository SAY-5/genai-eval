"""Translation task with explicit source-target language pairs."""

from __future__ import annotations

from pathlib import Path

import yaml

from genai_eval.metrics.chrf import chrf_score
from genai_eval.providers import ChatMessage
from genai_eval.tasks import Example

PASS_THRESHOLD = 0.40  # chrF >= 0.40 counts as a pass


def load_suite(language: str, suites_dir: Path) -> list[Example]:  # noqa: ARG001
    """Load translation pairs. `language` here is "all" — the suite enumerates pairs."""
    path = suites_dir / "translation" / "pairs.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Example(
            id=item["id"],
            task_type="translation",
            language=f"{item['source']}-{item['target']}",
            input={"text": item["text"], "source": item["source"], "target": item["target"]},
            gold={"text": item["gold"]},
            metadata=item.get("metadata", {}),
        )
        for item in raw["examples"]
    ]


def build_messages(ex: Example) -> list[ChatMessage]:
    src, tgt = ex.input["source"], ex.input["target"]
    tag = f"<<TAG kind=task|task_type=translation|example_id={ex.id}|lang_key={src}-{tgt}>>"
    sys = (
        "You are a translator. Translate the input from the source language to the "
        "target language. Output only the translation, nothing else."
    )
    user = f"{tag}\nSource ({src}): {ex.input['text']}\nTarget language: {tgt}"
    return [ChatMessage("system", sys), ChatMessage("user", user)]


def score(ex: Example, output: str) -> dict[str, float]:
    chrf = chrf_score(ex.gold["text"], output)
    return {"chrf": chrf, "pass": 1.0 if chrf >= PASS_THRESHOLD else 0.0}


__all__ = ["load_suite", "build_messages", "score", "PASS_THRESHOLD"]
