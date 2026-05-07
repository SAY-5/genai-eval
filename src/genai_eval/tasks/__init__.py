"""Task modules: each defines load_suite + score + build_prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    """A single eval example."""

    id: str
    task_type: str
    language: str  # "en" | "es" | "ja" | pair like "en-ja"
    input: dict[str, Any]
    gold: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


TASK_NAMES: tuple[str, ...] = (
    "summarization",
    "translation",
    "qa",
    "classification",
    "code_repair",
)


__all__ = ["Example", "TASK_NAMES"]
