"""LLM-as-judge for Japanese honorific/register appropriateness."""

from __future__ import annotations

from typing import Optional

from genai_eval.providers import ChatMessage, ChatProvider

_RUBRIC = (
    "You are a Japanese-language register reviewer. Score the appropriateness of "
    "honorifics and politeness level on a 0.0-1.0 scale, where 1.0 is fully "
    "appropriate for a neutral business/written context. Respond with only the "
    "numeric score."
)


async def honorific_score(
    judge: ChatProvider,
    text: str,
    judge_model: str = "fake-large",
) -> float:
    """Return a [0, 1] score from an LLM-as-judge call."""
    user = (
        f"<<TAG kind=judge|rubric=honorific_appropriate>> Evaluate this Japanese "
        f"output:\n\n{text}"
    )
    result = await judge.chat(
        [ChatMessage("system", _RUBRIC), ChatMessage("user", user)],
        model=judge_model,
    )
    return _safe_float(result.text)


def _safe_float(s: str) -> float:
    try:
        v = float(s.strip().split()[0])
    except (ValueError, IndexError):
        return 0.0
    return max(0.0, min(1.0, v))


__all__ = ["honorific_score"]


# Re-export Optional to keep mypy happy across versions if anyone narrows the import.
_ = Optional
