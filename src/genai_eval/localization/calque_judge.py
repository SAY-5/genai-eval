"""LLM-as-judge for Spanish calque (literal-translation artifact) detection."""

from __future__ import annotations

from genai_eval.providers import ChatMessage, ChatProvider

_RUBRIC = (
    "You review Spanish text for calques: literal translations of English idioms "
    "or unnatural English-flavored phrasing. Score from 0.0 to 1.0 where 1.0 means "
    "fully natural Spanish, 0.0 means heavy calque artifacts. Respond with only "
    "the numeric score."
)


async def calque_score(
    judge: ChatProvider,
    text: str,
    judge_model: str = "fake-large",
) -> float:
    """Return a [0, 1] score; higher = fewer calque artifacts."""
    user = f"<<TAG kind=judge|rubric=calque_free>> Evaluate this Spanish output:\n\n{text}"
    result = await judge.chat(
        [ChatMessage("system", _RUBRIC), ChatMessage("user", user)],
        model=judge_model,
    )
    try:
        v = float(result.text.strip().split()[0])
    except (ValueError, IndexError):
        return 0.0
    return max(0.0, min(1.0, v))


__all__ = ["calque_score"]
