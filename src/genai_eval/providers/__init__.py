"""LLM provider Protocol and concrete implementations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResult:
    """Provider response with usage telemetry."""

    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    model_version: str


class ChatProvider(Protocol):
    """Protocol every LLM provider implements."""

    name: str

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Call the chat endpoint and return a structured result."""
        ...


def build_provider(name: str) -> ChatProvider:
    """Factory for provider lookup by short name."""
    if name == "fake":
        from genai_eval.providers.fake import FakeProvider

        return FakeProvider()
    if name == "openai":
        from genai_eval.providers.openai import OpenAIProvider

        return OpenAIProvider()
    if name == "anthropic":
        from genai_eval.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    raise ValueError(f"Unknown provider: {name}")


__all__ = ["ChatMessage", "ChatResult", "ChatProvider", "build_provider"]
