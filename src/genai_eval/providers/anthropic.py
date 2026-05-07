"""Anthropic provider — thin wrapper, BYOK, tested with respx mocks."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import httpx

from genai_eval.providers import ChatMessage, ChatResult

_PRICES: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-latest": (0.003, 0.015),
    "claude-3-5-haiku-latest": (0.0008, 0.004),
    "claude-3-opus-latest": (0.015, 0.075),
}


class AnthropicProvider:
    """Live Anthropic Messages API provider. Reads ANTHROPIC_API_KEY from env."""

    name = "anthropic"

    def __init__(self, base_url: str = "https://api.anthropic.com/v1", timeout: float = 60.0):
        self._base_url = base_url
        self._timeout = timeout

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        temperature: float = 0.0,
    ) -> ChatResult:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        # Anthropic accepts a top-level system field plus user/assistant messages.
        system_chunks = [m.content for m in messages if m.role == "system"]
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in {"user", "assistant"}
        ]
        payload: dict[str, object] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 1024,
            "messages": chat_messages,
        }
        if system_chunks:
            payload["system"] = "\n\n".join(system_chunks)

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000

        # Concatenate text blocks.
        chunks = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        text = "".join(chunks)

        usage = data.get("usage", {})
        tokens_in = int(usage.get("input_tokens", 0))
        tokens_out = int(usage.get("output_tokens", 0))
        in_p, out_p = _PRICES.get(model, (0.0, 0.0))
        cost = (tokens_in / 1000.0) * in_p + (tokens_out / 1000.0) * out_p

        return ChatResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            model_version=str(data.get("model", model)),
        )
