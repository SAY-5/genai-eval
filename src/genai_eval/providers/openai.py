"""OpenAI provider — thin wrapper, BYOK, tested with respx mocks."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import httpx

from genai_eval.providers import ChatMessage, ChatResult

# Approximate per-1k-token prices, USD. Update as needed; only used for cost reporting.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
}


class OpenAIProvider:
    """Live OpenAI provider. Reads OPENAI_API_KEY from env."""

    name = "openai"

    def __init__(self, base_url: str = "https://api.openai.com/v1", timeout: float = 60.0):
        self._base_url = base_url
        self._timeout = timeout

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        temperature: float = 0.0,
    ) -> ChatResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
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
