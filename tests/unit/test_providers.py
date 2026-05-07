"""FakeProvider determinism + OpenAI/Anthropic respx-mocked happy paths."""

from __future__ import annotations

import os

import httpx
import pytest
import respx

from genai_eval.providers import ChatMessage
from genai_eval.providers.anthropic import AnthropicProvider
from genai_eval.providers.fake import FakeProvider
from genai_eval.providers.openai import OpenAIProvider

# ---- Fake ----


@pytest.mark.asyncio
async def test_fake_provider_deterministic() -> None:
    p = FakeProvider()
    msgs = [ChatMessage("user", "hello, world")]
    a = await p.chat(msgs, model="fake-large")
    b = await p.chat(msgs, model="fake-large")
    assert a.text == b.text


@pytest.mark.asyncio
async def test_fake_provider_scripted_lookup() -> None:
    """A tagged user message returns the scripted output."""
    p = FakeProvider()
    msgs = [
        ChatMessage("user", "<<TAG kind=task|task_type=qa|example_id=qa-001|lang_key=en>> ignored"),
    ]
    res = await p.chat(msgs, model="fake-large")
    assert res.text == "Paris"


@pytest.mark.asyncio
async def test_fake_provider_judge_default() -> None:
    """A judge tag returns the default rubric numeric score."""
    p = FakeProvider()
    msgs = [
        ChatMessage("user", "<<TAG kind=judge|rubric=preserves_meaning>> body"),
    ]
    res = await p.chat(msgs, model="fake-large")
    assert float(res.text) == 0.80


@pytest.mark.asyncio
async def test_fake_provider_unknown_tag_falls_back_to_hash() -> None:
    p = FakeProvider()
    res = await p.chat([ChatMessage("user", "no tag at all")], model="fake-large")
    assert res.text.startswith("[fake:")


@pytest.mark.asyncio
async def test_fake_provider_synthetic_echoes_ans_tag() -> None:
    """Synthetic examples replay the gold from the embedded ANS tag."""
    p = FakeProvider()
    msgs = [
        ChatMessage(
            "user",
            "<<TAG kind=task|task_type=qa|example_id=syn-001|lang_key=en>>\n"
            "Passage: Reference fact: Paris. <<ANS=Paris>>\n"
            "Question: What is the capital of France?",
        )
    ]
    res = await p.chat(msgs, model="fake-large")
    assert res.text == "Paris"


@pytest.mark.asyncio
async def test_fake_provider_synthetic_every_fifth_is_wrong() -> None:
    """The 5th, 10th, ... synthetic ids deterministically return a wrong stub."""
    p = FakeProvider()
    msgs = [
        ChatMessage(
            "user",
            "<<TAG kind=task|task_type=qa|example_id=syn-005|lang_key=en>>\n"
            "Passage: Reference fact: Paris. <<ANS=Paris>>",
        )
    ]
    res = await p.chat(msgs, model="fake-large")
    assert res.text != "Paris"
    assert "[no answer]" in res.text or "[wrong]" in res.text


# ---- OpenAI (respx-mocked) ----


@pytest.mark.asyncio
async def test_openai_provider_happy_path() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-test"
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )
        )
        p = OpenAIProvider()
        res = await p.chat([ChatMessage("user", "hello")], model="gpt-4o-mini")
        assert res.text == "hi"
        assert res.tokens_in == 5
        assert res.tokens_out == 1
        assert res.cost_usd > 0


@pytest.mark.asyncio
async def test_openai_provider_missing_key() -> None:
    os.environ.pop("OPENAI_API_KEY", None)
    p = OpenAIProvider()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await p.chat([ChatMessage("user", "hi")], model="gpt-4o-mini")


# ---- Anthropic (respx-mocked) ----


@pytest.mark.asyncio
async def test_anthropic_provider_happy_path() -> None:
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "claude-3-5-haiku-latest",
                    "content": [{"type": "text", "text": "hello back"}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            )
        )
        p = AnthropicProvider()
        res = await p.chat(
            [ChatMessage("system", "be brief"), ChatMessage("user", "hi")],
            model="claude-3-5-haiku-latest",
        )
        assert res.text == "hello back"
        assert res.tokens_in == 10
        assert res.tokens_out == 2
