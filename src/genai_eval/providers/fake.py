"""Deterministic FakeProvider for hermetic testing and CI."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from genai_eval.providers import ChatMessage, ChatResult

# Scripted responses keyed by (task_type, example_id, language).
# Some are deliberately wrong so the eval pipeline exercises the failure path.
SCRIPTED: dict[tuple[str, str, str], str] = {
    # ---- summarization ----
    ("summarization", "sum-001", "en"): (
        "The article describes how solar panels convert sunlight into electricity using "
        "photovoltaic cells."
    ),
    ("summarization", "sum-002", "en"): "Bees are dying because of pesticides.",  # weak
    ("summarization", "sum-003", "en"): (
        "The text discusses the history of the printing press and its impact on Europe."
    ),
    ("summarization", "sum-001", "es"): (
        "El artículo explica cómo los paneles solares convierten la luz solar en "
        "electricidad mediante células fotovoltaicas."
    ),
    ("summarization", "sum-002", "es"): "Las abejas están desapareciendo.",
    ("summarization", "sum-003", "es"): (
        "El texto describe la invención de la imprenta y su efecto en Europa."
    ),
    ("summarization", "sum-001", "ja"): (
        "この記事は、太陽光パネルが光起電力セルを用いて太陽光を電気に変換する仕組みを説明している。"
    ),
    ("summarization", "sum-002", "ja"): "ミツバチが減少している。",
    ("summarization", "sum-003", "ja"): (
        "本文は活版印刷の発明とそのヨーロッパへの影響について論じている。"
    ),
    # ---- translation ----
    ("translation", "trn-001", "en-es"): "El gato está sobre la alfombra.",
    ("translation", "trn-002", "en-es"): "Hoy llueve mucho.",
    ("translation", "trn-003", "en-es"): "Me gusta leer libros.",
    ("translation", "trn-001", "en-ja"): "猫はマットの上にいます。",
    ("translation", "trn-002", "en-ja"): "neko wa matto no ue ni imasu.",  # romaji = wrong script
    ("translation", "trn-003", "en-ja"): "私は本を読むのが好きです。",
    ("translation", "trn-001", "es-en"): "The cat is on the mat.",
    ("translation", "trn-002", "es-en"): "It is raining a lot today.",
    ("translation", "trn-003", "es-en"): "I like to read books.",
    # ---- qa ----
    ("qa", "qa-001", "en"): "Paris",
    ("qa", "qa-002", "en"): "1969",
    ("qa", "qa-003", "en"): "Tokyo",  # wrong: gold is "Kyoto"
    ("qa", "qa-001", "es"): "París",
    ("qa", "qa-002", "es"): "1969",
    ("qa", "qa-003", "es"): "Madrid",  # wrong
    ("qa", "qa-001", "ja"): "パリ",
    ("qa", "qa-002", "ja"): "1969年",
    ("qa", "qa-003", "ja"): "京都",
    # ---- classification ----
    ("classification", "cls-001", "en"): "positive",
    ("classification", "cls-002", "en"): "negative",
    ("classification", "cls-003", "en"): "positive",  # wrong: gold negative
    ("classification", "cls-001", "es"): "positivo",
    ("classification", "cls-002", "es"): "negativo",
    ("classification", "cls-003", "es"): "neutral",  # wrong
    ("classification", "cls-001", "ja"): "ポジティブ",
    ("classification", "cls-002", "ja"): "ネガティブ",
    ("classification", "cls-003", "ja"): "ポジティブ",  # wrong
    # ---- code_repair ----
    ("code_repair", "cr-001", "py"): ("def add(a, b):\n    return a + b\n"),
    ("code_repair", "cr-002", "py"): (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
    ),
    ("code_repair", "cr-003", "py"): (
        # deliberately still buggy: off-by-one
        "def find_max(xs):\n"
        "    m = xs[0]\n"
        "    for x in xs[1:-1]:\n"
        "        if x > m:\n"
        "            m = x\n"
        "    return m\n"
    ),
    # ---- judge rubric responses ----
    # The orchestrator never asks the judge unless the test asks for it; rubric
    # responses are short numeric strings we parse as floats.
}


# LLM-as-judge rubric defaults. The judge call's "messages" carry a "rubric:"
# tag so we can short-circuit deterministically.
JUDGE_DEFAULTS: dict[str, str] = {
    "covers_key_points": "0.85",
    "preserves_meaning": "0.80",
    "factually_grounded": "0.75",
    "honorific_appropriate": "0.90",
    "calque_free": "0.70",
}


class FakeProvider:
    """Deterministic, offline provider keyed by hash + scripted table.

    Deliberately produces some wrong outputs so the eval pipeline exercises the
    failure path. Used for hermetic CI runs and committed baselines.
    """

    name = "fake"

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Look up scripted response by tag, otherwise hash-derived stub."""
        del temperature  # deterministic
        prompt = "\n".join(f"{m.role}:{m.content}" for m in messages)
        text = self._lookup(messages) or self._hashed_stub(prompt, model)

        prompt_chars = sum(len(m.content) for m in messages)
        tokens_in = max(1, prompt_chars // 4)
        tokens_out = max(1, len(text) // 4)
        # FakeProvider has zero monetary cost.
        return ChatResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            latency_ms=1.0,
            model_version=f"{model}@fake",
        )

    @staticmethod
    def _lookup(messages: Sequence[ChatMessage]) -> str | None:
        """Decode the orchestrator's tag protocol from the user message."""
        for msg in messages:
            if msg.role != "user":
                continue
            tag = _parse_tag(msg.content)
            if tag is None:
                continue
            kind = tag.get("kind", "")
            if kind == "task":
                key = (tag["task_type"], tag["example_id"], tag["lang_key"])
                return SCRIPTED.get(key)
            if kind == "judge":
                return JUDGE_DEFAULTS.get(tag.get("rubric", ""), "0.80")
        return None

    @staticmethod
    def _hashed_stub(prompt: str, model: str) -> str:
        digest = hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()
        return f"[fake:{digest[:12]}]"


def _parse_tag(content: str) -> dict[str, str] | None:
    """Extract <<TAG ...>> metadata that the orchestrator embeds in prompts."""
    start = content.find("<<TAG ")
    if start < 0:
        return None
    end = content.find(">>", start)
    if end < 0:
        return None
    body = content[start + len("<<TAG ") : end]
    out: dict[str, str] = {}
    for part in body.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out
