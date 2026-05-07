"""Localization checks: script + judges."""

from __future__ import annotations

import pytest

from genai_eval.localization.calque_judge import calque_score
from genai_eval.localization.honorific_judge import honorific_score
from genai_eval.localization.script_check import in_script_ratio, script_check
from genai_eval.providers.fake import FakeProvider

# ---- script_check ----


def test_script_check_japanese_pure() -> None:
    """Correctly-scripted Japanese passes."""
    out = script_check("猫はマットの上にいます。", "ja")
    assert out["pass"] == 1.0
    assert out["ratio"] == 1.0


def test_script_check_japanese_romaji_fails() -> None:
    """Romaji output for ja fails the script check."""
    out = script_check("neko wa matto no ue ni imasu", "ja")
    assert out["pass"] == 0.0
    assert out["ratio"] == 0.0


def test_script_check_japanese_mixed_flagged() -> None:
    """Mixed scripts: ratio between 0 and 1, fails 0.9 threshold."""
    ratio = in_script_ratio("猫 wa マット no 上", "ja")
    assert 0.0 < ratio < 1.0


def test_script_check_english() -> None:
    out = script_check("The cat is on the mat", "en")
    assert out["pass"] == 1.0


def test_script_check_spanish_with_accents() -> None:
    out = script_check("El gato está sobre la alfombra.", "es")
    assert out["pass"] == 1.0


def test_script_check_empty_text() -> None:
    """Empty text is vacuously correct."""
    out = script_check("", "ja")
    assert out["ratio"] == 1.0


# ---- judges ----


@pytest.mark.asyncio
async def test_honorific_judge_returns_float_in_range() -> None:
    judge = FakeProvider()
    score = await honorific_score(judge, "これは丁寧な日本語の文章です。")
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_calque_judge_returns_float_in_range() -> None:
    judge = FakeProvider()
    score = await calque_score(judge, "El gato está sobre la alfombra.")
    assert 0.0 <= score <= 1.0
