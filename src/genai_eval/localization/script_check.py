"""Unicode-block based script-correctness check.

Returns the ratio of in-script characters over total alphabetic chars. Whitespace,
punctuation, and digits are excluded from the denominator. Numbers and ASCII
punctuation are universally accepted.
"""

from __future__ import annotations

import unicodedata


def _is_japanese(ch: str) -> bool:
    cp = ord(ch)
    # Hiragana
    if 0x3040 <= cp <= 0x309F:
        return True
    # Katakana (incl. phonetic ext)
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return True
    # CJK Unified Ideographs (incl. Ext A)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return True
    # CJK punctuation (e.g. 、 。)
    return 0x3000 <= cp <= 0x303F


def _is_latin(ch: str) -> bool:
    """Latin block plus Latin-1 supplement (covers en + es accented chars)."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return name.startswith("LATIN ")


def _is_alpha(ch: str) -> bool:
    return ch.isalpha()


def in_script_ratio(text: str, language: str) -> float:
    """Return ratio in [0, 1] of alphabetic chars belonging to the target script."""
    in_count = 0
    total = 0
    for ch in text:
        if not _is_alpha(ch):
            continue
        total += 1
        if (language == "ja" and _is_japanese(ch)) or (language in {"en", "es"} and _is_latin(ch)):
            in_count += 1
    if total == 0:
        return 1.0  # nothing alphabetic; treat as vacuously correct
    return in_count / total


def script_check(text: str, language: str, threshold: float = 0.9) -> dict[str, float]:
    """Return {ratio, pass} where pass=1.0 iff ratio >= threshold."""
    ratio = in_script_ratio(text, language)
    return {"ratio": ratio, "pass": 1.0 if ratio >= threshold else 0.0}
