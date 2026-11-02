"""Language detection (deterministic) used for replies and non-English ingestion."""
from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 42  # make langdetect deterministic

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "fr": "French", "es": "Spanish", "de": "German", "pt": "Portuguese",
    "ar": "Arabic", "zh-cn": "Chinese", "it": "Italian", "nl": "Dutch", "mr": "Marathi",
}

# Script-range shortcuts: very reliable for Indic scripts, even on short text.
_SCRIPT_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("ta", re.compile(r"[஀-௿]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
    ("te", re.compile(r"[ఀ-౿]")),
    ("kn", re.compile(r"[ಀ-೿]")),
    ("ml", re.compile(r"[ഀ-ൿ]")),
    ("ar", re.compile(r"[؀-ۿ]")),
]


def detect_language(text: str, default: str = "en") -> str:
    """Return an ISO language code.

    1) If a non-Latin script dominates, trust the script (works for 2-word messages).
    2) Otherwise use langdetect, accepting only confident (>0.80) results; short or
       mixed English text falls back to `default` to avoid false 'so'/'tl' guesses.
    """
    if not text or not text.strip():
        return default
    letters = [c for c in text if c.isalpha()]
    for code, pattern in _SCRIPT_HINTS:
        hits = len(pattern.findall(text))
        if letters and hits / len(letters) > 0.3:
            return code
    try:
        best = detect_langs(text)[0]
        if best.prob >= 0.80 and best.lang in LANGUAGE_NAMES:
            return best.lang
    except LangDetectException:
        pass
    return default


def language_name(code: str) -> str:
    """Human-readable language name for prompts."""
    return LANGUAGE_NAMES.get(code, code)
