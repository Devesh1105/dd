"""Translation providers, with dubbing-aware length control.

Dubbing is not plain translation: the output has to *fit the original
timestamp*. German runs ~20% longer than English, Japanese far shorter, so
every provider here accepts a per-line character budget and is asked to hit
it. `char_budgets` is computed by the pipeline from the segment duration.
"""
from __future__ import annotations

import json
import re
import urllib.request

from ..config import settings
from .base import MTProvider, module_available, provider

# Rough syllable-rate expansion relative to English, used to size budgets and
# to warn the editor before a line is rendered too long for its slot.
EXPANSION: dict[str, float] = {
    "en": 1.00, "es": 1.20, "fr": 1.22, "de": 1.18, "it": 1.16, "pt": 1.18,
    "ru": 1.12, "pl": 1.14, "nl": 1.14, "tr": 1.05, "ar": 0.94, "he": 0.90,
    "hi": 1.10, "bn": 1.12, "ta": 1.16, "te": 1.14, "mr": 1.10, "ur": 1.06,
    "zh": 0.62, "ja": 0.86, "ko": 0.88, "th": 0.92, "vi": 1.02, "id": 1.10,
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "pl": "Polish", "nl": "Dutch", "tr": "Turkish",
    "ar": "Arabic", "he": "Hebrew", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "ur": "Urdu", "zh": "Chinese (Mandarin)",
    "ja": "Japanese", "ko": "Korean", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
}


def expansion_factor(source: str, target: str) -> float:
    return EXPANSION.get(target, 1.0) / max(0.4, EXPANSION.get(source, 1.0))


@provider("mt", "llm", rank=10, requires="OPENAI_API_KEY (or any OpenAI-compatible base URL)")
class LLMTranslator:
    """LLM translation with explicit length targets — best dubbing quality."""

    @staticmethod
    def is_available() -> bool:
        return bool(settings.openai_api_key)

    def translate(self, texts: list[str], source: str, target: str,
                  char_budgets: list[int] | None = None) -> list[str]:
        if not texts:
            return []
        lines = []
        for i, t in enumerate(texts):
            budget = char_budgets[i] if char_budgets and i < len(char_budgets) else 0
            lines.append({"id": i, "text": t, "max_chars": budget or None})
        prompt = (
            f"You are a professional dubbing translator. Translate each line from "
            f"{LANGUAGE_NAMES.get(source, source)} to {LANGUAGE_NAMES.get(target, target)}.\n"
            "Rules:\n"
            "- Preserve tone, register and speaker intent.\n"
            "- Respect max_chars when given: the line must be speakable within it. "
            "Shorten by rephrasing, never by dropping meaning.\n"
            "- Keep proper nouns. Do not add commentary.\n"
            'Return JSON: {"lines":[{"id":0,"text":"..."}]}\n\n'
            + json.dumps({"lines": lines}, ensure_ascii=False)
        )
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            f"{settings.openai_base_url.rstrip('/')}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {settings.openai_api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        out = list(texts)
        for line in parsed.get("lines", []):
            idx = int(line.get("id", -1))
            if 0 <= idx < len(out):
                out[idx] = str(line.get("text", "")).strip() or texts[idx]
        return out


@provider("mt", "argos", rank=20, requires="pip install argostranslate + language package")
class ArgosTranslator:
    """Fully offline neural MT (Argos Translate / OpenNMT)."""

    @staticmethod
    def is_available() -> bool:
        return module_available("argostranslate")

    def translate(self, texts: list[str], source: str, target: str,
                  char_budgets: list[int] | None = None) -> list[str]:
        import argostranslate.translate as at  # type: ignore

        if source in ("auto", ""):
            source = "en"
        try:
            translation = at.get_translation_from_codes(source, target)
        except Exception:
            return list(texts)
        return [translation.translate(t) if t.strip() else t for t in texts]


@provider("mt", "passthrough", rank=90)
class PassthroughTranslator:
    """Offline fallback: returns the source text unchanged.

    With no MT model installed we must not fabricate a translation, and we
    must not truncate either — silently dropping the end of a line loses
    meaning the editor can't recover. The text passes through intact, the
    time-fit stage compresses it to the slot, and the UI flags the line as
    over-budget so a human (or a configured MT provider) can shorten it.
    """

    def translate(self, texts: list[str], source: str, target: str,
                  char_budgets: list[int] | None = None) -> list[str]:
        return list(texts)


def fit_text(text: str, max_chars: int) -> str:
    """Trim on a word boundary without cutting mid-word."""
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    words = text.split()
    out: list[str] = []
    total = 0
    for w in words:
        add = len(w) + (1 if out else 0)
        if total + add > max_chars:
            break
        out.append(w)
        total += add
    return " ".join(out) if out else text[:max_chars]


# Comfortable speaking rate in characters per second, per language. Logographic
# and syllabic scripts carry far more meaning per character, so their rates are
# much lower even though the *spoken* pace is similar.
CHARS_PER_SECOND: dict[str, float] = {
    "en": 15.0, "es": 16.0, "fr": 16.0, "de": 14.5, "it": 16.0, "pt": 15.5,
    "ru": 14.0, "pl": 14.0, "nl": 14.5, "tr": 13.0, "ar": 12.5, "he": 12.5,
    "hi": 13.0, "bn": 12.5, "ta": 13.0, "te": 13.0, "mr": 13.0, "ur": 12.5,
    "zh": 5.2, "ja": 7.5, "ko": 8.5, "th": 10.0, "vi": 15.0, "id": 15.0,
}


def char_budget(duration: float, language: str) -> int:
    """Speakable characters that fit in `duration` for the target language."""
    return max(8, int(duration * CHARS_PER_SECOND.get(language, 15.0)))


def detect_language(text: str) -> str:
    """Script-based language hint — cheap and good enough to seed the UI."""
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    if re.search(r"[؀-ۿ]", text):
        return "ar"
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    if re.search(r"[Ѐ-ӿ]", text):
        return "ru"
    if re.search(r"[฀-๿]", text):
        return "th"
    return "en"
