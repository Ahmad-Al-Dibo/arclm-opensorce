from __future__ import annotations

from typing import Tuple


def detect_language(text: str) -> Tuple[str, float]:
    try:
        from langdetect import detect_langs
        langs = detect_langs(text[:5000])
        if not langs:
            return "unknown", 0.0
        best = langs[0]
        return best.lang, float(best.prob)
    except Exception:
        arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        total = max(1, arabic + latin)
        if arabic / total > 0.4:
            return "ar", arabic / total
        if latin / total > 0.4:
            return "en", latin / total
        return "unknown", 0.0


def language_reasons(text: str, cfg) -> list[str]:
    lang, conf = detect_language(text)
    if lang not in cfg.allowed_languages:
        return [f"language_not_allowed:{lang}"]
    if conf < cfg.min_language_confidence:
        return [f"language_low_confidence:{lang}:{conf:.2f}"]
    return []
