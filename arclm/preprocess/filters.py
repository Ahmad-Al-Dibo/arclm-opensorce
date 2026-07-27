from __future__ import annotations

import math
import re
from collections import Counter

WORD_RE = re.compile(r"\w+", re.UNICODE)
REPEATED_CHAR_RE = re.compile(r"(.)\1{8,}", re.UNICODE)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def has_long_repeated_chars(text: str, max_run: int) -> bool:
    return re.search(r"(.)\1{" + str(max_run) + r",}", text) is not None


def repeated_word_ratio(text: str) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return repeats / max(1, len(words) - 1)


def basic_quality_reasons(text: str, cfg) -> list[str]:
    reasons: list[str] = []
    wc = word_count(text)
    ent = char_entropy(text)
    if len(text) < cfg.min_chars:
        reasons.append("too_short_chars")
    if len(text) > cfg.max_chars:
        reasons.append("too_long_chars")
    if wc < cfg.min_words:
        reasons.append("too_few_words")
    if has_long_repeated_chars(text, cfg.max_repeated_char_run):
        reasons.append("repeated_chars")
    if repeated_word_ratio(text) > cfg.max_repeated_word_ratio:
        reasons.append("repeated_words")
    if ent < cfg.min_entropy:
        reasons.append("low_entropy")
    if ent > cfg.max_entropy:
        reasons.append("high_entropy")
    return reasons
