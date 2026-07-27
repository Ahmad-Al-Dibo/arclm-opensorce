from __future__ import annotations

import math
from collections import Counter


def simple_perplexity(text: str) -> float:
    # Dependency-free approximation for filtering extremely noisy text.
    tokens = text.lower().split()
    if not tokens:
        return float("inf")
    counts = Counter(tokens)
    total = len(tokens)
    nll = -sum(math.log((counts[t] + 1) / (total + len(counts))) for t in tokens)
    return math.exp(nll / total)


def perplexity_reasons(text: str, cfg) -> list[str]:
    if not cfg.perplexity_enabled:
        return []
    ppl = simple_perplexity(text)
    if ppl > cfg.max_perplexity:
        return [f"perplexity:{ppl:.2f}"]
    return []
