from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def exact_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def simhash(text: str, bits: int = 64) -> int:
    tokens = TOKEN_RE.findall(text.lower())
    vector = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            vector[i] += 1 if h & (1 << i) else -1
    out = 0
    for i, value in enumerate(vector):
        if value >= 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class DuplicateIndex:
    exact_seen: set[str] = field(default_factory=set)
    simhashes: list[int] = field(default_factory=list)

    def check_and_add(self, text: str, *, exact: bool = True, near: bool = True, threshold: int = 3) -> list[str]:
        reasons: list[str] = []
        if exact:
            h = exact_hash(text)
            if h in self.exact_seen:
                return ["exact_duplicate"]
            self.exact_seen.add(h)
        if near:
            s = simhash(text)
            if any(hamming(s, old) <= threshold for old in self.simhashes):
                reasons.append("near_duplicate")
            self.simhashes.append(s)
        return reasons
