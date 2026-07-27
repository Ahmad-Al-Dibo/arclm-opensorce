from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, Any


@dataclass
class DatasetStats:
    total: int = 0
    kept: int = 0
    dropped: int = 0
    reasons: Counter = field(default_factory=Counter)
    chars: int = 0
    words: int = 0

    def add(self, text: str, kept: bool, reasons: list[str]) -> None:
        self.total += 1
        self.kept += int(kept)
        self.dropped += int(not kept)
        self.chars += len(text)
        self.words += len(text.split())
        for reason in reasons:
            self.reasons[reason] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "drop_rate": self.dropped / max(1, self.total),
            "avg_chars": self.chars / max(1, self.total),
            "avg_words": self.words / max(1, self.total),
            "reasons": dict(self.reasons),
        }
