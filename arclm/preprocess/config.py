from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PreprocessConfig:
    text_field: str = "text"
    output_field: str = "text"
    min_chars: int = 20
    max_chars: int = 100_000
    min_words: int = 3
    max_repeated_char_run: int = 8
    max_repeated_word_ratio: float = 0.35
    min_entropy: float = 2.0
    max_entropy: float = 6.5
    allowed_languages: List[str] = field(default_factory=lambda: ["en", "ar"])
    min_language_confidence: float = 0.55
    remove_html: bool = True
    normalize_unicode: bool = True
    lowercase: bool = False
    drop_urls: bool = False
    drop_emails: bool = True
    drop_phone_numbers: bool = False
    redact_pii: bool = True
    exact_dedup: bool = True
    near_dedup: bool = True
    simhash_threshold: int = 3
    toxicity_enabled: bool = False
    max_toxicity_score: float = 0.85
    perplexity_enabled: bool = False
    max_perplexity: float = 800.0
    workers: int = 1
    batch_size: int = 1000
    report_html: bool = True
    report_json: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PreprocessConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
