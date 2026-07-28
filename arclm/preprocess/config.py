from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PreprocessConfig:
    """Configuration for JSONL dataset cleaning and filtering.

    Parameters:
        text_field: Field read from each JSONL row.
        output_field: Field written with cleaned text.
        min_chars: Minimum cleaned-text character count.
        max_chars: Maximum cleaned-text character count.
        min_words: Minimum whitespace word count.
        max_repeated_char_run: Maximum repeated-character run before dropping.
        max_repeated_word_ratio: Maximum repeated-word ratio before dropping.
        min_entropy: Minimum character entropy heuristic.
        max_entropy: Maximum character entropy heuristic.
        allowed_languages: Heuristic language labels to keep.
        min_language_confidence: Minimum confidence for language detection.
        remove_html: Strip simple HTML markup before filtering.
        normalize_unicode: Normalize Unicode text.
        lowercase: Lowercase cleaned text.
        drop_urls: Replace URL-like text.
        drop_emails: Replace email-like text.
        drop_phone_numbers: Replace phone-like text.
        redact_pii: Apply built-in PII redaction heuristics.
        exact_dedup: Drop exact duplicate cleaned rows.
        near_dedup: Drop near-duplicates using SimHash.
        simhash_threshold: Maximum SimHash distance treated as duplicate.
        toxicity_enabled: Enable built-in toxicity heuristic.
        max_toxicity_score: Maximum toxicity score before dropping.
        perplexity_enabled: Enable simple perplexity heuristic.
        max_perplexity: Maximum simple perplexity score before dropping.
        workers: Reserved for future parallel execution; currently not used.
        batch_size: Reserved for future batching; currently not used by ``run``.
        report_html: Write an HTML report when ``report_dir`` is provided.
        report_json: Write a JSON report when ``report_dir`` is provided.

    Stability:
        Experimental in ArcLM 0.8.0.dev0. Field names are public but
        validation is intentionally lightweight.
    """

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
        """Load preprocessing settings from a YAML file.

        Raises:
            TypeError: If YAML values do not match dataclass fields.
            yaml.YAMLError: If the YAML file cannot be parsed.
        """

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the configuration."""

        return dict(self.__dict__)
