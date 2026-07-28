from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterator
from tqdm import tqdm

from .cleaner import normalize_text, redact_patterns
from .config import PreprocessConfig
from .duplicate import DuplicateIndex
from .filters import basic_quality_reasons
from .io import read_jsonl, write_jsonl
from .language import language_reasons
from .perplexity import perplexity_reasons
from .pii import redact_pii
from .statistics import DatasetStats
from .toxicity import toxicity_reasons
from .report import write_html_report, write_json_report


class PreprocessPipeline:
    """Run ArcLM's built-in JSONL cleaning and filtering pipeline.

    The pipeline reads rows with a text field, applies deterministic cleaning,
    quality filters, language/toxicity/perplexity heuristics when configured,
    deduplication, and optional JSON/HTML reporting.

    Parameters:
        config: A :class:`PreprocessConfig` instance.

    Stability:
        Experimental in ArcLM 0.8.0.dev0. The high-level class is public, but
        the individual heuristics are intentionally simple and should be
        validated for each dataset.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.duplicates = DuplicateIndex()
        self.stats = DatasetStats()

    def process_row(self, row: Dict[str, Any]) -> tuple[Dict[str, Any] | None, list[str]]:
        """Clean and validate one JSON-like row.

        Parameters:
            row: Input mapping containing ``config.text_field``.

        Returns:
            A pair ``(cleaned_row, reasons)``. ``cleaned_row`` is ``None`` when
            the row is dropped; ``reasons`` contains drop reason codes.
        """

        cfg = self.config
        if "_error" in row:
            return None, [row["_error"]]
        text = str(row.get(cfg.text_field, ""))
        text = normalize_text(text, remove_html=cfg.remove_html, normalize_unicode=cfg.normalize_unicode, lowercase=cfg.lowercase)
        text = redact_patterns(text, urls=cfg.drop_urls, emails=cfg.drop_emails, phones=cfg.drop_phone_numbers)
        if cfg.redact_pii:
            text = redact_pii(text)

        reasons: list[str] = []
        reasons += basic_quality_reasons(text, cfg)
        reasons += language_reasons(text, cfg)
        reasons += toxicity_reasons(text, cfg)
        reasons += perplexity_reasons(text, cfg)
        reasons += self.duplicates.check_and_add(text, exact=cfg.exact_dedup, near=cfg.near_dedup, threshold=cfg.simhash_threshold)

        kept = len(reasons) == 0
        self.stats.add(text, kept, reasons)
        if not kept:
            return None, reasons
        out = dict(row)
        out[cfg.output_field] = text
        return out, []

    def run(self, input_path: str | Path, output_path: str | Path, report_dir: str | Path | None = None) -> Dict[str, Any]:
        """Process a JSONL file and write kept rows.

        Parameters:
            input_path: Input JSONL path.
            output_path: Output JSONL path for kept rows.
            report_dir: Optional directory for JSON and/or HTML reports.

        Returns:
            Report dictionary with total, kept, dropped, reason counts, and
            written-row count.
        """

        def kept_rows() -> Iterator[Dict[str, Any]]:
            for row in tqdm(read_jsonl(input_path), desc="preprocess"):
                cleaned, _ = self.process_row(row)
                if cleaned is not None:
                    yield cleaned

        written = write_jsonl(output_path, kept_rows())
        report = self.stats.to_dict()
        report["written"] = written
        if report_dir:
            report_dir = Path(report_dir)
            if self.config.report_json:
                write_json_report(report_dir / "report.json", report)
            if self.config.report_html:
                write_html_report(report_dir / "report.html", report)
        return report
