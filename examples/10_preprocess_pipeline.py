"""Preprocessing: clean a JSONL file and write a report."""

from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    try:
        from arclm.preprocess import PreprocessConfig, PreprocessPipeline
    except ImportError as exc:
        raise SystemExit("Install preprocessing dependencies with: pip install -e .[preprocess]") from exc

    rows = [
        {"text": "<p>ArcLM cleans useful dataset rows for training. Contact demo@example.com.</p>"},
        {"text": "too short"},
        {"text": "ArcLM cleans useful dataset rows for training. Contact demo@example.com."},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw.jsonl"
        cleaned = root / "cleaned.jsonl"
        reports = root / "reports"
        raw.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        report = PreprocessPipeline(
            PreprocessConfig(
                min_chars=20,
                min_words=4,
                min_entropy=1.0,
                max_entropy=8.0,
                min_language_confidence=0.1,
                drop_emails=True,
                redact_pii=True,
                near_dedup=False,
                report_json=True,
                report_html=True,
            )
        ).run(raw, cleaned, reports)

        print(f"Written rows: {report['written']}")
        print(f"Dropped rows: {report['dropped']}")


if __name__ == "__main__":
    main()
