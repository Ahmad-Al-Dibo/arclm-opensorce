import json

from arclm.preprocess import PreprocessConfig, PreprocessPipeline


def test_preprocess_pipeline_filters_redacts_and_reports(tmp_path):
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "cleaned.jsonl"
    report_dir = tmp_path / "report"
    rows = [
        {
            "text": "<p>ArcLM cleans useful dataset rows for language model training. Email demo@example.com</p>"
        },
        {
            "text": "short"
        },
        {
            "text": "ArcLM cleans useful dataset rows for language model training. Email demo@example.com"
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    config = PreprocessConfig(
        min_chars=20,
        min_words=4,
        min_entropy=1.0,
        max_entropy=8.0,
        allowed_languages=["en"],
        min_language_confidence=0.1,
        drop_emails=True,
        redact_pii=True,
        near_dedup=False,
        report_html=False,
        report_json=True,
    )

    report = PreprocessPipeline(config).run(input_path, output_path, report_dir)
    cleaned = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert report["total"] == 3
    assert report["written"] == 1
    assert report["dropped"] == 2
    assert report["reasons"]["too_short_chars"] == 1
    assert report["reasons"]["exact_duplicate"] == 1
    assert cleaned[0]["text"].startswith("ArcLM cleans useful dataset rows")
    assert "[EMAIL]" in cleaned[0]["text"]
    assert (report_dir / "report.json").exists()
