from arclm import DataPipeline, Tokenizer


def test_pipeline_is_composable_reported_and_does_not_mutate_input():
    records = [
        {"text": "  ArcLM   prepares data  "},
        {"text": ""},
        {"text": "ArcLM prepares data"},
        {"text": "ArcLM prepares data"},
    ]

    processed, report = (
        DataPipeline(seed=123)
        .normalize_text("text")
        .remove_empty("text")
        .deduplicate("text")
        .validate("text", strict=True)
        .run(records)
    )

    assert records[0]["text"].startswith("  ")
    assert processed == [{"text": "ArcLM prepares data"}]
    assert report.input_count == 4
    assert report.output_count == 1
    assert report.removed_record_count == 3
    assert [operation["name"] for operation in report.configuration] == [
        "normalize_text",
        "remove_empty",
        "deduplicate",
        "validate",
    ]
    assert report.is_valid


def test_pipeline_formats_and_tokenizes_instruction_records():
    tokenizer = Tokenizer(max_vocab=20)
    tokenizer.build("Explain recursion Recursion is a function calling itself")

    records = [
        {
            "instruction": "Explain recursion",
            "input": "",
            "output": "Recursion is a function calling itself",
        }
    ]
    processed, report = (
        DataPipeline()
        .validate("instruction", strict=True)
        .format_instruction()
        .apply_tokenizer(tokenizer)
        .run(records)
    )

    assert "text" in processed[0]
    assert "tokens" in processed[0]
    assert report.operations[-1].name == "apply_tokenizer"


def test_pipeline_split_is_deterministic():
    records = [{"text": str(i)} for i in range(10)]

    first, _ = DataPipeline(seed=7).split(train=0.6, validation=0.2, test=0.2).run(records)
    second, _ = DataPipeline(seed=7).split(train=0.6, validation=0.2, test=0.2).run(records)

    assert first == second
    assert len(first[0]["train"]) == 6
