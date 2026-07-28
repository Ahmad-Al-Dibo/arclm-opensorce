import pytest

from arclm import (
    DatasetValidationError,
    TextRecord,
    validate_record,
    validate_records,
)


def test_text_schema_validates_and_returns_plain_dict():
    record, errors, warnings = validate_record(
        {"text": "Example training text", "metadata": {"source": "unit"}},
        schema="text",
    )

    assert isinstance(record, TextRecord)
    assert errors == []
    assert warnings == []
    assert record.to_dict() == {
        "text": "Example training text",
        "metadata": {"source": "unit"},
    }


def test_strict_unknown_fields_are_errors_but_permissive_warns():
    strict_report = validate_records([{"text": "hello", "extra": 1}], schema="text", strict=True)
    permissive_report = validate_records([{"text": "hello", "extra": 1}], schema="text", strict=False)

    assert strict_report.is_valid is False
    assert strict_report.error_categories["unknown_field"] == 1
    assert permissive_report.is_valid is True
    assert permissive_report.warning_count == 1


def test_conversation_schema_validates_roles_and_empty_content():
    report = validate_records(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "content": "Nope"},
                ]
            }
        ],
        schema="conversation",
        strict=True,
    )

    assert report.invalid_record_indexes == [0]
    assert report.error_categories["empty_required_field"] == 1
    assert report.error_categories["invalid_role"] == 1
    with pytest.raises(DatasetValidationError):
        report.raise_for_errors()


def test_batch_validation_reports_duplicates_lengths_and_fields():
    report = validate_records(
        [{"prompt": "Q", "completion": "A"}, {"prompt": "Q", "completion": "A"}],
        schema="prompt_completion",
        check_duplicates=True,
    )

    assert report.total_records == 2
    assert report.valid_records == 2
    assert report.duplicates["duplicate_records"] == 1
    assert report.length_stats["count"] == 2
    assert report.detected_fields == ["completion", "prompt"]
