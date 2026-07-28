import json

import pytest

from arclm.data import (
    analyze_dataset,
    check_leakage,
    find_duplicates,
    find_near_duplicates,
    open_dataset,
    shard_dataset,
    split_dataset,
)
from arclm.exceptions import DatasetFormatError


pytestmark = [pytest.mark.integration, pytest.mark.scale]


def test_streaming_jsonl_repeatable_and_malformed_report(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"id":"a","text":"hello"}\nnot-json\n{"id":"b","text":"world"}\n', encoding="utf-8")

    source = open_dataset(path, format="jsonl", streaming=True, malformed="report")
    first = list(source)
    second = list(source)

    assert source.streaming
    assert source.seekable
    assert first == second
    assert first[1]["_error"]
    assert first[1]["_path"] == str(path)
    assert first[1]["_index"] == 1


def test_streaming_jsonl_malformed_raise(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"ok"}\nnope\n', encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="record 1"):
        list(open_dataset(path, format="jsonl", malformed="raise"))


def test_sharding_has_no_duplication_or_loss():
    rows = [{"id": str(index), "text": f"row {index}"} for index in range(17)]
    for strategy in ["contiguous", "round_robin", "hash"]:
        result = shard_dataset(rows, num_shards=4, strategy=strategy, key="id", seed=7)
        flattened = [row["id"] for shard in result.shards for row in shard]
        assert sorted(flattened, key=int) == [str(index) for index in range(17)]
        assert len(flattened) == len(set(flattened))
        assert sum(result.report.counts) == 17


def test_splitting_is_deterministic_and_group_aware():
    rows = [{"id": str(index), "group": str(index // 2), "text": f"row {index}"} for index in range(20)]
    first = split_dataset(rows, train=0.7, validation=0.2, test=0.1, strategy="hash", key="id", seed=42)
    second = split_dataset(rows, train=0.7, validation=0.2, test=0.1, strategy="hash", key="id", seed=42)
    assert first.to_dict() == second.to_dict()

    grouped = split_dataset(rows, train=0.7, validation=0.2, test=0.1, group_key="group", seed=42)
    assignments = {}
    for split_name, split_rows in grouped.splits.items():
        for row in split_rows:
            assignments.setdefault(row["group"], split_name)
            assert assignments[row["group"]] == split_name


def test_quality_duplicate_and_leakage_privacy():
    rows = [
        {"text": "hello user@example.com"},
        {"text": "hello user@example.com"},
        {"text": ""},
        {"messages": [{"role": "bad", "content": "x"}]},
    ]
    quality = analyze_dataset(rows, schema=None, include_samples=True)
    assert quality.total_records == 4
    assert quality.metrics["duplicate_records"] == 1
    assert any(issue.category == "empty_text" for issue in quality.issues)
    assert any("[EMAIL]" in sample["sample"] for sample in quality.samples)

    dupes = find_duplicates(rows, fields=["text"], normalize=True)
    assert dupes.has_duplicates
    near = find_near_duplicates([{"text": "hello world"}, {"text": "hello world again"}], threshold=0.5)
    assert near.has_duplicates

    leakage = check_leakage([{"text": "same"}], [{"text": "same"}, {"text": "different"}], fields=["text"])
    assert leakage.duplicate_records == 1
