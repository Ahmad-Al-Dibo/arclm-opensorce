from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def test_load_clean_split_pipeline(tmp_path: Path):
    from arclm import Dataset

    path = _write_jsonl(
        tmp_path / "records.jsonl",
        [
            {"text": " alpha   beta ", "label": "keep"},
            {"text": ""},
            {"text": "alpha beta", "label": "duplicate"},
            {"text": "gamma delta", "label": None},
        ],
    )

    dataset = Dataset.load(path).clean(
        remove_empty=True,
        deduplicate=True,
        normalize_whitespace=True,
    )
    split = dataset.split(train=0.5, validation=0.5, shuffle=False)
    report = dataset.inspect()

    assert dataset.text() == "alpha beta\ngamma delta"
    assert split.to_dict() == {"train": 1, "validation": 1, "test": 0}
    assert report.fields == ["label", "text"]
    assert report.missing_values["label"] == 1
    assert report.splits == {"train": 1, "validation": 1}


def test_custom_transformation_works():
    from arclm import Dataset

    class PrefixTransform:
        def __call__(self, record):
            record["text"] = f"research {record['text']}"
            return record

    dataset = Dataset.load([{"text": "alpha"}, {"text": "beta"}]).map(PrefixTransform())

    assert dataset.text() == "research alpha\nresearch beta"
    assert dataset.history[-1]["operation"] == "map"


def test_student_and_professional_apis_produce_compatible_datasets(tmp_path: Path):
    from arclm import Dataset, Lab

    path = _write_jsonl(
        tmp_path / "student-professional.jsonl",
        [{"text": "alpha beta gamma delta alpha beta gamma delta"}],
    )
    lab = Lab()
    student = lab.dataset(path)
    professional = Dataset.load(path)
    prepared = lab.prepare(student, block_size=4, batch_size=1, shuffle=False)

    assert type(student) is type(professional)
    assert student.text() == professional.text()
    assert prepared.tokenizer.get_vocab_size() > 0
    assert lab.decisions[-1]["stage"] == "student.data.prepare"


def test_dataset_can_be_consumed_by_existing_trainer(tmp_path: Path):
    from arclm import Dataset, Lab, Runtime, Trainer

    dataset = Dataset.load(
        [
            {"text": "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"},
            {"text": "epsilon zeta eta theta epsilon zeta eta theta"},
        ]
    ).clean(normalize_whitespace=True)
    model = Lab(runtime=Runtime.auto(prefer="cpu")).model(data=dataset, size="tiny")
    history = Trainer(
        model=model,
        dataset=dataset,
        epochs=1,
        steps=1,
        batch_size=2,
        block_size=4,
        shuffle=False,
        artifact_path=tmp_path / "unused.arcmodel",
    ).train()

    assert history["global_steps"] == 1
    assert history["stopping_reason"] == "completed"
