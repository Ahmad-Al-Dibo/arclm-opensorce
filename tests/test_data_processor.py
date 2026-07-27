import json

from arclm import DataProcessor, Tokenizer


def test_data_processor_loads_jsonl_transforms_and_splits(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {"question": "What is ArcLM?", "context": "A library", "answer": "A compact LM toolkit"},
        {"question": "Use?", "context": "", "answer": "Training"},
        {"question": "", "context": "skip", "answer": "empty"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    dataset = DataProcessor.load(path)
    processed = (
        dataset.clean()
        .filter(lambda row: bool(row.get("question")))
        .transform(
            format="instruction",
            mapping={
                "instruction": "question",
                "input": "context",
                "output": "answer",
            },
            template="{instruction}\n{input}\n{output}",
        )
    )
    splits = processed.split(train=0.5, validation=0.5, test=0.0, seed=1)

    assert processed.samples[0]["text"] == "What is ArcLM?\nA library\nA compact LM toolkit"
    assert len(splits["train"]) == 1
    assert len(splits["validation"]) == 1
    assert splits["test"] == []


def test_data_processor_loads_txt_and_tokenizes(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("arc lm\ntrains data", encoding="utf-8")

    tokenizer = Tokenizer(max_vocab=10)
    tokenizer.build("arc lm trains data")

    processed = DataProcessor.load(path).transform(format="pretraining").tokenize(tokenizer)

    assert processed.samples[0]["text"] == "arc lm"
    assert processed.samples[0]["tokens"] == tokenizer.encode_text("arc lm")
