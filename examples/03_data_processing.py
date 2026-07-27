"""Data processing: load JSONL records and format prompt text."""

from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import DataProcessor


def main():
    rows = [
        {"question": "What is ArcLM?", "answer": "A compact language-model toolkit."},
        {"question": "What is SFT?", "answer": "Training on instruction-response examples."},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qa.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows),
            encoding="utf-8",
        )

        dataset = (
            DataProcessor.load(path)
            .clean()
            .transform(
                format="instruction",
                mapping={"instruction": "question", "output": "answer"},
                template="Question: {instruction}\nAnswer: {output}",
            )
        )

        print(dataset.samples[0]["text"])


if __name__ == "__main__":
    main()
