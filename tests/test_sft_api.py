import json

import pytest

from arclm import SFTTrainingResult, train_sft
from arclm.sft import _find_subsequence, _load_sft_records


def test_train_sft_is_public_api():
    assert callable(train_sft)
    assert SFTTrainingResult.__name__ == "SFTTrainingResult"


def test_train_sft_rejects_unimplemented_backend(tmp_path):
    with pytest.raises(ValueError, match="backend='huggingface'"):
        train_sft(
            model="dummy",
            dataset=str(tmp_path / "missing.jsonl"),
            output_dir=str(tmp_path / "out"),
            backend="arclm",
        )


def test_load_sft_records_supports_messages_jsonl(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "What is SFT?"},
                    {"role": "assistant", "content": "Instruction fine-tuning."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = _load_sft_records(str(path))

    assert records[0][-1]["role"] == "assistant"
    assert records[0][-1]["content"] == "Instruction fine-tuning."


def test_find_subsequence():
    assert _find_subsequence([1, 2, 3, 2, 3], [2, 3], start=2) == 3
    assert _find_subsequence([1, 2, 3], [4]) is None
