import json

import pytest

from arclm.cache import clear_cache, inspect_cache
from arclm.tokenization import tokenize_dataset


class FakeTokenizer:
    name_or_path = "fake-tokenizer"
    eos_token = "<eos>"
    bos_token = "<bos>"

    def __call__(self, text, max_length=None, truncation=True, padding=False):
        ids = [len(token) for token in str(text).split()]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.mark.integration
def test_tokenize_prompt_completion_with_cache_hit(tmp_path):
    records = [{"prompt": "Say hello", "completion": "Hello"}]
    cache_dir = tmp_path / "cache"

    first = tokenize_dataset(
        records,
        tokenizer=FakeTokenizer(),
        schema="prompt_completion",
        max_length=8,
        cache_dir=str(cache_dir),
        prompt_masking=True,
    )
    second = tokenize_dataset(
        records,
        tokenizer=FakeTokenizer(),
        schema="prompt_completion",
        max_length=8,
        cache_dir=str(cache_dir),
        prompt_masking=True,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert second.records[0]["labels"][0] == -100
    assert inspect_cache(cache_dir).entries == 1


def test_cache_clear_and_corrupted_entry(tmp_path):
    cache_dir = tmp_path / "cache"
    result = tokenize_dataset([{"text": "hello"}], tokenizer=FakeTokenizer(), cache_dir=str(cache_dir))
    metadata = cache_dir / result.cache_key / "metadata.json"
    metadata.write_text("{bad", encoding="utf-8")

    stats = inspect_cache(cache_dir)
    assert result.cache_key in stats.corrupted

    cleared = clear_cache(cache_dir)
    assert cleared.entries == 0
