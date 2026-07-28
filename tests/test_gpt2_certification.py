import json

import pytest
import torch

from arclm import DataPipeline, train_sft
from arclm.models import load_model


pytestmark = [pytest.mark.integration, pytest.mark.hf, pytest.mark.transformers, pytest.mark.slow]


def test_tiny_gpt2_certified_cpu_inference_sft_save_reload(tmp_path):
    source = "hf-internal-testing/tiny-random-gpt2"
    records = [{"prompt": "Hello", "completion": "Hi"}]
    processed, report = (
        DataPipeline()
        .validate("prompt_completion", strict=True)
        .format_prompt_completion()
        .run(records)
    )
    assert report.is_valid
    assert "Prompt:" in processed[0]["text"]

    torch.manual_seed(123)
    bundle = load_model(source, device="cpu", trust_remote_code=False)
    first = bundle.predict("Hello", max_new_tokens=3, do_sample=False)
    torch.manual_seed(123)
    second = bundle.predict("Hello", max_new_tokens=3, do_sample=False)
    assert first == second
    assert bundle.capability_report.causal_lm_compatible

    data_path = tmp_path / "sft.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    output_dir = tmp_path / "sft-out"
    result = train_sft(
        model=source,
        dataset=str(data_path),
        output_dir=str(output_dir),
        batch_size=1,
        max_length=32,
        max_steps=1,
        num_epochs=1,
        trust_remote_code=False,
        save_tokenizer=True,
    )
    assert result.steps == 1
    assert output_dir.exists()

    reloaded = load_model(output_dir, device="cpu", trust_remote_code=False)
    after_reload = reloaded.predict("Hello", max_new_tokens=3, do_sample=False)
    assert isinstance(after_reload, str)
