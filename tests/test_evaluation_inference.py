import pytest

from arclm.evaluation import evaluate
from arclm.exceptions import DatasetError
from arclm.inference import GenerationConfig, generate, stream_generate


class FakeBundle:
    def predict(self, prompt, **_kwargs):
        return f"{prompt} done"


def test_generation_config_validation_and_batched_order():
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)

    result = generate(FakeBundle(), prompts=["a", "b"], config=GenerationConfig(max_new_tokens=2), batch_size=2)
    assert result.outputs == ["a done", "b done"]
    assert len(result.prompt_tokens) == 2
    assert result.latency_seconds >= 0


def test_stream_generate_events():
    events = list(stream_generate(FakeBundle(), prompt="hello", config=GenerationConfig(max_new_tokens=2)))
    assert [event.type for event in events] == ["start", "delta", "completed"]


def test_evaluate_generation_metrics_and_empty_dataset():
    with pytest.raises(DatasetError):
        evaluate(FakeBundle(), [])

    report = evaluate(FakeBundle(), [{"text": "hello"}], metrics=["generation_length", "latency"])
    assert report.total_records == 1
    assert "latency_seconds" in report.metrics
