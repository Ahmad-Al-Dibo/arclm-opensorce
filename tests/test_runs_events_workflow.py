import json

import pytest

from arclm.events import CallbackManager, Event, EventHandler, JSONLMetricsLogger
from arclm.runs import Run, inspect_run, list_runs
from arclm.training.engine import _dataset_to_jsonl
from arclm.workflow import run_workflow


class FailingHandler(EventHandler):
    def on_run_started(self, _event):
        raise RuntimeError("boom")


def test_events_handle_callback_failures_and_jsonl(tmp_path):
    logger = JSONLMetricsLogger(tmp_path / "metrics.jsonl")
    manager = CallbackManager([FailingHandler(), logger])
    manager.emit(Event("run_started", {"ok": True}))

    assert manager.warnings
    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")


def test_run_directory_records_metadata_metrics_and_artifact(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    with Run("example", output_dir=tmp_path / "runs") as run:
        run.log_config({"x": 1}, "data")
        run.log_metric("loss", 1.2, step=1)
        saved = run.save_artifact(artifact)

    metadata = inspect_run(run.path)
    assert metadata["status"] == "completed"
    assert metadata["metrics"][0]["name"] == "loss"
    assert saved.exists()
    assert list_runs(tmp_path / "runs")


def test_training_jsonl_conversion_keeps_text_records_non_empty(tmp_path):
    path = _dataset_to_jsonl([{"text": "hello world"}], tmp_path / "train.jsonl")

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {"prompt": "hello world", "completion": "hello world"}


@pytest.mark.integration
@pytest.mark.transformers
def test_workflow_dry_run_creates_reports(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps({"id": "1", "text": "hello world"}) + "\n" + json.dumps({"id": "2", "text": "hello world"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {"name": "dry", "output_dir": str(tmp_path / "runs")},
                "data": {"path": str(data_path), "format": "jsonl", "schema": "text", "streaming": True},
                "deduplication": {"fields": ["text"]},
                "split": {"train": 0.8, "validation": 0.1, "test": 0.1, "strategy": "hash", "key": "id"},
                "tokenization": {"tokenizer": "hf-internal-testing/tiny-random-gpt2", "cache_dir": str(tmp_path / "cache")},
                "model": {"source": "hf-internal-testing/tiny-random-gpt2", "device": "cpu"},
                "training": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    result = run_workflow(config_path, dry_run=True)
    assert result.status == "dry_run"
    assert (tmp_path / "runs").exists()
    assert any(stage.name == "deduplicate" and stage.status == "passed" for stage in result.stages)
    assert any(stage.name == "model" and stage.status == "passed" for stage in result.stages)
