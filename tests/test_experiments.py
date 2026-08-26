from __future__ import annotations

from pathlib import Path


def test_evaluation_report_creation():
    from arclm import EvaluationEngine

    report = EvaluationEngine().evaluate(
        history={
            "train_losses": [2.0, 1.5],
            "validation_losses": [1.25],
            "tokens_processed": 100,
            "samples_processed": 10,
            "global_steps": 2,
            "elapsed_time": 5.0,
        }
    )

    assert report.metrics["training_loss"] == 1.5
    assert report.metrics["validation_loss"] == 1.25
    assert report.metrics["perplexity"] > 1
    assert report.metrics["tokens_per_second"] == 20.0


def test_custom_evaluator_plugin():
    from arclm import EvaluationEngine

    class AccuracyEvaluator:
        name = "toy_accuracy"

        def evaluate(self, context):
            return {"accuracy": 0.75, "steps": context.history["global_steps"]}

    report = EvaluationEngine().evaluate(history={"global_steps": 3}, evaluators=[AccuracyEvaluator()])

    assert report.custom_metrics["toy_accuracy"] == {"accuracy": 0.75, "steps": 3}


def test_experiment_metadata_round_trip(tmp_path: Path):
    from arclm import Dataset, Experiment, Lab, Runtime

    dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    model = Lab(runtime=Runtime.auto(prefer="cpu")).model(data=dataset, size="tiny")
    history = {
        "strategy": "adapter",
        "train_losses": [1.0],
        "validation_losses": [0.9],
        "global_steps": 1,
        "tokens_processed": 8,
        "samples_processed": 2,
        "elapsed_time": 1.0,
        "artifact": str(tmp_path / "model.arcmodel"),
    }
    experiment = Experiment("round-trip", root=tmp_path, seed=123).capture(model=model, dataset=dataset, history=history)
    saved = experiment.save()
    loaded = Experiment.load(saved)
    payload = loaded.to_dict()

    assert payload["name"] == "round-trip"
    assert payload["seed"] == 123
    assert payload["dataset"]["fingerprint"]
    assert payload["reproducibility"]["model_fingerprint"]
    assert payload["evaluation"]["metrics"]["validation_loss"] == 0.9
    assert payload["artifacts"]["model"].endswith("model.arcmodel")


def test_two_experiment_results_can_be_compared(tmp_path: Path):
    from arclm import Experiment

    first = Experiment("first", root=tmp_path).capture(
        history={
            "train_losses": [1.2],
            "validation_losses": [1.1],
            "tokens_processed": 100,
            "elapsed_time": 10.0,
            "metrics": {"trainable": {"trainable_parameters": 12}},
        }
    )
    second = Experiment("second", root=tmp_path).capture(
        history={
            "train_losses": [0.8],
            "validation_losses": [0.7],
            "tokens_processed": 120,
            "elapsed_time": 6.0,
            "metrics": {"trainable": {"trainable_parameters": 8}},
        }
    )
    first.save()
    second.save()

    comparison = Experiment.compare("first", "second", root=tmp_path)

    assert comparison.experiments == ["first", "second"]
    assert comparison.rows[0]["validation_loss"] == 1.1
    assert comparison.rows[1]["validation_loss"] == 0.7
    assert comparison.rows[1]["tokens_per_second"] == 20.0
    assert comparison.rows[0]["trainable_parameters"] == 12


def test_inspectors_create_structured_reports(tmp_path: Path):
    from arclm import Dataset, DatasetInspector, Lab, ModelInspector, Runtime, RuntimeInspector, TrainingInspector

    dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    model = Lab(runtime=Runtime.auto(prefer="cpu")).model(data=dataset, size="tiny")
    history = {"strategy": "pretrain", "train_losses": [1.0], "global_steps": 1, "elapsed_time": 0.5}

    assert DatasetInspector().inspect(dataset).summary["records"] == 1
    assert ModelInspector().inspect(model).summary["architecture_id"] == "arclm-native-causal-lm"
    assert RuntimeInspector().inspect(model.runtime).summary["device"] == "cpu"
    assert TrainingInspector().inspect(history).summary["training_loss"] == 1.0
