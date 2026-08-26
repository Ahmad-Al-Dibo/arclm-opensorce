from __future__ import annotations

from pathlib import Path


def _tiny_model_and_data(tmp_path: Path):
    from arclm import Dataset, Lab, Runtime

    data_path = tmp_path / "train.txt"
    data_path.write_text("alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta", encoding="utf-8")
    dataset = Dataset.load(data_path)
    model = Lab(runtime=Runtime.auto(prefer="cpu")).create(size="tiny", data=dataset)
    return model, dataset


def test_small_training_smoke(tmp_path: Path):
    from arclm import Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    history = Trainer(model=model, dataset=dataset, epochs=1, batch_size=2, block_size=4, max_steps=2, shuffle=False).train()

    assert history["global_steps"] == 2
    assert history["optimizer_steps"] == 2
    assert history["train_losses"]
    assert history["tokens_processed"] > 0
    assert history["samples_processed"] > 0


def test_steps_are_enforced_per_epoch(tmp_path: Path):
    from arclm import Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    history = Trainer(
        model=model,
        dataset=dataset,
        epochs=3,
        steps=5,
        batch_size=2,
        block_size=4,
        shuffle=False,
    ).train()

    assert history["global_steps"] == 15
    assert len(history["steps"]) == 15
    assert len(history["train_losses"]) == 3
    assert [step["epoch_step"] for step in history["steps"][:5]] == [1, 2, 3, 4, 5]
    assert [step["epoch"] for step in history["steps"][10:]] == [3, 3, 3, 3, 3]
    assert history["stopping_reason"] == "completed"


def test_max_steps_is_global_safety_cap_not_epoch_steps(tmp_path: Path):
    from arclm import Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    history = Trainer(
        model=model,
        dataset=dataset,
        epochs=3,
        steps_per_epoch=5,
        max_steps=7,
        batch_size=2,
        block_size=4,
        shuffle=False,
    ).train()

    assert history["global_steps"] == 7
    assert len(history["steps"]) == 7
    assert history["stopping_status"] == "stopped"
    assert history["stopping_reason"] == "max_steps"


def test_early_stopping_is_configurable(tmp_path: Path):
    from arclm import Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    history = Trainer(
        model=model,
        dataset=dataset,
        epochs=5,
        steps_per_epoch=1,
        batch_size=2,
        block_size=4,
        early_stopping=True,
        early_stopping_patience=0,
        early_stopping_min_delta=999.0,
        early_stopping_metric="train_loss",
        shuffle=False,
    ).train()

    assert history["global_steps"] == 2
    assert history["stopping_status"] == "stopped"
    assert history["stopping_reason"] == "early_stopping"


def test_progress_display_is_clean_and_opt_in(tmp_path: Path, capsys):
    from arclm import Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    Trainer(
        model=model,
        dataset=dataset,
        epochs=1,
        steps=1,
        batch_size=2,
        block_size=4,
        show_progress=True,
        shuffle=False,
    ).train()
    output = capsys.readouterr().err

    assert "Epoch 1 | Step 1/1 | Loss" in output
    assert "Progress 100.0%" in output
    assert "Training completed | Reason completed" in output
    assert "Epoch 1/1, Step" not in output


def test_validation_loss_tracking(tmp_path: Path):
    from arclm import Dataset, Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    validation = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    history = Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        epochs=1,
        batch_size=2,
        block_size=4,
        max_steps=1,
        shuffle=False,
    ).train()

    assert len(history["validation_losses"]) == 1
    assert history["validation_losses"][0] is not None
    assert history["metrics"]["validation_epoch_1"]["loss"] == history["validation_losses"][0]


def test_checkpoint_resume(tmp_path: Path):
    from arclm import Lab, Runtime, Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    first = Trainer(
        model=model,
        dataset=dataset,
        epochs=1,
        batch_size=2,
        block_size=4,
        max_steps=1,
        checkpoint_interval=1,
        checkpoint_dir=checkpoint_dir,
        shuffle=False,
    ).train()

    resumed_model = Lab(runtime=Runtime.auto(prefer="cpu")).create(size="tiny", data=dataset)
    second = Trainer(
        model=resumed_model,
        dataset=dataset,
        epochs=2,
        batch_size=2,
        block_size=4,
        max_steps=2,
        checkpoint_dir=checkpoint_dir,
        resume_from=first["checkpoints"][0],
        shuffle=False,
    ).train()

    assert Path(first["checkpoints"][0]).exists()
    assert second["resumed_from"] == first["checkpoints"][0]
    assert second["global_steps"] == 2


def test_fine_tuning_configuration_plan(tmp_path: Path):
    from arclm import Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    trainer = Trainer(
        model=model,
        dataset=dataset,
        method="lora",
        rank=2,
        alpha=4.0,
        freeze_base_model=True,
        target_modules=("head",),
    )
    plan = trainer.make_plan().to_dict()

    assert plan["strategy"] == "adapter"
    assert plan["fine_tuning"]["rank"] == 2
    assert plan["fine_tuning"]["alpha"] == 4.0
    assert plan["fine_tuning"]["target_modules"] == ["head"]


def test_lora_fine_tuning_freezes_base_parameters(tmp_path: Path):
    from arclm import Trainer

    model, dataset = _tiny_model_and_data(tmp_path)
    history = Trainer(
        model=model,
        dataset=dataset,
        method="lora",
        rank=2,
        alpha=4.0,
        target_modules=("head",),
        epochs=1,
        batch_size=2,
        block_size=4,
        max_steps=1,
        shuffle=False,
    ).train()

    trainable = history["metrics"]["trainable"]
    assert trainable["trainable_parameters"] > 0
    assert trainable["frozen_parameters"] > 0
    assert all("lora_" in name for name in trainable["trainable_names"])


def test_training_can_save_model_artifact(tmp_path: Path):
    from arclm import Trainer
    from arclm.artifacts import ArcModelArtifact

    model, dataset = _tiny_model_and_data(tmp_path)
    artifact_path = tmp_path / "trained.arcmodel"
    history = Trainer(
        model=model,
        dataset=dataset,
        epochs=1,
        batch_size=2,
        block_size=4,
        max_steps=1,
        save_artifact=True,
        artifact_path=artifact_path,
        shuffle=False,
    ).train()
    report = ArcModelArtifact(artifact_path).inspect()

    assert history["artifact"] == str(artifact_path)
    assert artifact_path.exists()
    assert report["training_metadata"]["plan"]["strategy"] == "pretrain"
    assert report["training_metadata"]["history"]["global_steps"] == 1


def test_training_callbacks_receive_events(tmp_path: Path):
    from arclm import Dataset, Trainer

    class Recorder:
        def __init__(self):
            self.names = []

        def on_train_start(self, event):
            self.names.append(event.name)

        def on_step_end(self, event):
            self.names.append(event.name)

        def on_validation_end(self, event):
            self.names.append(event.name)

        def on_checkpoint(self, event):
            self.names.append(event.name)

        def on_train_end(self, event):
            self.names.append(event.name)

    model, dataset = _tiny_model_and_data(tmp_path)
    validation = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    recorder = Recorder()
    Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        callbacks=[recorder],
        epochs=1,
        batch_size=2,
        block_size=4,
        max_steps=1,
        checkpoint_interval=1,
        checkpoint_dir=tmp_path / "events",
        shuffle=False,
    ).train()

    assert recorder.names == ["train_start", "checkpoint", "step_end", "validation_end", "train_end"]
