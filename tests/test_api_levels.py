from __future__ import annotations

from pathlib import Path


TEXT = "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"


def test_student_lab_api_trains_tiny_task():
    from arclm import Lab

    lab = Lab()
    dataset = lab.dataset(text=TEXT)
    model = lab.model(data=dataset, size="tiny")
    history = lab.train(epochs=1, steps=1, shuffle=False)

    assert model.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert history["global_steps"] == 1
    assert lab.last_trainer is not None


def test_professional_api_uses_explicit_config_and_same_trainer(tmp_path: Path):
    import torch

    from arclm import Dataset, Model, Runtime, Tokenizer, Trainer
    from arclm.training import FineTuningConfig, TrainingConfig

    class Recorder:
        def __init__(self):
            self.events = []

        def on_train_start(self, event):
            self.events.append(event.name)

        def on_step_end(self, event):
            self.events.append(event.name)

        def on_validation_end(self, event):
            self.events.append(event.name)

        def on_checkpoint(self, event):
            self.events.append(event.name)

        def on_train_end(self, event):
            self.events.append(event.name)

    runtime = Runtime.auto(prefer="cpu")
    dataset = Dataset.load([{"text": TEXT}])
    validation = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    tokenizer = Tokenizer(strategy="word", max_vocab=16).build(dataset.text())
    model = Model.create(
        architecture="arclm-native",
        tokenizer=tokenizer,
        runtime=runtime,
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        batch_size=2,
    )
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        block_size=4,
        steps_per_epoch=1,
        learning_rate=5e-4,
        validation_interval=1,
        checkpoint_dir=tmp_path / "checkpoints",
        save_artifact=True,
        artifact_path=tmp_path / "professional.arcmodel",
        shuffle=False,
    )
    recorder = Recorder()
    trainer = Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        config=config,
        fine_tuning=FineTuningConfig(method="pretrain"),
        callbacks=[recorder],
    )
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate)
    history = trainer.train(optimizer=optimizer)

    assert trainer.plan.to_dict()["steps_per_epoch"] == 1
    assert history["global_steps"] == 1
    assert history["validation_losses"]
    assert Path(history["artifact"]).exists()
    assert recorder.events == ["train_start", "step_end", "validation_end", "checkpoint", "train_end"]


def test_research_api_replaces_loss_and_strategy_on_same_engine():
    from arclm import Dataset, Lab
    from arclm.research import BaseStrategy, NextTokenLoss, TrainingConfig, Trainer

    class HalfScaleLoss:
        def __init__(self):
            self.calls = 0
            self.base = NextTokenLoss()

        def __call__(self, *, model, batch, engine):
            self.calls += 1
            return self.base(model=model, batch=batch, engine=engine) * 0.5

    class MyTrainingStrategy(BaseStrategy):
        def __init__(self, loss):
            super().__init__(name="research_half_loss", loss_function=loss)

    dataset = Dataset.load([{"text": TEXT}])
    model = Lab().create(size="tiny", data=dataset)
    loss = HalfScaleLoss()
    strategy = MyTrainingStrategy(loss)
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=TrainingConfig(epochs=1, batch_size=2, block_size=4, steps_per_epoch=1, shuffle=False),
        strategy=strategy,
    )
    history = trainer.train()

    assert trainer.plan.to_dict()["strategy"] == "research_half_loss"
    assert history["strategy"] == "research_half_loss"
    assert history["global_steps"] == 1
    assert loss.calls == 1
