from __future__ import annotations

from pathlib import Path


TEXT = "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"


def _seed(value: int = 1234) -> None:
    import torch

    torch.manual_seed(value)


def _dataset():
    from arclm import Dataset

    return Dataset.load([{"text": TEXT}])


def _built_tokenizer(dataset):
    from arclm import Tokenizer

    return Tokenizer(strategy="word", max_vocab=64).build(dataset.text())


def _native_model(dataset, *, seed: int = 1234):
    from arclm import Model, Runtime

    _seed(seed)
    return Model.create(
        architecture="arclm-native",
        tokenizer=_built_tokenizer(dataset),
        runtime=Runtime.auto(prefer="cpu"),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        batch_size=1,
    )


def test_student_tiny_pretraining_workflow():
    from arclm import Lab, Runtime

    _seed()
    lab = Lab(runtime=Runtime.auto(prefer="cpu"))
    dataset = lab.dataset(text=TEXT)
    model = lab.model(data=dataset, size="tiny")
    history = lab.train(epochs=1, steps=1, batch_size=1, block_size=4, shuffle=False)

    assert model.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert history["global_steps"] == 1
    assert lab.inspect()["trainer"]["history"]["global_steps"] == 1


def test_professional_explicit_training_workflow(tmp_path: Path):
    from arclm import Dataset, Trainer
    from arclm.training import FineTuningConfig, TrainingConfig

    dataset = _dataset()
    validation = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    model = _native_model(dataset)
    trainer = Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        config=TrainingConfig(
            epochs=1,
            steps_per_epoch=1,
            batch_size=1,
            block_size=4,
            learning_rate=1e-3,
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_interval=1,
            shuffle=False,
        ),
        fine_tuning=FineTuningConfig(method="pretrain"),
    )
    history = trainer.train()

    assert trainer.make_plan().to_dict()["steps_per_epoch"] == 1
    assert history["global_steps"] == 1
    assert history["validation_losses"]
    assert Path(history["checkpoints"][0]).exists()


def test_research_custom_component_workflow():
    from arclm.research import BaseStrategy, NextTokenLoss, Trainer, TrainingConfig

    class OffsetLoss:
        def __init__(self):
            self.calls = 0
            self.base = NextTokenLoss()

        def __call__(self, *, model, batch, engine):
            self.calls += 1
            return self.base(model=model, batch=batch, engine=engine)

    class ResearchStrategy(BaseStrategy):
        def __init__(self, loss):
            super().__init__(name="research_e2e", loss_function=loss)

    dataset = _dataset()
    model = _native_model(dataset)
    loss = OffsetLoss()
    history = Trainer(
        model=model,
        dataset=dataset,
        config=TrainingConfig(epochs=1, steps_per_epoch=1, batch_size=1, block_size=4, shuffle=False),
        strategy=ResearchStrategy(loss),
    ).train()

    assert history["strategy"] == "research_e2e"
    assert history["global_steps"] == 1
    assert loss.calls == 1


def test_data_template_training_workflow():
    from arclm import ChatTemplate, Dataset, Model, Runtime, Tokenizer, Trainer

    template = ChatTemplate(
        user="[USER] {content} [/USER]",
        assistant="[ASSISTANT] {content} [/ASSISTANT]",
        separator="\n",
    )
    dataset = Dataset.load(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]
    )
    tokenizer = Tokenizer(strategy="word", max_vocab=64, chat_template=template)
    prepared = dataset.prepare(tokenizer=tokenizer, block_size=4, batch_size=1, padding=True, shuffle=False)
    _seed()
    model = Model.create(
        architecture="arclm-native",
        tokenizer=prepared.tokenizer,
        runtime=Runtime.auto(prefer="cpu"),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        batch_size=1,
    )
    history = Trainer(model=model, dataset=dataset, epochs=1, steps=1, batch_size=1, block_size=4, shuffle=False).train()

    assert "[USER] Hello [/USER]" in prepared.model_inputs.text
    assert "[ASSISTANT] Hi there [/ASSISTANT]" in prepared.model_inputs.text
    assert history["global_steps"] == 1


def test_artifact_save_load_round_trip(tmp_path: Path):
    from arclm import Model, Runtime

    dataset = _dataset()
    model = _native_model(dataset)
    artifact = model.save(tmp_path / "roundtrip.arcmodel", overwrite=True)
    loaded = Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))

    assert artifact.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert loaded.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert loaded.tokenizer.decode(loaded.tokenizer.encode("alpha beta")) == "alpha beta"


def test_fine_tuning_adapter_round_trip(tmp_path: Path):
    from arclm import Trainer

    dataset = _dataset()
    model = _native_model(dataset, seed=77)
    history = Trainer(
        model=model,
        dataset=dataset,
        method="lora",
        rank=2,
        alpha=4.0,
        target_modules=("head",),
        epochs=1,
        steps=1,
        batch_size=1,
        block_size=4,
        save_adapter=True,
        adapter_path=tmp_path / "tiny-lora",
        shuffle=False,
    ).fine_tune()

    fresh = _native_model(dataset, seed=77)
    manifest = fresh.load_adapter(history["adapter"])

    assert Path(history["adapter"]).exists()
    assert manifest.adapter_type == "lora"
    assert manifest.target_modules == ["head"]
    assert any(".lora_" in name for name, _ in fresh.model.named_parameters())
