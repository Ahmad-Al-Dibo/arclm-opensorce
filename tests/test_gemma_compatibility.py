from __future__ import annotations

from pathlib import Path


TEXT = "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"


def test_model_load_resolves_tiny_gemma_compatibility_fixture():
    from arclm import Model, Runtime

    model = Model.load("arclm://compat/gemma-tiny", runtime=Runtime.auto(prefer="cpu"))
    report = model.inspect()

    assert report["architecture_id"] == "gemma-transformers-causal-lm"
    assert report["architecture_kind"] == "compatibility"
    assert report["config"]["model_family"] == "gemma"
    assert report["base_model_id"] == "arclm://compat/gemma-tiny"


def test_gemma_architecture_mapping_is_registered():
    from arclm import ArchitectureKind, architectures

    architecture = architectures.resolve({"model_type": "gemma"})

    assert architecture.architecture_id == "gemma-transformers-causal-lm"
    assert architecture.kind == ArchitectureKind.COMPATIBILITY
    assert architecture.adapter_targets == ("q_proj", "v_proj")


def test_adapter_targets_resolve_from_gemma_architecture_metadata():
    from arclm import Model, Runtime

    model = Model.load("arclm://compat/gemma-tiny", runtime=Runtime.auto(prefer="cpu"))
    attached = model.attach_lora(rank=2, alpha=4.0)

    assert attached
    assert all(name.endswith(("q_proj", "v_proj")) for name in attached)
    assert set(model.architecture.metadata().adapter_targets) == {"q_proj", "v_proj"}


def test_gemma_lora_fine_tuning_uses_shared_trainer_and_saves_outputs(tmp_path: Path):
    from arclm import Dataset, Model, Runtime, Trainer

    dataset = Dataset.load([{"text": TEXT}])
    validation = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    model = Model.load("arclm://compat/gemma-tiny", runtime=Runtime.auto(prefer="cpu"))
    trainer = Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        method="lora",
        rank=2,
        alpha=4.0,
        epochs=1,
        steps=1,
        batch_size=1,
        block_size=4,
        checkpoint_interval=1,
        checkpoint_dir=tmp_path / "checkpoints",
        save_adapter=True,
        adapter_path=tmp_path / "gemma.adapter",
        save_artifact=True,
        artifact_path=tmp_path / "gemma.arcmodel",
        shuffle=False,
    )
    plan = trainer.memory_plan()
    history = trainer.fine_tune()

    assert plan["architecture_id"] == "gemma-transformers-causal-lm"
    assert plan["fine_tuning_strategy"] == "lora"
    assert plan["adapter_targets"] == ["q_proj", "v_proj"]
    assert history["strategy"] == "adapter"
    assert history["global_steps"] == 1
    assert Path(history["adapter"]).exists()
    assert Path(history["artifact"]).exists()
    assert history["checkpoints"][0].endswith(".arcckpt")
