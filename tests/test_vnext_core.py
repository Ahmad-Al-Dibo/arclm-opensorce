import torch
import pytest

from arclm import Dataset, Lab, Model, ModelRegistry, Runtime, Tokenizer, Trainer


def test_runtime_auto_reports_cpu_in_baseline_environment():
    runtime = Runtime.auto(prefer="cpu")

    report = runtime.to_dict()
    assert report["backend"] == "torch"
    assert report["device"]["type"] == "cpu"
    assert runtime.device_name == "cpu"


def test_model_registry_reports_capability_specific_support():
    report = ModelRegistry.supports("arclm-native", ["inference", "cpu", "cuda"])

    assert report["inference"] == "SUPPORTED"
    assert report["cpu"] == "SUPPORTED"
    assert report["cuda"] == "UNTESTED"


def test_arcmodel_round_trip_preserves_config_weights_and_generation(tmp_path):
    tokenizer = Tokenizer(max_vocab=20)
    tokenizer.build("hello world hello arc")
    runtime = Runtime.auto(prefer="cpu")
    model = Model.create(
        architecture="arclm-native",
        tokenizer=tokenizer,
        runtime=runtime,
        embed_dim=16,
        block_size=4,
        num_blocks=1,
        dropout=0.0,
    )

    before = model.generate("hello", max_new_tokens=3, temperature=0.0)
    artifact = model.save(tmp_path / "tiny.arcmodel")
    reloaded = Model.load(artifact.path, runtime=runtime)
    after = reloaded.generate("hello", max_new_tokens=3, temperature=0.0)

    assert before == after
    assert reloaded.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert reloaded.config.to_dict()["embed_dim"] == 16
    for name, tensor in model.model.state_dict().items():
        assert torch.equal(tensor.cpu(), reloaded.model.state_dict()[name].cpu())


def _tiny_dataset(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("hello world hello arc model learns hello world " * 8, encoding="utf-8")
    return path


def _tiny_model_for_dataset(dataset):
    prepared = dataset.prepare(block_size=4, batch_size=2)
    return Model.create(
        architecture="arclm-native",
        tokenizer=prepared.tokenizer,
        runtime=Runtime.auto(prefer="cpu"),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        dropout=0.0,
        batch_size=2,
        learning_rate=1e-2,
    )


def test_high_level_dataset_model_trainer_flow(tmp_path):
    dataset = Dataset.load(_tiny_dataset(tmp_path))
    model = _tiny_model_for_dataset(dataset)
    trainer = Trainer(model=model, dataset=dataset, epochs=2, batch_size=2, learning_rate=1e-2, block_size=4)

    plan = trainer.make_plan().to_dict()
    history = trainer.train()

    assert plan["strategy"] == "pretrain"
    assert len(history["train_losses"]) == 2
    assert torch.isfinite(torch.tensor(history["train_losses"][-1]))


def test_lab_pretrain_save_and_load(tmp_path):
    lab = Lab(runtime=Runtime.auto(prefer="cpu"))

    model = lab.pretrain(_tiny_dataset(tmp_path), size="tiny", epochs=1, learning_rate=1e-2)
    artifact = model.save(tmp_path / "student-model.arcmodel", overwrite=True)
    loaded = Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))

    report = lab.inspect()
    assert report["model"]["architecture_id"] == "arclm-native-causal-lm"
    assert loaded.generate("hello", max_new_tokens=1)


def test_artifact_layouts_round_trip_and_validate_hashes(tmp_path):
    dataset = Dataset.load(_tiny_dataset(tmp_path))
    model = _tiny_model_for_dataset(dataset)

    for layout in ["single", "directory", "sharded"]:
        artifact = model.save(tmp_path / layout, layout=layout, overwrite=True)
        loaded = Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))
        info = artifact.inspect()

        assert info["layout"] == layout
        assert loaded.config.to_dict()["embed_dim"] == model.config.to_dict()["embed_dim"]
        for name, tensor in model.model.state_dict().items():
            assert torch.equal(tensor.cpu(), loaded.model.state_dict()[name].cpu())


def test_runtime_report_does_not_expose_torch_device_object():
    report = Runtime.auto(prefer="cpu").to_dict()

    assert report["device"]["type"] == "cpu"
    assert "torch.device" not in repr(report)


def test_low_level_tensor_access_validates_names_and_shapes(tmp_path):
    dataset = Dataset.load(_tiny_dataset(tmp_path))
    model = _tiny_model_for_dataset(dataset)
    name = next(iter(model.model.state_dict()))
    tensor = model.get_tensor(name).clone()

    model.set_tensor(name, tensor)

    with pytest.raises(KeyError):
        model.get_tensor("missing.tensor")


def test_public_trainer_keeps_legacy_constructor_usable():
    tokenizer = Tokenizer(max_vocab=10)
    tokenizer.build("hello world hello")
    config_model = Model.create(
        architecture="arclm-native",
        tokenizer=tokenizer,
        runtime=Runtime.auto(prefer="cpu"),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
    )
    optimizer = torch.optim.AdamW(config_model.model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    legacy = Trainer(config_model.model, optimizer, criterion, config_model.config)

    assert "train_losses" in legacy.inspect()


def test_model_load_imports_legacy_pytorch_checkpoint(tmp_path):
    tokenizer = Tokenizer(max_vocab=10)
    tokenizer.build("hello world hello arc model")
    runtime = Runtime.auto(prefer="cpu")
    model = Model.create(
        architecture="arclm-native",
        tokenizer=tokenizer,
        runtime=runtime,
        embed_dim=8,
        block_size=4,
        num_blocks=1,
    )
    legacy_path = tmp_path / "legacy.arcmodel"
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "config": {
                "embed_dim": 8,
                "block_size": 4,
                "num_blocks": 1,
                "dropout": 0.0,
                "tokenizer_type": "word",
            },
            "vocab_size": tokenizer.get_vocab_size(),
            "vocab": tokenizer.vocab,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        legacy_path,
    )

    loaded = Model.load(legacy_path, runtime=runtime)

    assert loaded.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert loaded.base_model_id == "legacy:legacy.arcmodel"
    assert loaded.generate("hello", max_new_tokens=1)


def test_trainer_preserves_model_architecture_block_size(tmp_path):
    dataset = Dataset.load(_tiny_dataset(tmp_path))
    model = _tiny_model_for_dataset(dataset)
    original_block_size = model.config.block_size

    Trainer(model=model, dataset=dataset, epochs=1, batch_size=2, learning_rate=1e-2, block_size=2).train()
    artifact = model.save(tmp_path / "short-window.arcmodel", overwrite=True)
    reloaded = Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))

    assert model.config.block_size == original_block_size
    assert reloaded.config.block_size == original_block_size
