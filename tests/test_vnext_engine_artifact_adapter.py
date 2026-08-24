import json

import pytest
import torch

from arclm import ArtifactIntegrityError, Dataset, Model, ModelRegistry, Runtime, Tokenizer, Trainer
from arclm.core import (
    ArcLMNativeArchitecture,
    BackendContract,
    ModelContract,
    ModelSpecContract,
    RuntimeContract,
    TorchBackend,
)
from arclm.exceptions import ModelCompatibilityError


def _tiny_text(tmp_path, repeat=16):
    path = tmp_path / "tiny.txt"
    path.write_text("hello world hello arc model learns hello world " * repeat, encoding="utf-8")
    return path


def _tiny_model(dataset=None, *, seed=1234):
    torch.manual_seed(seed)
    if dataset is None:
        tokenizer = Tokenizer(max_vocab=32)
        tokenizer.build("hello world hello arc model learns")
    else:
        tokenizer = dataset.prepare(block_size=4, batch_size=2, shuffle=False).tokenizer
    return Model.create(
        architecture="arclm-native",
        tokenizer=tokenizer,
        runtime=Runtime.auto(prefer="cpu"),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        dropout=0.0,
        batch_size=2,
        learning_rate=5e-2,
    )


def test_training_engine_reduces_loss_scheduler_checkpoint_and_skips_legacy(tmp_path, monkeypatch):
    dataset = Dataset.load(_tiny_text(tmp_path))
    model = _tiny_model(dataset)
    calls = []

    class BombLegacy:
        def __init__(self, *args, **kwargs):
            raise AssertionError("high-level Trainer delegated to legacy Trainer")

    import arclm.vnext.trainer as vtrainer

    monkeypatch.setattr(vtrainer, "LegacyTrainer", BombLegacy)
    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=3,
        batch_size=2,
        learning_rate=5e-2,
        block_size=4,
        checkpoint_interval=1,
    )
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=5e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    history = trainer.train(
        mode="pretrain",
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_hook=lambda engine, event: calls.append(dict(event)),
    )

    assert history["optimizer_steps"] > 0
    assert history["scheduler_steps"] == 3
    assert len(history["train_losses"]) == 3
    assert history["train_losses"][-1] <= history["train_losses"][0]
    assert any(call["event"] == "step" for call in calls)
    assert any(call["event"] == "epoch" for call in calls)


def test_core_contracts_registry_resolution_architecture_and_backend(tmp_path):
    runtime = Runtime.auto(prefer="cpu")
    spec = ModelRegistry.resolve({"architecture_id": "arclm-native-causal-lm"})
    architecture = ArcLMNativeArchitecture()
    model = _tiny_model()
    backend = TorchBackend()

    assert isinstance(runtime, RuntimeContract)
    assert isinstance(spec, ModelSpecContract)
    assert isinstance(model, ModelContract)
    assert isinstance(backend, BackendContract)
    assert spec.supports("pretraining") == "SUPPORTED"
    assert spec.supports("lora") == "SUPPORTED"
    assert architecture.build(model.config, runtime).block_size == model.config.block_size


def test_arcmodel_safetensors_layouts_round_trip_and_true_multishard(tmp_path):
    model = _tiny_model()
    before = model.generate("hello", max_new_tokens=2)

    for layout in ("single", "directory"):
        artifact = model.save(tmp_path / layout, layout=layout, overwrite=True)
        info = artifact.inspect()
        loaded = Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))

        assert info["layout"] == layout
        assert all(not name.endswith(".pt") for name in info["weight_files"])
        assert loaded.generate("hello", max_new_tokens=2) == before
        for name, tensor in model.model.state_dict().items():
            assert torch.equal(tensor.cpu(), loaded.model.state_dict()[name].cpu())

    sharded = model.save(tmp_path / "sharded", layout="sharded", shard_size=256, overwrite=True)
    index = json.loads((sharded.path / "weights" / "index.json").read_text(encoding="utf-8"))
    loaded = Model.load(sharded.path, runtime=Runtime.auto(prefer="cpu"))

    assert len(index["shards"]) > 1
    assert set(index["weight_map"]) == set(model.model.state_dict())
    assert loaded.generate("hello", max_new_tokens=2) == before


def test_arcmodel_detects_corrupted_shard(tmp_path):
    model = _tiny_model()
    artifact = model.save(tmp_path / "sharded", layout="sharded", shard_size=256, overwrite=True)
    shard = next((artifact.path / "weights").glob("shard-*.safetensors"))

    with shard.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"X")

    with pytest.raises(ArtifactIntegrityError):
        Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))


def test_native_pretrained_family_registry_weight_mapping_and_generation(tmp_path):
    model = _tiny_model()
    artifact = model.save(tmp_path / "base.arcmodel", layout="single", overwrite=True)
    manifest = artifact.manifest()
    spec = ModelRegistry.resolve({"architecture_id": manifest.architecture_id})
    loaded = Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))

    mapped = spec.weight_mapper.map_name("blocks.0.attn.query.weight")
    assert mapped == "blocks.0.attn.query.weight"
    assert loaded.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert loaded.generate("hello", max_new_tokens=1)


def test_lora_attach_tiny_finetune_arcadapter_reload_and_mismatch_rejection(tmp_path):
    dataset = Dataset.load(_tiny_text(tmp_path))
    base = _tiny_model(dataset)
    base_artifact = base.save(tmp_path / "base.arcmodel", overwrite=True)
    model = Model.load(base_artifact.path, runtime=Runtime.auto(prefer="cpu"))
    attached = model.attach_lora(rank=2, alpha=4.0)

    assert attached
    assert all((".lora_" in name) == parameter.requires_grad for name, parameter in model.model.named_parameters())

    trainer = Trainer(model=model, dataset=dataset, epochs=1, batch_size=2, learning_rate=1e-2, block_size=4)
    history = trainer.train(mode="adapter")
    adapter_manifest = model.save_adapter(tmp_path / "tiny.arcadapter", overwrite=True)

    assert history["optimizer_steps"] > 0
    assert adapter_manifest.adapter_type == "lora"

    reloaded = Model.load(base_artifact.path, runtime=Runtime.auto(prefer="cpu"))
    reloaded.load_adapter(tmp_path / "tiny.arcadapter")
    assert reloaded.generate("hello", max_new_tokens=1)

    incompatible = _tiny_model(seed=999)
    with pytest.raises(ModelCompatibilityError):
        incompatible.load_adapter(tmp_path / "tiny.arcadapter")
