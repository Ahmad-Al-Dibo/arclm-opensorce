from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_top_level_public_api_is_intentional():
    import arclm
    from arclm import Architecture, ArchitectureCapabilities, ArchitectureKind, ArchitectureRegistry, Dataset, Lab, Model, Runtime, Tokenizer, Trainer, architectures

    assert set(arclm.__all__) == {
        "__version__",
        "Architecture",
        "ArchitectureCapabilities",
        "ArchitectureKind",
        "ArchitectureRegistry",
        "Dataset",
        "Lab",
        "Model",
        "Runtime",
        "Tokenizer",
        "Trainer",
        "architectures",
    }
    assert Architecture.__module__ == "arclm.architectures.base"
    assert ArchitectureCapabilities.__module__ == "arclm.architectures.base"
    assert ArchitectureKind.NATIVE == "native"
    assert ArchitectureRegistry.__module__ == "arclm.architectures.registry"
    assert architectures.get("arclm-native").architecture_id == "arclm-native-causal-lm"
    assert Dataset.__module__.startswith("arclm.datasets")
    assert Lab.__module__ == "arclm.lab"
    assert Model.__module__ == "arclm.models.model"
    assert Runtime.__module__.startswith("arclm.runtime")
    assert Tokenizer.__module__ == "arclm.tokenizers.base"
    assert Trainer.__module__ == "arclm.training.trainer"


def test_removed_legacy_import_paths_are_not_available():
    removed_modules = [
        "arclm.vnext",
        "arclm.trainer",
        "arclm.model",
        "arclm.dataset",
        "arclm.tokenizer",
        "arclm.loaders",
        "arclm.preprocess",
        "arclm.sft",
        "arclm.external_inference",
        "arclm.pipeline",
    ]
    for module_name in removed_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_domain_packages_import_without_cycles():
    modules = [
        "arclm.architectures",
        "arclm.artifacts",
        "arclm.core",
        "arclm.datasets",
        "arclm.finetuning",
        "arclm.models",
        "arclm.research",
        "arclm.runtime",
        "arclm.tokenizers",
        "arclm.training",
    ]
    for module_name in modules:
        importlib.import_module(module_name)


def test_basic_dataset_model_trainer_creation(tmp_path: Path):
    from arclm import Dataset, Lab, Runtime, Trainer

    data_path = tmp_path / "tiny.txt"
    data_path.write_text("alpha beta gamma alpha beta gamma alpha beta gamma", encoding="utf-8")

    dataset = Dataset.load(data_path)
    runtime = Runtime.auto(prefer="cpu")
    lab = Lab(runtime=runtime)
    model = lab.create(size="tiny", data=dataset)
    trainer = Trainer(model=model, dataset=dataset, epochs=1, batch_size=2, block_size=4)

    assert dataset.inspect().records == 1
    assert model.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert model.inspect()["architecture_kind"] == "native"
    assert trainer.make_plan().to_dict()["epochs"] == 1


def test_tokenizer_facade_word_and_character_contracts():
    from arclm import Tokenizer

    word = Tokenizer(strategy="word", max_vocab=16).build("alpha beta alpha")
    assert word.tokenize("alpha beta") == ["alpha", "beta"]
    assert word.decode(word.encode("alpha beta")) == "alpha beta"
    assert word.to_json()["strategy"] == "word"

    restored = Tokenizer.from_json(word.to_json())
    assert restored.decode(restored.encode("alpha beta")) == "alpha beta"

    character = Tokenizer(strategy="character", max_vocab=16).build("abc")
    assert character.tokenize("ab") == ["a", "b"]
    assert character.decode(character.encode("abc")) == "abc"


def test_arcmodel_round_trip(tmp_path: Path):
    from arclm import Dataset, Lab, Model, Runtime

    data_path = tmp_path / "tiny.txt"
    data_path.write_text("one two three one two three one two three", encoding="utf-8")

    dataset = Dataset.load(data_path)
    lab = Lab(runtime=Runtime.auto(prefer="cpu"))
    model = lab.create(size="tiny", data=dataset)
    artifact = model.save(tmp_path / "model.arcmodel", overwrite=True)
    loaded = Model.load(artifact.path, runtime=Runtime.auto(prefer="cpu"))

    assert loaded.inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert loaded.tokenizer is not None
    assert artifact.inspect()["tokenizer_metadata"]["strategy"] == "word"
    assert artifact.inspect()["architecture_version"] == "1"
    assert artifact.inspect()["required_capabilities"] == ["load", "generation"]
    assert artifact.inspect()["architecture_metadata"]["kind"] == "native"


def test_architecture_registry_discovery_capabilities_and_duplicates():
    from arclm import ArchitectureKind
    from arclm.architectures import ArchitectureRegistry, ArcLMNativeCausalLM

    registry = ArchitectureRegistry()
    native = registry.register(ArcLMNativeCausalLM, aliases=("native",))

    assert registry.resolve("native").architecture_id == native.architecture_id
    assert registry.resolve({"model_type": "native"}).architecture_id == native.architecture_id
    assert registry.capabilities("native")["load"] == "supported"
    assert registry.available(kind=ArchitectureKind.NATIVE)[0]["component_slots"]
    assert native.supports_component("head")

    with pytest.raises(ValueError):
        registry.register(ArcLMNativeCausalLM)


def test_architecture_config_validation():
    from arclm.config import Config
    from arclm.architectures import architectures

    with pytest.raises(ValueError, match="vocab_size"):
        architectures.get("arclm-native").validate_config(Config())


def test_custom_architecture_creation_artifact_and_loader(tmp_path: Path):
    from arclm import Architecture, ArchitectureCapabilities, ArchitectureKind, Model, Runtime, Tokenizer, architectures
    from arclm.architectures import CapabilitySupport
    from arclm.models.loading import ModelLoader

    class TinyCustomArchitecture(Architecture):
        architecture_id = "test-custom-tiny-causal-lm"
        name = "Test Custom Tiny Causal LM"
        version = "2026.1"
        kind = ArchitectureKind.CUSTOM
        capabilities = ArchitectureCapabilities(
            load=CapabilitySupport.SUPPORTED,
            inference=CapabilitySupport.SUPPORTED,
            generation=CapabilitySupport.SUPPORTED,
            training=CapabilitySupport.EXPERIMENTAL,
        )
        config_schema = {"required": ["vocab_size"], "fields": ["vocab_size", "embed_dim", "block_size", "num_blocks", "dropout"]}
        component_slots = ("token_embedding", "blocks", "head")
        adapter_targets = ("head",)

        def build(self, config, runtime):
            self.validate_config(config)
            from arclm.models.native import ArcLM

            return ArcLM(
                vocab_size=int(config.vocab_size),
                embed_dim=int(config.embed_dim),
                block_size=int(config.block_size),
                num_blocks=int(config.num_blocks),
                dropout=float(config.dropout),
            ).to(runtime.torch_device())

    architecture = architectures.register(TinyCustomArchitecture, aliases=("test-custom-tiny",), replace=True)
    tokenizer = Tokenizer(strategy="character", max_vocab=16).build("abc abc")
    model = Model.create(
        architecture="test-custom-tiny",
        tokenizer=tokenizer,
        runtime=Runtime.auto(prefer="cpu"),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        dropout=0.0,
    )
    artifact = model.save(tmp_path / "custom.arcmodel", overwrite=True)

    loaded = ModelLoader().load(artifact.path, runtime=Runtime.auto(prefer="cpu"))
    report = artifact.inspect()

    assert architecture.kind == "custom"
    assert model.inspect()["architecture_id"] == architecture.architecture_id
    assert isinstance(model.generate("ab", max_new_tokens=1), str)
    assert report["architecture_id"] == architecture.architecture_id
    assert report["architecture_version"] == "2026.1"
    assert report["architecture_metadata"]["component_slots"] == ["token_embedding", "blocks", "head"]
    assert loaded.architecture.architecture_id == architecture.architecture_id
    assert loaded.tokenizer.strategy == "character"


def test_optional_backends_are_isolated_after_import():
    from arclm.core import BackendContract, TorchBackend

    assert isinstance(TorchBackend(), BackendContract)
    script = (
        "import arclm, sys; "
        "blocked={'transformers','sentencepiece','safetensors','torch'}; "
        "loaded=blocked.intersection(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
