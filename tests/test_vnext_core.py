import torch

from arclm import Model, ModelRegistry, Runtime, Tokenizer


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
