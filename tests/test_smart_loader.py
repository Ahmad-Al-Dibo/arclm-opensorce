import json

from arclm import SmartLoader, inspect_model_source
from arclm.loaders import ModelInspector


def test_inspect_huggingface_like_folder_and_apply_overrides(tmp_path):
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}),
        encoding="utf-8",
    )
    (model_dir / "scheduler.pt").write_bytes(b"")
    (model_dir / "trainer_state.json").write_text(
        json.dumps({"epoch": 3, "global_step": 4200}),
        encoding="utf-8",
    )

    plan = SmartLoader.inspect(
        model_dir,
        precision="bf16",
        load_optimizer=False,
        load_scheduler=True,
        resume_training=True,
    )

    assert plan.model_type == "qwen2"
    assert plan.architecture == "Qwen2ForCausalLM"
    assert plan.tokenizer == "found"
    assert plan.weight_format == "safetensors"
    assert plan.precision == "bf16"
    assert plan.load_as == "adapter"
    assert plan.load_optimizer is False
    assert plan.load_scheduler is True
    assert plan.resume_training is True
    assert plan.can_resume_training is True
    assert "Training: can be resumed from epoch 3, step 4200" in plan.format_report()


def test_manual_mode_uses_explicit_settings(tmp_path):
    model_dir = tmp_path / "manual"
    model_dir.mkdir()

    plan = SmartLoader.inspect(
        model_dir,
        auto_detect=False,
        model_type="mistral",
        tokenizer="auto",
        weight_format="bin",
        precision="fp16",
        load_as="full_model",
    )

    assert plan.model_type == "mistral"
    assert plan.tokenizer == "auto"
    assert plan.weight_format == "bin"
    assert plan.precision == "fp16"
    assert plan.load_as == "full_model"


def test_huggingface_uri_alias_inspects_as_model_id():
    info = inspect_model_source("hf://gpt2")

    assert info.source_type == "hf_full_model"
    assert info.resolved_source == "gpt2"
    assert info.load_as == "full_model"


def test_custom_inspector_can_extend_detection(tmp_path):
    source = tmp_path / "custom.model"
    source.write_text("x", encoding="utf-8")

    class CustomInspector(ModelInspector):
        def can_inspect(self, source):
            return str(source).endswith(".model")

        def inspect(self, source):
            plan = super().inspect(source)
            plan.model_type = "custom"
            plan.weight_format = "custom"
            return plan

    plan = inspect_model_source(source, inspectors=[CustomInspector()])

    assert plan.model_type == "custom"
    assert plan.weight_format == "custom"
