from arclm import TokenizerFactory, create_simple_interface_app, run_simple_interface
from arclm.tokenizers import Tokenizer, create_tokenizer
from arclm.training import (
    BaseModelAdapter,
    BaseModelLoader,
    BaseTrainingPipeline,
    ModelAdapter,
    PreTrainedModelLoader,
    UnifiedPipeline,
)
from arclm.pipeline import build_model
from arclm.pipeline_v2 import build_model as legacy_build_model


def test_tokenizer_factory_selects_word_tokenizer():
    tokenizer = TokenizerFactory.create("word", max_vocab=8)
    tokenizer.build("arc lm arc")

    assert tokenizer.encode_text("arc")[0] != tokenizer.get_unknown_index()
    assert create_tokenizer("word", max_vocab=8).__class__ is tokenizer.__class__


def test_tokenizer_factory_registers_custom_tokenizer():
    class CustomTokenizer(Tokenizer):
        pass

    TokenizerFactory.register("custom_test", CustomTokenizer)

    assert isinstance(TokenizerFactory.create("custom_test", max_vocab=4), CustomTokenizer)


def test_training_exports_are_inheritable_base_classes():
    assert issubclass(UnifiedPipeline, BaseTrainingPipeline)
    assert issubclass(PreTrainedModelLoader, BaseModelLoader)
    assert issubclass(ModelAdapter, BaseModelAdapter)


def test_pipeline_v2_keeps_legacy_helper_imports():
    assert legacy_build_model is build_model


def test_simple_interface_helpers_are_exported_lazily():
    assert callable(create_simple_interface_app)
    assert callable(run_simple_interface)


def test_simple_interface_can_save_loaded_model(monkeypatch, tmp_path):
    import arclm.simple_interface as simple_interface

    class FakeSourceInfo:
        source_type = "hf_full_model"

        def to_dict(self):
            return {"source_type": self.source_type, "warnings": []}

    class FakeModel:
        source_info = FakeSourceInfo()
        device = "cpu"

        def save(self, output_dir, save_config=None):
            assert save_config.save_mode == "auto"
            assert save_config.overwrite is True
            return tmp_path / "saved-model"

    monkeypatch.setattr(simple_interface, "_model", None)
    monkeypatch.setattr(simple_interface, "_model_cache_key", None)
    monkeypatch.setattr(simple_interface, "_model_source", None)
    monkeypatch.setattr(simple_interface, "_model_options", {})
    monkeypatch.setattr(simple_interface, "load_any_model", lambda *args, **kwargs: FakeModel())

    app = simple_interface.create_simple_interface_app()
    client = app.test_client()

    home = client.get("/")
    assert home.status_code == 200
    assert simple_interface.DEFAULT_MODEL_SAVE_PATH in home.get_data(as_text=True)

    response = client.post(
        "/model/save",
        json={
            "model_source": "example/model",
            "save_path": str(tmp_path),
            "save_mode": "auto",
            "overwrite": True,
        },
    )

    assert response.status_code == 200
    assert response.json["success"] is True
    assert response.json["saved_path"] == str(tmp_path / "saved-model")
