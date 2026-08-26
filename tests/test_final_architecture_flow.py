from __future__ import annotations

import inspect


def test_api_levels_share_the_same_training_components():
    from arclm import Lab, Trainer
    from arclm.research import Trainer as ResearchTrainer
    from arclm.research import TrainingEngine as ResearchTrainingEngine
    from arclm.training import TrainingEngine

    assert ResearchTrainer is Trainer
    assert ResearchTrainingEngine is TrainingEngine
    assert "Trainer(" in inspect.getsource(Lab.train)
    assert "self.train(" in inspect.getsource(Lab.fine_tune)
    assert "TrainingEngine(" in inspect.getsource(Trainer.train)


def test_shared_core_owns_data_tokenization_artifacts_and_evaluation():
    from arclm import Dataset, Experiment, Model

    assert "DataEngine().ingest" in inspect.getsource(Dataset.load)
    assert "ModelInputPreparer" in inspect.getsource(Dataset.prepare)
    assert "ArcModelArtifact.save" in inspect.getsource(Model.save)
    assert "EvaluationEngine().evaluate" in inspect.getsource(Experiment.capture)


def test_api_levels_do_not_define_independent_engines():
    import arclm.lab as lab
    import arclm.models.model as model_api
    import arclm.research as research

    for module in (lab, model_api, research):
        source = inspect.getsource(module)
        assert "class DataEngine" not in source
        assert "class TrainingEngine" not in source
        assert "class RuntimeEngine" not in source
        assert "class ArtifactEngine" not in source
        assert "class EvaluationEngine" not in source
