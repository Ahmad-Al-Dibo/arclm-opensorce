from __future__ import annotations

import json
from pathlib import Path


def _tiny_model(tmp_path: Path):
    from arclm import Dataset, Lab, Runtime

    dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"}])
    model = Lab(runtime=Runtime.auto(prefer="cpu")).model(data=dataset, size="tiny")
    return model


def test_runtime_compute_plan_reports_device_precision_and_memory():
    from arclm import Runtime

    runtime = Runtime.auto(prefer="cpu", precision="float16")
    plan = runtime.compute_plan(parameters=10, trainable_parameters=4)
    report = runtime.to_dict()

    assert runtime.device_name == "cpu"
    assert runtime.precision == "float32"
    assert plan["estimated_total_memory_bytes"] == 10 * 4 + 4 * 4 + 4 * 4 * 2
    assert report["memory"]["selected"]["type"] == "cpu"
    assert "future_backends" in report["backend_capabilities"]


def test_artifact_single_and_directory_layouts_use_expected_files(tmp_path: Path):
    from arclm import Model, Runtime

    model = _tiny_model(tmp_path)
    single = model.save(tmp_path / "single.arcmodel", layout="single", overwrite=True)
    directory = model.save(tmp_path / "directory-model", layout="directory", overwrite=True)

    assert single.path.is_file()
    assert (directory.path / "manifest.json").exists()
    assert (directory.path / "config.json").exists()
    assert (directory.path / "tokenizer" / "tokenizer.json").exists()
    assert (directory.path / "weights" / "model.safetensors").exists()
    assert Model.load(single.path, runtime=Runtime.auto(prefer="cpu")).inspect()["architecture_id"] == "arclm-native-causal-lm"
    assert Model.load(directory.path, runtime=Runtime.auto(prefer="cpu")).inspect()["architecture_id"] == "arclm-native-causal-lm"


def test_sharded_artifact_splits_tensors_and_iterates_chunks(tmp_path: Path):
    from arclm.artifacts import ArcModelArtifact

    model = _tiny_model(tmp_path)
    artifact = model.save(tmp_path / "sharded-model", layout="sharded", shard_size=1, overwrite=True)
    report = artifact.inspect()
    chunks = list(ArcModelArtifact(artifact.path).iter_state_dict())

    assert (artifact.path / "manifest.json").exists()
    assert (artifact.path / "index.json").exists()
    assert (artifact.path / "weights").is_dir()
    assert report["layout"] == "sharded"
    assert len(report["weight_files"]) > 1
    assert len(chunks) == len(report["weight_files"])
    assert sum(len(chunk) for chunk in chunks) == report["tensor_count"]


def test_artifact_version_errors_are_clear(tmp_path: Path):
    from arclm.artifacts import ArcModelArtifact, ArtifactVersionError

    model = _tiny_model(tmp_path)
    artifact = model.save(tmp_path / "versioned", layout="directory", overwrite=True)
    manifest_path = artifact.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        ArcModelArtifact(artifact.path).manifest()
    except ArtifactVersionError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ArtifactVersionError.")

    assert "[ArcLM:artifacts]" in message
    assert "schema_version" in message
    assert "Action:" in message
