import hashlib
import json
import subprocess
import sys

import pytest
import torch

from arclm.checkpoints import inspect_checkpoint, verify_checkpoint, write_checkpoint_manifest
from arclm.config import load_arclm_config, migrate_config
from arclm.doctor import run_doctor
from arclm.exceptions import CheckpointError, ConfigurationError
from arclm.resources import DeviceConfig
from arclm.stability import cli_manifest, stable_api_paths


pytestmark = pytest.mark.unit


@pytest.mark.config
def test_typed_config_rejects_unknown_and_resolves_paths(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text":"hello"}\n', encoding="utf-8")
    config_path = tmp_path / "arclm.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "data": {"path": "data.jsonl", "format": "jsonl", "schema": "text"},
                "model": {"name": "gpt2"},
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown field"):
        load_arclm_config(config_path)

    cfg = load_arclm_config(config_path, permissive=True)
    assert cfg.data.path == str(data_path.resolve())


@pytest.mark.config
def test_config_migration_renames_model_source(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(
        json.dumps({"version": "0", "data": {"path": "data.jsonl"}, "model": {"source": "gpt2"}}),
        encoding="utf-8",
    )

    report = migrate_config(old, output=new)
    assert "version -> schema_version" in report.fields_renamed
    assert "model.source -> model.name" in report.fields_renamed
    assert report.config.model.name == "gpt2"
    assert new.exists()


@pytest.mark.config
def test_config_env_expansion_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCLM_TEST_DATA", "data.jsonl")
    raw = {"schema_version": "1", "data": {"path": "${ARCLM_TEST_DATA}"}}

    with pytest.raises(ConfigurationError, match="Environment-variable expansion is disabled"):
        load_arclm_config(raw)
    assert load_arclm_config(raw, allow_env=True).data.path.endswith("data.jsonl")


@pytest.mark.compatibility
def test_stable_api_snapshot_is_explicit():
    snapshot = json.loads(open("tests/fixtures/api_snapshot_0_9.json", encoding="utf-8").read())
    assert stable_api_paths() == snapshot
    assert cli_manifest()["arclm doctor"] == "stable"


@pytest.mark.checkpoint
def test_directory_checkpoint_manifest_hash_verification(tmp_path):
    root = tmp_path / "checkpoint"
    (root / "model").mkdir(parents=True)
    (root / "training").mkdir()
    (root / "tokenizer").mkdir()
    (root / "model" / "config.json").write_text("{}", encoding="utf-8")
    (root / "model" / "model.safetensors").write_bytes(b"safe")
    (root / "training" / "state.json").write_text("{}", encoding="utf-8")
    write_checkpoint_manifest(root, model_config={"architecture": "test"})

    report = verify_checkpoint(root)
    assert report.is_verified
    assert report.model_weight_format == "safetensors"

    (root / "model" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="Hash mismatch"):
        verify_checkpoint(root)


@pytest.mark.checkpoint
def test_safe_mode_rejects_legacy_pickle_checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}}, path)

    report = inspect_checkpoint(path)
    assert report.trusted_pickle_required
    assert report.errors


@pytest.mark.security
def test_device_config_and_doctor_are_cpu_safe(tmp_path):
    selection = DeviceConfig(device="auto", precision="auto").resolve()
    assert selection.selected_device in {"cpu", "cuda:0"}

    report = run_doctor(run_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")
    assert any(check.name == "python" for check in report.checks)


@pytest.mark.cli
def test_phase4_cli_help_surfaces():
    commands = [
        ["doctor", "--json"],
        ["config", "validate", "--help"],
        ["config", "migrate", "--help"],
        ["model", "certify", "--help"],
        ["checkpoint", "inspect", "--help"],
        ["checkpoint", "verify", "--help"],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, "-m", "arclm", *command], capture_output=True, text=True)
        assert result.returncode == 0
