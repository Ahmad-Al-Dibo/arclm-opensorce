import json

import pytest

from arclm.benchmarks import benchmark_deduplication, benchmark_jsonl_loading, benchmark_validation
from arclm.registry import Registry
from arclm.reproducibility import fingerprint
from arclm.security import artifact_digest, scan_for_secrets, validate_safe_model_options


def test_registry_duplicate_and_override():
    registry = Registry("test")
    registry.register("item", lambda: None)
    with pytest.raises(Exception):
        registry.register("item", lambda: None)
    registry.register("item", lambda: "new", override=True)
    assert registry.get("item")() == "new"


def test_fingerprint_stable_and_secret_redacted(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello", encoding="utf-8")
    assert fingerprint(path).value == fingerprint(path).value
    secret_fp = fingerprint({"token": "hf_" + "a" * 30})
    assert secret_fp.value


@pytest.mark.security
def test_security_secret_scan_and_safe_remote_code(tmp_path):
    path = tmp_path / "secret.txt"
    path.write_text("api_key=abc123", encoding="utf-8")
    report = scan_for_secrets([path])
    assert not report.is_valid
    assert artifact_digest(path)
    with pytest.raises(ValueError):
        validate_safe_model_options(trust_remote_code=True)


@pytest.mark.benchmark
def test_benchmark_smoke(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"text": f"row {index}"} for index in range(5)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert benchmark_jsonl_loading(path).items == 5
    assert benchmark_validation(rows).items == 5
    assert benchmark_deduplication(rows).items == 5
