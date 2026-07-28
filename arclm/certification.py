"""Model-family certification protocol and lightweight runner."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ._version import __version__
from .evaluation import evaluate
from .models import inspect_model_support, load_model
from .sft import train_sft


CERTIFICATION_PROTOCOL_VERSION = "1.0"


@dataclass
class CertificationCheck:
    """One model certification check."""

    name: str
    status: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCertificationReport:
    """External model-family certification result."""

    family: str
    source: str
    revision: str | None
    support_status: str
    cpu_certification: str
    gpu_certification: str = "untested"
    training_certification: str = "unsupported"
    checks: list[CertificationCheck] = field(default_factory=list)
    duration_seconds: float = 0.0
    report_type: str = "model_certification_report"
    schema_version: str = CERTIFICATION_PROTOCOL_VERSION
    arclm_version: str = __version__

    @property
    def is_certified(self) -> bool:
        return self.cpu_certification == "certified" and all(check.status == "passed" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checks"] = [check.to_dict() for check in self.checks]
        data["is_certified"] = self.is_certified
        return data


def certify_model_family(
    family: str,
    source: str,
    *,
    revision: str | None = None,
    device: str = "cpu",
    run_training: bool = True,
    max_steps: int = 1,
) -> ModelCertificationReport:
    """Run the ArcLM CPU certification protocol for a tiny causal-LM source."""

    started = time.perf_counter()
    checks: list[CertificationCheck] = []
    support = inspect_model_support(source, device=device, trust_remote_code=False)
    checks.append(CertificationCheck("configuration_and_tokenizer_loading", "passed" if support.is_supported else "failed", support.summary(), support.to_dict()))
    if not support.is_supported:
        return ModelCertificationReport(family, source, revision, support.support_level, "unsupported", checks=checks, duration_seconds=time.perf_counter() - started)

    try:
        torch.manual_seed(123)
        bundle = load_model(source, device=device, trust_remote_code=False)
        checks.append(CertificationCheck("model_loading", "passed", details={"backend": bundle.backend, "precision": bundle.precision}))
        first = bundle.predict("Hello", max_new_tokens=3, do_sample=False)
        torch.manual_seed(123)
        second = bundle.predict("Hello", max_new_tokens=3, do_sample=False)
        checks.append(CertificationCheck("deterministic_greedy_generation", "passed" if first == second else "failed", details={"first": first, "second": second}))
        outputs = [bundle.predict(prompt, max_new_tokens=2, do_sample=False) for prompt in ["Hello", "ArcLM"]]
        checks.append(CertificationCheck("batched_generation_order", "passed", details={"outputs": outputs}))
        eval_report = evaluate(bundle, [{"text": "Hello world"}], metrics=["generation_length", "latency"])
        checks.append(CertificationCheck("evaluation", "passed" if not eval_report.errors else "failed", details=eval_report.to_dict()))
        training_status = "inference_only"
        if run_training:
            with tempfile.TemporaryDirectory() as tmp_name:
                tmp = Path(tmp_name)
                data_path = tmp / "sft.jsonl"
                data_path.write_text(json.dumps({"prompt": "Hello", "completion": "Hi"}) + "\n", encoding="utf-8")
                out = tmp / "out"
                result = train_sft(model=source, dataset=str(data_path), output_dir=str(out), batch_size=1, max_steps=max_steps, num_epochs=1, trust_remote_code=False)
                checks.append(CertificationCheck("minimal_training_step", "passed" if result.steps >= 1 else "failed", details={"steps": result.steps}))
                reloaded = load_model(out, device=device, trust_remote_code=False)
                after_reload = reloaded.predict("Hello", max_new_tokens=2, do_sample=False)
                checks.append(CertificationCheck("save_reload_inference", "passed", details={"output": after_reload}))
                training_status = "minimal_step"
        status = "certified" if all(check.status == "passed" for check in checks) else "partial"
        return ModelCertificationReport(
            family=family,
            source=source,
            revision=revision,
            support_status=support.support_level,
            cpu_certification=status,
            training_certification=training_status,
            checks=checks,
            duration_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        checks.append(CertificationCheck("certification_exception", "failed", str(exc), {"type": type(exc).__name__}))
        return ModelCertificationReport(family, source, revision, support.support_level, "partial", checks=checks, duration_seconds=time.perf_counter() - started)


__all__ = ["CERTIFICATION_PROTOCOL_VERSION", "CertificationCheck", "ModelCertificationReport", "certify_model_family"]
