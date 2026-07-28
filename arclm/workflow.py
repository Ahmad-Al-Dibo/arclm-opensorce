"""Configuration-driven ArcLM workflow runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from ._version import __version__
from .data_quality import analyze_dataset, find_duplicates, split_dataset
from .data_sources import open_dataset
from .exceptions import ArcLMError, ConfigurationError
from .models import inspect_model_support, load_model
from .reproducibility import fingerprint
from .runs import Run
from .schemas import validate_records
from .tokenization import tokenize_dataset


@dataclass
class WorkflowStageResult:
    name: str
    status: str
    duration_seconds: float = 0.0
    report_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowResult:
    status: str
    run_dir: str
    stages: list[WorkflowStageResult]
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report_type: str = "workflow_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = [stage.to_dict() for stage in self.stages]
        return data


def run_workflow(
    config: str | Path | dict[str, Any],
    *,
    dry_run: bool = False,
    stages: Optional[Sequence[str]] = None,
    resume: bool = False,
) -> WorkflowResult:
    """Run a local ArcLM workflow from JSON/TOML-like configuration."""

    cfg = load_workflow_config(config)
    selected = set(stages or cfg.get("stages") or ["load", "validate", "quality", "deduplicate", "split", "tokenize", "model", "train", "evaluate"])
    run_cfg = cfg.get("run", {})
    with Run(run_cfg.get("name", "workflow"), output_dir=run_cfg.get("output_dir", "runs")) as run:
        run.log_config(cfg, "workflow")
        stage_results: list[WorkflowStageResult] = []
        context: dict[str, Any] = {"config": cfg, "run": run}
        try:
            for stage_name in ["load", "validate", "quality", "deduplicate", "split", "tokenize", "model", "train", "evaluate"]:
                if stage_name not in selected:
                    stage_results.append(WorkflowStageResult(stage_name, "skipped"))
                    continue
                result = _run_stage(stage_name, context, dry_run=dry_run)
                stage_results.append(result)
                if result.error:
                    raise ArcLMError(result.error)
            status = "dry_run" if dry_run else "completed"
        except Exception as exc:
            status = "failed"
            stage_results.append(WorkflowStageResult("failure", "failed", error=str(exc)))
            run.fail(str(exc))
        workflow = WorkflowResult(status=status, run_dir=str(run.path), stages=stage_results, dry_run=dry_run)
        run.log_report(workflow, "workflow")
        return workflow


def load_workflow_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load JSON or TOML workflow configuration."""

    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except Exception as exc:
                raise ConfigurationError("TOML workflow files require Python 3.11+ or tomli.") from exc
        return tomllib.loads(text)
    raise ConfigurationError("Workflow config must be .json or .toml.")


def _run_stage(name: str, context: dict[str, Any], *, dry_run: bool) -> WorkflowStageResult:
    import time

    started = time.perf_counter()
    run: Run = context["run"]
    cfg = context["config"]
    try:
        if name == "load":
            data_cfg = cfg.get("data", {})
            source = open_dataset(
                data_cfg["path"],
                format=data_cfg.get("format", "jsonl"),
                streaming=data_cfg.get("streaming", True),
                malformed=data_cfg.get("malformed", "raise"),
            )
            context["dataset"] = source
            run.log_report({"metadata": source.metadata.to_dict(), "fingerprint": fingerprint(data_cfg).value}, "dataset-source")
        elif name == "validate":
            records = _materialize(context)
            schema = cfg.get("data", {}).get("schema", "text")
            report = validate_records(records, schema=schema, strict=cfg.get("data", {}).get("strict", False))
            context["records"] = records
            path = run.log_report(report, "validation")
            return WorkflowStageResult(
                name,
                "passed" if report.is_valid else "failed",
                time.perf_counter() - started,
                str(path),
                None if report.is_valid else "Dataset validation failed.",
            )
        elif name == "quality":
            records = _materialize(context)
            report = analyze_dataset(records, schema=cfg.get("data", {}).get("schema"), include_samples=False)
            path = run.log_report(report, "data-quality")
            return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
        elif name == "deduplicate":
            records = _materialize(context)
            dedupe_cfg = cfg.get("deduplication", {})
            report = find_duplicates(records, fields=dedupe_cfg.get("fields"), normalize=dedupe_cfg.get("normalize", True))
            duplicate_indexes = {index for group in report.groups for index in group.indexes[1:]}
            if duplicate_indexes:
                context["records"] = [record for index, record in enumerate(records) if index not in duplicate_indexes]
            payload = report.to_dict()
            payload["removed_records"] = len(duplicate_indexes)
            payload["output_records"] = len(context.get("records", records))
            path = run.log_report(payload, "duplicates")
            return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
        elif name == "split":
            records = _materialize(context)
            split_cfg = cfg.get("split", {})
            result = split_dataset(
                records,
                train=split_cfg.get("train", 0.8),
                validation=split_cfg.get("validation", 0.1),
                test=split_cfg.get("test", 0.1),
                seed=split_cfg.get("seed", 42),
                strategy=split_cfg.get("strategy", "hash"),
                key=split_cfg.get("key"),
                group_key=split_cfg.get("group_key"),
                split_field=split_cfg.get("split_field"),
            )
            context["splits"] = result.splits
            path = run.log_report(result.report, "splits")
            return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
        elif name == "tokenize":
            if dry_run:
                path = run.log_report({"status": "checked", "cache_dir": cfg.get("tokenization", {}).get("cache_dir")}, "tokenization")
                return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
            else:
                tok_cfg = cfg.get("tokenization", {})
                rows = context.get("splits", {}).get("train") or _materialize(context)
                result = tokenize_dataset(
                    rows,
                    tokenizer=tok_cfg.get("tokenizer", cfg.get("model", {}).get("source", "gpt2")),
                    schema=tok_cfg.get("schema", cfg.get("data", {}).get("schema", "text")),
                    max_length=tok_cfg.get("max_length"),
                    cache_dir=tok_cfg.get("cache_dir"),
                )
                context["tokenized"] = result
                path = run.log_report(result, "tokenization")
                return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
        elif name == "model":
            model_cfg = cfg.get("model", {})
            support = inspect_model_support(
                model_cfg["source"], trust_remote_code=model_cfg.get("trust_remote_code", False), device=model_cfg.get("device", "cpu")
            )
            context["model_support"] = support
            path = run.log_report(support, "model-support")
            if not support.is_supported:
                return WorkflowStageResult(name, "failed", time.perf_counter() - started, str(path), "Model is unsupported.")
            if not dry_run:
                context["model_bundle"] = load_model(
                    model_cfg["source"], device=model_cfg.get("device", "cpu"), trust_remote_code=model_cfg.get("trust_remote_code", False)
                )
            return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
        elif name == "train":
            if dry_run or not cfg.get("training", {}).get("enabled", False):
                path = run.log_report({"status": "skipped", "dry_run": dry_run}, "training")
                return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
            else:
                from .training import TrainingConfig, train

                train_cfg = cfg.get("training", {})
                report = train(
                    model=context.get("model_bundle"),
                    dataset=context.get("splits", {}).get("train") or _materialize(context),
                    config=TrainingConfig(
                        output_dir=str(run.path / "checkpoints" / "training"),
                        max_steps=train_cfg.get("max_steps", 1),
                        batch_size=train_cfg.get("batch_size", 1),
                        epochs=train_cfg.get("epochs", 1),
                    ),
                )
                path = run.log_report(report, "training")
                return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
        elif name == "evaluate":
            if dry_run or "model_bundle" not in context:
                path = run.log_report({"status": "skipped", "dry_run": dry_run}, "evaluation")
                return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
            else:
                from .evaluation import evaluate

                rows = context.get("splits", {}).get("test") or _materialize(context)
                report = evaluate(context["model_bundle"], rows, metrics=["generation_length", "latency"])
                path = run.log_report(report, "evaluation")
                return WorkflowStageResult(name, "passed", time.perf_counter() - started, str(path))
        return WorkflowStageResult(name, "passed", time.perf_counter() - started)
    except Exception as exc:
        return WorkflowStageResult(name, "failed", time.perf_counter() - started, error=str(exc))


def _materialize(context: dict[str, Any]) -> list[dict[str, Any]]:
    if "records" not in context:
        context["records"] = [dict(row) for row in context["dataset"]]
    return context["records"]


__all__ = ["WorkflowResult", "WorkflowStageResult", "load_workflow_config", "run_workflow"]
