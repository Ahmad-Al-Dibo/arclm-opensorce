"""ArcLM command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .config import create_config
from .config import load_arclm_config, migrate_config, validate_arclm_config
from .config_loader import load_config
from .checkpoints import inspect_checkpoint, verify_checkpoint
from .data_processor import DataProcessor
from .data_pipeline import DataPipeline
from .data_quality import analyze_dataset, shard_dataset, split_dataset
from .data_sources import open_dataset
from .exceptions import ArcLMError, CheckpointError, ConfigurationError, DatasetValidationError, ModelLoadError, OptionalDependencyError, TrainingError, UnsupportedModelError
from .logging import configure_logging
from .models import inspect_model_support
from .pipeline import train_model
from .reproducibility import fingerprint
from .cache import clear_cache, inspect_cache
from .certification import certify_model_family
from .doctor import run_doctor
from .runs import inspect_run, list_runs
from .schemas import validate_records
from .supported_models import get_supported_models
from .workflow import run_workflow


EXIT_SUCCESS = 0
EXIT_GENERAL_FAILURE = 1
EXIT_INVALID_USAGE = 2
EXIT_CONFIGURATION_ERROR = 3
EXIT_DATASET_VALIDATION_ERROR = 4
EXIT_UNSUPPORTED_MODEL = 5
EXIT_MODEL_LOAD_ERROR = 6
EXIT_TRAINING_ERROR = 7
EXIT_CHECKPOINT_ERROR = 8
EXIT_OPTIONAL_DEPENDENCY_MISSING = 9
EXIT_WORKFLOW_PARTIAL = 10


def _json_default(value: Any) -> str:
    return str(value)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _load_records(path: str, fmt: str | None = None) -> list[dict[str, Any]]:
    return DataProcessor.load(path, format=fmt).samples


def version_command(_args: argparse.Namespace) -> int:
    """Print ArcLM version."""

    print(__version__)
    return 0


def info_command(args: argparse.Namespace) -> int:
    """Print environment and package information."""

    import platform
    import torch

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None

    payload = {
        "arclm": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "transformers": transformers_version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if args.json:
        _print_json(payload)
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def doctor_command(args: argparse.Namespace) -> int:
    report = run_doctor(config=args.config, checkpoint=args.checkpoint, run_dir=args.run_dir, cache_dir=args.cache_dir)
    if args.json:
        _print_json(report.to_dict())
    else:
        for check in report.checks:
            print(f"{check.status}: {check.name} - {check.message}")
    return 0 if report.is_valid else EXIT_GENERAL_FAILURE


def config_validate_command(args: argparse.Namespace) -> int:
    config = validate_arclm_config(args.config, permissive=args.permissive, allow_env=args.allow_env)
    payload = {"valid": True, "schema_version": config.schema_version, "config": config.to_dict(redact=True)}
    _print_json(payload) if args.json else print(f"valid schema_version={config.schema_version}")
    return 0


def config_show_command(args: argparse.Namespace) -> int:
    config = load_arclm_config(args.config, permissive=args.permissive, allow_env=args.allow_env)
    _print_json(config.to_dict(redact=not args.show_secrets))
    return 0


def config_migrate_command(args: argparse.Namespace) -> int:
    report = migrate_config(args.config, target_version=args.target_version, output=args.output, permissive=args.permissive)
    _print_json(report.to_dict()) if args.json else print(f"{report.source_schema_version} -> {report.target_schema_version}")
    return 0


def data_inspect_command(args: argparse.Namespace) -> int:
    records = _load_records(args.input, args.format)
    fields = sorted({field for row in records for field in row})
    payload = {
        "input": args.input,
        "records": len(records),
        "fields": fields,
        "sample": records[: args.sample],
    }
    _print_json(payload) if args.json else print(json.dumps(payload, indent=2))
    return 0


def data_validate_command(args: argparse.Namespace) -> int:
    records = _load_records(args.input, args.format)
    report = validate_records(
        records,
        schema=args.schema,
        strict=args.strict,
        allow_empty=args.allow_empty,
        check_duplicates=args.check_duplicates,
        duplicate_field=args.duplicate_field,
    )
    if args.json:
        _print_json(report.to_dict())
    else:
        print(report.summary())
        for issue in report.errors[:10]:
            print(f"error record={issue.index} field={issue.field} category={issue.category}: {issue.message}")
        for issue in report.warnings[:10]:
            print(f"warning record={issue.index} field={issue.field} category={issue.category}: {issue.message}")
    return 0 if report.is_valid else 1


def data_prepare_command(args: argparse.Namespace) -> int:
    records = _load_records(args.input, args.format)
    pipeline = DataPipeline(seed=args.seed)
    if args.normalize_text:
        pipeline.normalize_text(args.text_field)
    if args.remove_empty:
        pipeline.remove_empty(args.text_field)
    if args.deduplicate:
        pipeline.deduplicate(args.text_field)
    if args.schema:
        pipeline.validate(args.schema, strict=args.strict, allow_empty=args.allow_empty)
    processed, report = pipeline.run(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in processed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = report.to_dict()
    payload["output"] = str(output)
    if args.json:
        _print_json(payload)
    else:
        print(report.summary())
        print(f"wrote: {output}")
    return 0 if report.is_valid else 1


def data_analyze_command(args: argparse.Namespace) -> int:
    dataset = open_dataset(args.input, format=args.format, streaming=True, malformed=args.malformed)
    report = analyze_dataset(
        dataset,
        schema=args.schema,
        checks=args.checks,
        include_samples=args.include_samples,
        redact_samples=not args.no_redact_samples,
        max_sample_chars=args.max_sample_chars,
    )
    _print_json(report.to_dict()) if args.json else print(report.summary())
    return 0 if not report.errors else EXIT_DATASET_VALIDATION_ERROR


def data_split_command(args: argparse.Namespace) -> int:
    dataset = open_dataset(args.input, format=args.format, streaming=True, malformed=args.malformed)
    result = split_dataset(
        dataset,
        train=args.train,
        validation=args.validation,
        test=args.test,
        seed=args.seed,
        strategy=args.strategy,
        key=args.key,
        group_key=args.group_key,
        split_field=args.split_field,
    )
    _print_json(result.to_dict()) if args.json else print(result.report.to_dict())
    return 0


def data_shard_command(args: argparse.Namespace) -> int:
    dataset = open_dataset(args.input, format=args.format, streaming=True, malformed=args.malformed)
    result = shard_dataset(dataset, num_shards=args.num_shards, strategy=args.strategy, key=args.key, seed=args.seed)
    _print_json(result.to_dict()) if args.json else print(result.report.to_dict())
    return 0


def data_fingerprint_command(args: argparse.Namespace) -> int:
    report = fingerprint(Path(args.input), mode=args.mode)
    _print_json(report.to_dict()) if args.json else print(report.value)
    return 0


def model_inspect_command(args: argparse.Namespace) -> int:
    report = inspect_model_support(
        args.source,
        task=args.task,
        device=args.device,
        precision=args.precision,
        trust_remote_code=args.trust_remote_code,
        tokenizer_path=args.tokenizer_path,
    )
    _print_json(report.to_dict()) if args.json else print(report.summary())
    return 0 if report.is_supported else 1


def model_list_command(args: argparse.Namespace) -> int:
    records = [record.to_dict() for record in get_supported_models(args.status)]
    _print_json(records)
    return 0


def model_load_check_command(args: argparse.Namespace) -> int:
    report = inspect_model_support(
        args.source,
        task=args.task,
        device=args.device,
        precision=args.precision,
        trust_remote_code=args.trust_remote_code,
    )
    if args.json:
        _print_json(report.to_dict())
    else:
        print(report.summary())
        for error in report.errors:
            print(f"error: {error}")
    return 0 if report.is_supported else 1


def model_certify_command(args: argparse.Namespace) -> int:
    report = certify_model_family(
        args.family,
        args.source,
        revision=args.revision,
        device=args.device,
        run_training=not args.no_training,
        max_steps=args.max_steps,
    )
    _print_json(report.to_dict()) if args.json else print(json.dumps(report.to_dict(), indent=2, default=_json_default))
    return 0 if report.is_certified else EXIT_GENERAL_FAILURE


def checkpoint_inspect_command(args: argparse.Namespace) -> int:
    report = inspect_checkpoint(args.path, trust=args.trust)
    _print_json(report.to_dict()) if args.json else print(report.summary())
    return 0 if not report.errors else EXIT_CHECKPOINT_ERROR


def checkpoint_verify_command(args: argparse.Namespace) -> int:
    report = verify_checkpoint(args.path, trust=args.trust)
    _print_json(report.to_dict()) if args.json else print(report.summary())
    return 0


def train_command(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else create_config(
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        tokenizer_type=args.tokenizer_type,
        max_vocab=args.max_vocab,
        validation_split=args.validation_split,
        device=args.device,
        training_log_interval=args.training_log_interval,
    )
    config.validate()
    result = train_model(
        mode=args.mode,
        data=args.data,
        output=args.output,
        checkpoint=args.checkpoint,
        config=config,
    )
    payload = {
        "mode": result.mode,
        "model_path": result.model_path,
        "vocab_size": result.vocab_size,
        "history": result.history,
    }
    _print_json(payload) if args.json else print(f"saved: {result.model_path}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    import torch

    from .diagnostics import calculate_metrics, export_metrics_to_json, export_metrics_to_markdown
    from .inference import load_model

    loaded = load_model(args.model, device=args.device)
    config = loaded.config
    data = DataProcessor.load(args.data).transform(format="pretraining")
    tokenizer = getattr(loaded.generator, "tokenizer", None)
    if tokenizer is None:
        raise ArcLMError("Loaded model has no tokenizer for evaluation data.")
    encoded = []
    for row in data.samples:
        encoded.extend(tokenizer.encode_text(row["text"]))
    from .dataset import create_dataloader

    loader = create_dataloader(encoded, config.block_size, args.batch_size, shuffle=False)
    metrics = calculate_metrics(loaded.model, loader, config, torch.device(args.device))
    output = Path(args.output)
    export_metrics_to_json(metrics, str(output))
    export_metrics_to_markdown(metrics, str(output.with_suffix(".md")))
    payload = metrics.to_dict()
    payload["output"] = str(output)
    _print_json(payload) if args.json else print(metrics.to_dict())
    return 0


def generate_command(args: argparse.Namespace) -> int:
    from .inference import GenerationConfig, generate, load_model

    loaded = load_model(args.model, device=args.device)
    result = generate(
        loaded,
        prompts=args.prompt,
        config=GenerationConfig(max_new_tokens=args.length, temperature=args.temperature, top_k=args.top_k),
        batch_size=args.batch_size,
    )
    payload = result.to_dict()
    _print_json(payload) if args.json else print("\n".join(result.outputs))
    return 0


def workflow_run_command(args: argparse.Namespace) -> int:
    result = run_workflow(args.config, dry_run=args.dry_run, stages=args.stage)
    _print_json(result.to_dict()) if args.json else print(f"{result.status}: {result.run_dir}")
    if result.status == "completed" or result.status == "dry_run":
        return 0
    if result.status == "failed" and any(stage.status == "passed" for stage in result.stages):
        return EXIT_WORKFLOW_PARTIAL
    return EXIT_GENERAL_FAILURE


def runs_list_command(args: argparse.Namespace) -> int:
    rows = list_runs(args.output_dir)
    _print_json(rows) if args.json else print("\n".join(row.get("path", "") for row in rows))
    return 0


def runs_inspect_command(args: argparse.Namespace) -> int:
    _print_json(inspect_run(args.path))
    return 0


def cache_inspect_command(args: argparse.Namespace) -> int:
    _print_json(inspect_cache(args.cache_dir).to_dict())
    return 0


def cache_clear_command(args: argparse.Namespace) -> int:
    _print_json(clear_cache(args.cache_dir, key=args.key).to_dict())
    return 0


def plugins_list_command(args: argparse.Namespace) -> int:
    from .registry import list_plugins

    _print_json(list_plugins())
    return 0


def run_command(args: argparse.Namespace) -> int:
    if args.run in {"simple-interface", "simple_interface"}:
        try:
            from .simple_interface import run_simple_interface
        except ModuleNotFoundError as exc:
            if exc.name == "flask":
                raise ArcLMError("The simple interface requires Flask. Install arclm[web].") from exc
            raise
        run_simple_interface(host=args.host, port=args.port, debug=args.debug)
        return 0
    raise ArcLMError(f"Unknown runtime target: {args.run}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ArcLM data and causal-LM workflow tools")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--run", choices=["simple-interface", "simple_interface"], help="Run a runtime helper")
    parser.add_argument("--host", default=None, help="Host for --run simple-interface")
    parser.add_argument("--port", type=int, default=None, help="Port for --run simple-interface")
    parser.add_argument("--debug", action="store_true", default=None, help="Enable debug mode for --run")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode for --run")
    parser.add_argument("--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--verbose", action="store_true", help="Show verbose logs")
    parser.add_argument("--traceback", action="store_true", help="Show tracebacks for expected ArcLM errors")

    subparsers = parser.add_subparsers(dest="command")

    version_parser = subparsers.add_parser("version", help="Print ArcLM version")
    version_parser.set_defaults(func=version_command)

    info_parser = subparsers.add_parser("info", help="Print environment information")
    info_parser.add_argument("--json", action="store_true")
    info_parser.set_defaults(func=info_command)

    doctor_parser = subparsers.add_parser("doctor", help="Run local ArcLM diagnostics")
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--config")
    doctor_parser.add_argument("--checkpoint")
    doctor_parser.add_argument("--run-dir", default="runs")
    doctor_parser.add_argument("--cache-dir", default=".arclm/cache")
    doctor_parser.set_defaults(func=doctor_command)

    config_parser = subparsers.add_parser("config", help="Validate, show, and migrate ArcLM configuration files")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_validate = config_sub.add_parser("validate", help="Validate an ArcLM JSON/TOML config")
    config_validate.add_argument("config")
    config_validate.add_argument("--json", action="store_true")
    config_validate.add_argument("--permissive", action="store_true")
    config_validate.add_argument("--allow-env", action="store_true")
    config_validate.set_defaults(func=config_validate_command)
    config_show = config_sub.add_parser("show", help="Show effective typed configuration")
    config_show.add_argument("config")
    config_show.add_argument("--permissive", action="store_true")
    config_show.add_argument("--allow-env", action="store_true")
    config_show.add_argument("--show-secrets", action="store_true")
    config_show.set_defaults(func=config_show_command)
    config_migrate = config_sub.add_parser("migrate", help="Migrate an older ArcLM config to the current schema")
    config_migrate.add_argument("config")
    config_migrate.add_argument("--output")
    config_migrate.add_argument("--target-version", default="1")
    config_migrate.add_argument("--json", action="store_true")
    config_migrate.add_argument("--permissive", action="store_true")
    config_migrate.set_defaults(func=config_migrate_command)

    data_parser = subparsers.add_parser("data", help="Dataset tools")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)
    _add_data_common(data_sub.add_parser("inspect", help="Inspect records"))
    data_sub.choices["inspect"].add_argument("--sample", type=int, default=3)
    data_sub.choices["inspect"].set_defaults(func=data_inspect_command)

    validate_parser = _add_data_common(data_sub.add_parser("validate", help="Validate records"))
    _add_validation_options(validate_parser)
    validate_parser.set_defaults(func=data_validate_command)

    prepare_parser = _add_data_common(data_sub.add_parser("prepare", help="Prepare records into JSONL"))
    prepare_parser.add_argument("--output", "-o", required=True)
    prepare_parser.add_argument("--schema", choices=["text", "prompt_completion", "instruction", "conversation"])
    prepare_parser.add_argument("--strict", action="store_true")
    prepare_parser.add_argument("--allow-empty", action="store_true")
    prepare_parser.add_argument("--normalize-text", action="store_true", default=True)
    prepare_parser.add_argument("--no-normalize-text", dest="normalize_text", action="store_false")
    prepare_parser.add_argument("--remove-empty", action="store_true", default=True)
    prepare_parser.add_argument("--deduplicate", action="store_true")
    prepare_parser.add_argument("--text-field", default="text")
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(func=data_prepare_command)

    analyze_parser = _add_data_common(data_sub.add_parser("analyze", help="Analyze dataset quality"))
    analyze_parser.add_argument("--schema", choices=["text", "prompt_completion", "instruction", "conversation"])
    analyze_parser.add_argument("--checks", nargs="*")
    analyze_parser.add_argument("--malformed", default="raise", choices=["raise", "report"])
    analyze_parser.add_argument("--include-samples", action="store_true")
    analyze_parser.add_argument("--no-redact-samples", action="store_true")
    analyze_parser.add_argument("--max-sample-chars", type=int, default=120)
    analyze_parser.set_defaults(func=data_analyze_command)

    split_parser = _add_data_common(data_sub.add_parser("split", help="Deterministically split records"))
    split_parser.add_argument("--malformed", default="raise", choices=["raise", "report"])
    split_parser.add_argument("--train", type=float, default=0.8)
    split_parser.add_argument("--validation", type=float, default=0.1)
    split_parser.add_argument("--test", type=float, default=0.1)
    split_parser.add_argument("--strategy", default="hash", choices=["hash", "random", "chronological"])
    split_parser.add_argument("--key")
    split_parser.add_argument("--group-key")
    split_parser.add_argument("--split-field")
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.set_defaults(func=data_split_command)

    shard_parser = _add_data_common(data_sub.add_parser("shard", help="Deterministically shard records"))
    shard_parser.add_argument("--malformed", default="raise", choices=["raise", "report"])
    shard_parser.add_argument("--num-shards", type=int, required=True)
    shard_parser.add_argument("--strategy", default="contiguous", choices=["contiguous", "round_robin", "hash"])
    shard_parser.add_argument("--key")
    shard_parser.add_argument("--seed", type=int, default=0)
    shard_parser.set_defaults(func=data_shard_command)

    fingerprint_parser = data_sub.add_parser("fingerprint", help="Fingerprint a dataset file or directory")
    fingerprint_parser.add_argument("input")
    fingerprint_parser.add_argument("--mode", default="content", choices=["content", "metadata", "sampled"])
    fingerprint_parser.add_argument("--json", action="store_true")
    fingerprint_parser.set_defaults(func=data_fingerprint_command)

    model_parser = subparsers.add_parser("model", help="Model support tools")
    model_sub = model_parser.add_subparsers(dest="model_command", required=True)
    inspect_parser = model_sub.add_parser(
        "inspect",
        help="Inspect model support",
        description="Inspect model support",
    )
    _add_model_options(inspect_parser)
    inspect_parser.set_defaults(func=model_inspect_command)
    list_parser = model_sub.add_parser("list", help="List ArcLM model support metadata")
    list_parser.add_argument("--status", choices=["official", "experimental", "compatible_unverified", "unsupported"])
    list_parser.set_defaults(func=model_list_command)
    load_check = model_sub.add_parser("load-check", help="Check whether a model can be loaded")
    _add_model_options(load_check)
    load_check.set_defaults(func=model_load_check_command)
    certify_parser = model_sub.add_parser("certify", help="Run a model-family certification protocol")
    certify_parser.add_argument("source")
    certify_parser.add_argument("--family", default="unknown")
    certify_parser.add_argument("--revision")
    certify_parser.add_argument("--device", default="cpu")
    certify_parser.add_argument("--max-steps", type=int, default=1)
    certify_parser.add_argument("--no-training", action="store_true")
    certify_parser.add_argument("--json", action="store_true")
    certify_parser.set_defaults(func=model_certify_command)

    checkpoint_parser = subparsers.add_parser("checkpoint", help="Inspect and verify ArcLM checkpoints")
    checkpoint_sub = checkpoint_parser.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_inspect = checkpoint_sub.add_parser("inspect", help="Inspect a checkpoint without loading unsafe weights")
    checkpoint_inspect.add_argument("path")
    checkpoint_inspect.add_argument("--trust", default="safe", choices=["safe", "trusted_local", "legacy_unsafe"])
    checkpoint_inspect.add_argument("--json", action="store_true")
    checkpoint_inspect.set_defaults(func=checkpoint_inspect_command)
    checkpoint_verify = checkpoint_sub.add_parser("verify", help="Verify a checkpoint manifest and integrity hashes")
    checkpoint_verify.add_argument("path")
    checkpoint_verify.add_argument("--trust", default="safe", choices=["safe", "trusted_local", "legacy_unsafe"])
    checkpoint_verify.add_argument("--json", action="store_true")
    checkpoint_verify.set_defaults(func=checkpoint_verify_command)

    train_parser = subparsers.add_parser("train", help="Train a native ArcLM causal LM")
    train_parser.add_argument("--config")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--output", default="models/trained_model.pth")
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument("--mode", default="pretrain", choices=["pretrain", "finetune", "continue_training"])
    train_parser.add_argument("--embed-dim", type=int, default=64)
    train_parser.add_argument("--num-blocks", type=int, default=2)
    train_parser.add_argument("--block-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--tokenizer-type", default="word", choices=["word", "sentencepiece"])
    train_parser.add_argument("--max-vocab", type=int, default=50000)
    train_parser.add_argument("--validation-split", type=float, default=0.0)
    train_parser.add_argument("--training-log-interval", type=int, default=50)
    train_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    train_parser.add_argument("--json", action="store_true")
    train_parser.set_defaults(func=train_command)

    evaluate_parser = subparsers.add_parser("evaluate", aliases=["eval"], help="Evaluate a native ArcLM checkpoint")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--output", default="metrics_report.json")
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    evaluate_parser.add_argument("--json", action="store_true")
    evaluate_parser.set_defaults(func=evaluate_command)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a native ArcLM checkpoint")
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--prompt", required=True, nargs="+")
    generate_parser.add_argument("--length", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=None)
    generate_parser.add_argument("--batch-size", type=int, default=1)
    generate_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    generate_parser.add_argument("--json", action="store_true")
    generate_parser.set_defaults(func=generate_command)

    run_parser = subparsers.add_parser("run", help="Run a configuration-driven ArcLM workflow")
    run_parser.add_argument("config")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--stage", action="append")
    run_parser.add_argument("--json", action="store_true")
    run_parser.set_defaults(func=workflow_run_command)

    runs_parser = subparsers.add_parser("runs", help="Inspect local ArcLM runs")
    runs_sub = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_sub.add_parser("list")
    runs_list.add_argument("--output-dir", default="runs")
    runs_list.add_argument("--json", action="store_true")
    runs_list.set_defaults(func=runs_list_command)
    runs_inspect = runs_sub.add_parser("inspect")
    runs_inspect.add_argument("path")
    runs_inspect.set_defaults(func=runs_inspect_command)

    cache_parser = subparsers.add_parser("cache", help="Inspect and clear ArcLM caches")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_inspect = cache_sub.add_parser("inspect")
    cache_inspect.add_argument("cache_dir")
    cache_inspect.set_defaults(func=cache_inspect_command)
    cache_clear = cache_sub.add_parser("clear")
    cache_clear.add_argument("cache_dir")
    cache_clear.add_argument("--key")
    cache_clear.set_defaults(func=cache_clear_command)

    plugins_parser = subparsers.add_parser("plugins", help="Inspect local extension plugins")
    plugins_sub = plugins_parser.add_subparsers(dest="plugins_command", required=True)
    plugins_list = plugins_sub.add_parser("list")
    plugins_list.set_defaults(func=plugins_list_command)
    return parser


def _add_data_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("input")
    parser.add_argument("--format", choices=["json", "jsonl", "csv", "txt"])
    parser.add_argument("--json", action="store_true")
    return parser


def _add_validation_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--schema", required=True, choices=["text", "prompt_completion", "instruction", "conversation"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--check-duplicates", action="store_true")
    parser.add_argument("--duplicate-field")
    return parser


def _add_model_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("source")
    parser.add_argument("--task", default="causal-lm", choices=["causal-lm"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the ArcLM CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging("quiet" if args.quiet else "verbose" if args.verbose else "normal")
    if args.run:
        return run_command(args)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except ConfigurationError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR
    except DatasetValidationError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATASET_VALIDATION_ERROR
    except UnsupportedModelError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED_MODEL
    except ModelLoadError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MODEL_LOAD_ERROR
    except TrainingError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_TRAINING_ERROR
    except CheckpointError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT_ERROR
    except OptionalDependencyError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_OPTIONAL_DEPENDENCY_MISSING
    except ArcLMError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_GENERAL_FAILURE
    except (FileNotFoundError, ValueError) as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_GENERAL_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
