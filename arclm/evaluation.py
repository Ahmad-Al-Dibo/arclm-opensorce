"""Serializable evaluation framework for causal language-model workflows."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch

from ._version import __version__
from .exceptions import DatasetError
from .inference import GenerationConfig, generate


MetricCallable = Callable[[Any, list[dict[str, Any]]], dict[str, float]]


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation configuration."""

    metrics: list[str] = field(default_factory=lambda: ["loss", "perplexity"])
    text_field: str = "text"
    batch_size: int = 1
    max_length: Optional[int] = None
    include_examples: bool = False
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Serializable evaluation report."""

    metrics: dict[str, Any]
    config: dict[str, Any]
    total_records: int
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    report_type: str = "evaluation_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    model: Any,
    dataset: Iterable[Mapping[str, Any]],
    metrics: Sequence[str | MetricCallable] | None = None,
    *,
    config: EvaluationConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> EvaluationReport:
    """Evaluate a model bundle or native loaded model without gradients."""

    rows = [dict(row) for row in dataset]
    if not rows:
        raise DatasetError("Evaluation dataset is empty.")
    cfg = config or EvaluationConfig(metrics=[metric for metric in metrics if isinstance(metric, str)] if metrics else ["loss", "perplexity"])
    requested = list(metrics or cfg.metrics)
    started = time.perf_counter()
    was_training = getattr(getattr(model, "model", model), "training", None)
    if hasattr(getattr(model, "model", model), "eval"):
        getattr(model, "model", model).eval()
    output_metrics: dict[str, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []
    examples: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for metric in requested:
                if callable(metric):
                    output_metrics.update(metric(model, rows))
                elif metric in {"loss", "perplexity", "token_accuracy"}:
                    output_metrics.update(_loss_metrics(model, rows, cfg.text_field, cfg.max_length))
                elif metric == "exact_match":
                    output_metrics["exact_match"] = _exact_match(model, rows, generation_config)
                elif metric in {"generation_length", "latency", "tokens_per_second"}:
                    generation = generate(
                        model,
                        prompts=[str(row.get(cfg.text_field, "")) for row in rows],
                        config=generation_config or GenerationConfig(max_new_tokens=4),
                        batch_size=cfg.batch_size,
                    )
                    output_metrics["generation_length_mean"] = sum(len(item.split()) for item in generation.outputs) / len(generation.outputs)
                    output_metrics["latency_seconds"] = generation.latency_seconds
                    output_metrics["tokens_per_second"] = generation.tokens_per_second
                else:
                    warnings.append(f"Unknown metric {metric!r} skipped.")
    finally:
        target = getattr(model, "model", model)
        if was_training and hasattr(target, "train"):
            target.train()
    for key, value in list(output_metrics.items()):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            errors.append(f"Metric {key} is not finite: {value}")
    if cfg.include_examples:
        examples = [{"index": index, "fields": sorted(row)} for index, row in enumerate(rows[:10])]
    return EvaluationReport(output_metrics, cfg.to_dict(), len(rows), time.perf_counter() - started, warnings, errors, examples)


def _loss_metrics(model_bundle: Any, rows: list[dict[str, Any]], text_field: str, max_length: Optional[int]) -> dict[str, Any]:
    if not hasattr(model_bundle, "tokenizer") or not hasattr(model_bundle, "model"):
        return {"loss": None, "perplexity": None, "token_accuracy": None}
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = getattr(model_bundle, "device", torch.device("cpu"))
    losses: list[float] = []
    correct = 0
    total = 0
    for row in rows:
        text = str(row.get(text_field, ""))
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if encoded["input_ids"].shape[-1] < 2:
            continue
        outputs = model(**encoded, labels=encoded["input_ids"])
        loss = getattr(outputs, "loss", None)
        if loss is not None:
            losses.append(float(loss.detach().cpu()))
        logits = getattr(outputs, "logits", None)
        if logits is not None:
            predictions = logits[:, :-1].argmax(dim=-1)
            labels = encoded["input_ids"][:, 1:]
            correct += int((predictions == labels).sum().detach().cpu())
            total += int(labels.numel())
    mean_loss = sum(losses) / len(losses) if losses else None
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss is not None and mean_loss < 50 else None,
        "token_accuracy": correct / total if total else None,
    }


def _exact_match(model: Any, rows: list[dict[str, Any]], generation_config: Optional[GenerationConfig]) -> float:
    examples = [row for row in rows if "prompt" in row and "completion" in row]
    if not examples:
        return 0.0
    generation = generate(model, prompts=[str(row["prompt"]) for row in examples], config=generation_config or GenerationConfig(max_new_tokens=8), batch_size=1)
    matches = sum(1 for row, output in zip(examples, generation.outputs) if str(row["completion"]).strip() == output.strip())
    return matches / len(examples)


__all__ = ["EvaluationConfig", "EvaluationReport", "evaluate"]
