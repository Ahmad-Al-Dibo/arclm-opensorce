"""
Diagnostics and concept benchmarks for trained language models.
"""

from dataclasses import dataclass

import torch

from .config import Config
from .data import prepare_data
from .generator import Generator
from .pipeline import build_model, build_trainer


@dataclass
class TopKPrediction:
    """One next-token prediction."""

    rank: int
    token: str
    probability: float


@dataclass
class ConceptBenchmarkCase:
    """Expected concept relationships for a prompt."""

    prompt: str
    expected_concepts: list


@dataclass
class ConceptBenchmarkResult:
    """Concept benchmark score for one prompt."""

    prompt: str
    expected_concepts: list
    predicted_tokens: list
    matched_concepts: list
    score: float


@dataclass
class LongContextResult:
    """Summary for one long-context training/evaluation run."""

    block_size: int
    validation_loss: float
    concept_score: float
    generated_sample: str


@dataclass
class MetricsReport:
    """Comprehensive metrics report for model evaluation."""

    perplexity: float
    avg_loss: float
    total_tokens: int
    correct_predictions: int
    accuracy: float
    timestamp: str = ""
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "perplexity": self.perplexity,
            "avg_loss": self.avg_loss,
            "total_tokens": self.total_tokens,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp,
        }


DEFAULT_CONCEPT_BENCHMARKS = [
    ConceptBenchmarkCase("python", ["programming", "language", "software"]),
    ConceptBenchmarkCase("machine learning", ["data", "models", "neural"]),
    ConceptBenchmarkCase("database", ["storage", "query", "sql"]),
    ConceptBenchmarkCase("united states", ["government", "country"]),
    ConceptBenchmarkCase("white house", ["president", "administration"]),
]


def _encode_prompt(prompt, stoi, tokenizer=None):
    if tokenizer is not None and hasattr(tokenizer, "encode_text"):
        return tokenizer.encode_text(prompt.lower())

    unk_idx = stoi.get("<UNK>", 0)
    return [stoi.get(token, unk_idx) for token in prompt.lower().split() if token]


def predict_top_k(model, stoi, itos, block_size, device, prompt, k=5, tokenizer=None):
    """Return the top-k next-token probabilities for a prompt."""
    model.eval()
    encoded = _encode_prompt(prompt, stoi, tokenizer=tokenizer)
    if not encoded:
        return []

    idx = torch.tensor([encoded[-block_size:]], device=torch.device(device))
    with torch.no_grad():
        logits = model(idx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, min(k, probs.shape[-1]), dim=-1)

    predictions = []
    for rank, (probability, token_idx) in enumerate(
        zip(top_probs[0].tolist(), top_indices[0].tolist()),
        start=1,
    ):
        predictions.append(
            TopKPrediction(
                rank=rank,
                token=itos.get(token_idx, "<UNK>"),
                probability=probability,
            )
        )
    return predictions


def format_top_k_predictions(prompt, predictions):
    """Create a readable diagnostic report for one prompt."""
    lines = [f"Prompt: {prompt}"]
    if not predictions:
        lines.append("  No predictions; prompt contains no usable tokens.")
        return "\n".join(lines)

    for prediction in predictions:
        lines.append(
            f"  {prediction.rank}. {prediction.token} "
            f"({prediction.probability * 100:.2f}%)"
        )
    return "\n".join(lines)


def score_concept_relationships(
    model,
    stoi,
    itos,
    block_size,
    device,
    benchmark_cases=None,
    k=10,
    tokenizer=None,
):
    """Score expected concepts against top-k next-token predictions."""
    cases = benchmark_cases or DEFAULT_CONCEPT_BENCHMARKS
    results = []

    for case in cases:
        predictions = predict_top_k(
            model,
            stoi,
            itos,
            block_size,
            device,
            case.prompt,
            k=k,
            tokenizer=tokenizer,
        )
        predicted_tokens = [prediction.token.lower() for prediction in predictions]
        expected = [concept.lower() for concept in case.expected_concepts]
        matched = [concept for concept in expected if concept in predicted_tokens]
        score = len(matched) / len(expected) if expected else 0.0
        results.append(
            ConceptBenchmarkResult(
                prompt=case.prompt,
                expected_concepts=case.expected_concepts,
                predicted_tokens=predicted_tokens,
                matched_concepts=matched,
                score=score,
            )
        )

    overall_score = (
        sum(result.score for result in results) / len(results)
        if results
        else 0.0
    )
    return {
        "overall_score": overall_score,
        "results": results,
    }


def format_concept_benchmark_report(report):
    """Create a readable concept benchmark report."""
    lines = [f"Overall concept score: {report['overall_score'] * 100:.2f}%"]
    for result in report["results"]:
        expected = ", ".join(result.expected_concepts)
        matched = ", ".join(result.matched_concepts) or "-"
        predicted = ", ".join(result.predicted_tokens)
        lines.append(
            f"  {result.prompt}: {result.score * 100:.2f}% "
            f"(matched: {matched}; expected: {expected}; top predictions: {predicted})"
        )
    return "\n".join(lines)


def format_tokenizer_coverage_report(coverage):
    """Create a readable tokenizer coverage report."""
    lines = [
        f"Total tokens checked: {coverage['total_tokens']}",
        f"Unknown tokens: {coverage['unknown_tokens']} "
        f"({coverage['unknown_rate'] * 100:.2f}%)",
        f"Vocabulary coverage: {coverage['vocab_coverage'] * 100:.2f}% "
        f"of unique tokens",
        f"Vocabulary size: {coverage['vocab_size']}",
    ]

    if coverage["top_unknown_tokens"]:
        top_unknown = ", ".join(
            f"{token} ({count})"
            for token, count in coverage["top_unknown_tokens"]
        )
        lines.append(f"Top unknown tokens: {top_unknown}")

    if coverage["rare_training_tokens"]:
        rare_tokens = ", ".join(
            f"{token} ({count})"
            for token, count in coverage["rare_training_tokens"]
        )
        lines.append(
            f"Rare training tokens below {coverage['rare_threshold']}: {rare_tokens}"
        )

    return "\n".join(lines)


def build_training_diagnostics_report(model, data, config):
    """Build tokenizer, top-k, and concept benchmark diagnostics for a run."""
    sections = []

    coverage = data.tokenizer.analyze_coverage(
        data.val_tokens or data.tokens,
        rare_threshold=config.tokenizer_rare_threshold,
    )
    sections.append("Tokenizer coverage:")
    sections.append(format_tokenizer_coverage_report(coverage))

    sections.append("Top-K prediction diagnostics:")
    for prompt in config.diagnostic_prompts:
        predictions = predict_top_k(
            model,
            data.tokenizer.stoi,
            data.tokenizer.itos,
            config.block_size,
            config.device,
            prompt,
            k=config.diagnostic_top_k,
            tokenizer=data.tokenizer,
        )
        sections.append(format_top_k_predictions(prompt, predictions))

    concept_report = score_concept_relationships(
        model,
        data.tokenizer.stoi,
        data.tokenizer.itos,
        config.block_size,
        config.device,
        k=config.concept_benchmark_top_k,
        tokenizer=data.tokenizer,
    )
    sections.append("Concept relationship benchmark:")
    sections.append(format_concept_benchmark_report(concept_report))

    return "\n\n".join(sections)


def format_long_context_results(results):
    """Create a readable long-context comparison report."""
    lines = ["Long context evaluation:"]
    for result in results:
        lines.append(
            f"block_size={result.block_size} | "
            f"val_loss={result.validation_loss:.4f} | "
            f"concept_score={result.concept_score * 100:.2f}%"
        )
        lines.append(f"sample: {result.generated_sample}")
    return "\n".join(lines)


def run_long_context_evaluation(
    base_config,
    block_sizes=(32, 64, 128),
    benchmark_cases=None,
    sample_prompt="Programming is",
):
    """Train and compare multiple block sizes by validation loss and concepts."""
    results = []

    for block_size in block_sizes:
        config = Config(**base_config.to_dict())
        config.block_size = block_size
        torch.manual_seed(config.seed)

        data = prepare_data(config)
        model = build_model(config, data.vocab_size)
        trainer = build_trainer(model, config)
        trainer.train(
            data.train_loader,
            config.num_epochs,
            val_loader=data.val_loader,
            early_stopping_patience=config.early_stopping_patience,
        )

        generator = Generator(
            model,
            data.tokenizer.stoi,
            data.tokenizer.itos,
            config.block_size,
            torch.device(config.device),
        )
        concept_report = score_concept_relationships(
            model,
            data.tokenizer.stoi,
            data.tokenizer.itos,
            config.block_size,
            config.device,
            benchmark_cases=benchmark_cases,
            k=config.concept_benchmark_top_k,
            tokenizer=data.tokenizer,
        )
        generated_sample = generator.generate_string(
            sample_prompt,
            max_new_tokens=config.diagnostic_sample_tokens,
            temperature=0.9,
            repetition_penalty=1.2,
            top_p=0.9,
        )
        validation_loss = (
            trainer.val_losses[-1]
            if trainer.val_losses
            else float("nan")
        )
        results.append(
            LongContextResult(
                block_size=block_size,
                validation_loss=validation_loss,
                concept_score=concept_report["overall_score"],
                generated_sample=generated_sample,
            )
        )

    return results


# ============================================================================
# P4-3: COMPREHENSIVE METRICS (Production Diagnostics)
# ============================================================================

def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    Lower is better (1.0 = perfect prediction)
    
    Args:
        loss (float): Cross-entropy loss value
        
    Returns:
        float: Perplexity value
    """
    import math
    if loss < 0 or math.isnan(loss) or math.isinf(loss):
        return float('inf')
    return math.exp(loss)


def calculate_metrics(model, val_loader, config, device="cpu"):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        config: Config object
        device: Device for computation
        
    Returns:
        MetricsReport: Metrics dictionary
    """
    import math
    from datetime import datetime
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='mean'
            )
            
            # Accumulate metrics
            batch_size = y.numel()
            total_loss += loss.item() * batch_size
            total_tokens += batch_size
            
            # Count correct predictions
            predictions = logits.argmax(dim=-1)
            correct = (predictions == y).sum().item()
            correct_predictions += correct
    
    # Calculate final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = calculate_perplexity(avg_loss)
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
    
    return MetricsReport(
        perplexity=perplexity,
        avg_loss=avg_loss,
        total_tokens=total_tokens,
        correct_predictions=correct_predictions,
        accuracy=accuracy,
        timestamp=datetime.now().isoformat(),
    )


def export_metrics_to_json(metrics_report, filepath):
    """
    Export metrics report to JSON file.
    
    Args:
        metrics_report (MetricsReport): Metrics to export
        filepath (str): Output file path
    """
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metrics_report.to_dict(), f, indent=2)


def export_metrics_to_markdown(metrics_report, filepath):
    """
    Export metrics report to Markdown file.
    
    Args:
        metrics_report (MetricsReport): Metrics to export
        filepath (str): Output file path
    """
    md_content = f"""# Evaluation Metrics Report

## Summary

- **Perplexity:** {metrics_report.perplexity:.2f}
- **Average Loss:** {metrics_report.avg_loss:.4f}
- **Accuracy:** {metrics_report.accuracy * 100:.2f}%
- **Total Tokens:** {metrics_report.total_tokens:,}
- **Correct Predictions:** {metrics_report.correct_predictions:,}
- **Timestamp:** {metrics_report.timestamp}

## Interpretation

- **Perplexity**: Lower is better. A perplexity of N means the model is as confused as if it had to choose uniformly from N options.
- **Accuracy**: Percentage of correctly predicted next tokens.
- **Loss**: Cross-entropy loss value (lower is better).

---
Generated: {metrics_report.timestamp}
"""
    
    with open(filepath, 'w') as f:
        f.write(md_content)
