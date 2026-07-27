"""Public ArcLM package interface.

ArcLM is a compact PyTorch toolkit for native causal language-model training,
checkpoint loading, dataset preparation, diagnostics, and Hugging Face SFT.
"""

from pathlib import Path

import torch

from ._version import __version__
from .config import Config, create_config
from .data import DataBundle, load_tokens, prepare_data, read_tokens, split_train_val
from .data_processor import DataProcessor, ProcessedDataset
from .dataset import TextDataset, create_dataloader
from .diagnostics import (
    ConceptBenchmarkCase,
    ConceptBenchmarkResult,
    DEFAULT_CONCEPT_BENCHMARKS,
    LongContextResult,
    MetricsReport,
    TopKPrediction,
    build_training_diagnostics_report,
    calculate_metrics,
    calculate_perplexity,
    export_metrics_to_json,
    export_metrics_to_markdown,
    format_concept_benchmark_report,
    format_long_context_results,
    format_tokenizer_coverage_report,
    format_top_k_predictions,
    predict_top_k,
    run_long_context_evaluation,
    score_concept_relationships,
)
from .generator import Generator
from .inference import DEFAULT_MODEL_PATH, LoadedModel, load_model, predict
from .instruction_dataset import InstructionDataset, create_instruction_dataloader
from .loaders import (
    AdaptedModelBundle,
    LoadPlan,
    LoadedCheckpoint,
    ModelInspector,
    SmartLoader,
    adapt_for_training,
    inspect_model_source,
    load_external_model,
    register_model_inspector,
    validate_tokenizer_compatibility,
)
from .external_inference import (
    ExternalLoadedModel,
    ExternalModelConfig,
    GenerationConfig,
    ModelSaveConfig,
    ModelSourceInfo,
    continue_native_training,
    fine_tune_external_model,
    fine_tune_native_model,
    inspect_model_source,
    load_any_model,
    load_external_for_inference,
    predict_external,
    save_loaded_model,
    train_native_model,
)
from .logics import (
    And,
    Biconditional,
    Implication,
    Not,
    Or,
    Sentence,
    Symbol,
    model_check,
)
from .model import ArcLM, MiniGPT
from .pipeline import (
    TrainingResult,
    build_model,
    build_trainer,
    checkpoint_is_compatible_for_continue_training,
    checkpoint_is_compatible_for_tuining,
    create_checkpoint_callback,
    create_epoch_checkpoint_callback,
    load_compatible_checkpoint,
    save_training_checkpoint,
    tokenizer_from_checkpoint,
    train_model,
)
from .regularization import (
    EarlyStopping,
    GeneralizationMonitor,
    L1Regularization,
    L2Regularization,
    LabelSmoothing,
    LearningRateScheduler,
    MixupAugmentation,
)
from .sft import SFTTrainingResult, train_sft
from .tokenizers import (
    SentencePieceTokenizer,
    Tokenizer,
    TokenizerFactory,
    create_tokenizer,
    get_tokenizer_from_config,
)
from .trainer import Trainer
from .training import (
    BaseModelAdapter,
    BaseModelLoader,
    BaseTrainingPipeline,
    ModelAdapter,
    PreTrainedModelLoader,
    StoppingCriteria,
    UnifiedPipeline,
)
from .utils import format_duration

__author__ = "Ahmad Al Dibo"

__all__ = [
    "ArcLM",
    "MiniGPT",
    "Config",
    "create_config",
    "Tokenizer",
    "SentencePieceTokenizer",
    "TokenizerFactory",
    "create_tokenizer",
    "get_tokenizer_from_config",
    "TextDataset",
    "create_dataloader",
    "DataBundle",
    "DataProcessor",
    "ProcessedDataset",
    "load_tokens",
    "prepare_data",
    "read_tokens",
    "split_train_val",
    "Trainer",
    "Generator",
    "DEFAULT_MODEL_PATH",
    "LoadedModel",
    "load_model",
    "predict",
    "build_model",
    "build_trainer",
    "train_model",
    "TrainingResult",
    "train_sft",
    "SFTTrainingResult",
    "ExternalModelConfig",
    "ExternalLoadedModel",
    "GenerationConfig",
    "ModelSaveConfig",
    "ModelSourceInfo",
    "inspect_model_source",
    "load_any_model",
    "load_external_for_inference",
    "predict_external",
    "fine_tune_external_model",
    "train_native_model",
    "fine_tune_native_model",
    "continue_native_training",
    "save_loaded_model",
    "checkpoint_is_compatible_for_continue_training",
    "checkpoint_is_compatible_for_tuining",
    "create_checkpoint_callback",
    "create_epoch_checkpoint_callback",
    "load_compatible_checkpoint",
    "save_training_checkpoint",
    "tokenizer_from_checkpoint",
    "UnifiedPipeline",
    "PreTrainedModelLoader",
    "ModelAdapter",
    "StoppingCriteria",
    "BaseTrainingPipeline",
    "BaseModelLoader",
    "BaseModelAdapter",
    "LoadedCheckpoint",
    "LoadPlan",
    "AdaptedModelBundle",
    "ModelInspector",
    "SmartLoader",
    "register_model_inspector",
    "load_external_model",
    "adapt_for_training",
    "validate_tokenizer_compatibility",
    "build_training_diagnostics_report",
    "calculate_metrics",
    "calculate_perplexity",
    "export_metrics_to_json",
    "export_metrics_to_markdown",
    "ConceptBenchmarkCase",
    "ConceptBenchmarkResult",
    "DEFAULT_CONCEPT_BENCHMARKS",
    "LongContextResult",
    "MetricsReport",
    "TopKPrediction",
    "format_concept_benchmark_report",
    "format_long_context_results",
    "format_tokenizer_coverage_report",
    "format_top_k_predictions",
    "predict_top_k",
    "run_long_context_evaluation",
    "score_concept_relationships",
    "format_duration",
    "L1Regularization",
    "L2Regularization",
    "EarlyStopping",
    "LearningRateScheduler",
    "GeneralizationMonitor",
    "MixupAugmentation",
    "LabelSmoothing",
    "create_instruction_dataloader",
    "InstructionDataset",
    "Sentence",
    "Symbol",
    "Not",
    "And",
    "Or",
    "Implication",
    "Biconditional",
    "model_check",
    "disable_debug",
    "enable_debug",
    "get_device",
    "get_version",
    "list_available_models",
    "load_training_checkpoint",
    "report_environment",
    "create_simple_interface_app",
    "run_simple_interface",
]


def get_version():
    """Return the installed ArcLM version."""
    return __version__


def get_device():
    """Return the preferred local torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_available_models(models_dir: str | Path = "models"):
    """List local ``.pth`` checkpoints in a models directory."""
    path = Path(models_dir)
    if not path.exists():
        return []
    return sorted(item.stem for item in path.glob("*.pth") if item.is_file())


def load_training_checkpoint(path: str | Path, device_type: str = "cpu"):
    """Load a trusted ArcLM/PyTorch training checkpoint.

    Only load checkpoints from trusted sources. PyTorch checkpoint files can
    execute code during deserialization.
    """
    return torch.load(
        path,
        map_location=torch.device(device_type),
        weights_only=False,
    )


def report_environment(models_dir: str | Path = "models") -> str:
    """Return a short human-readable runtime report."""
    device = get_device()
    device_note = "CUDA available" if device.type == "cuda" else "CUDA not available"
    models = list_available_models(models_dir)
    return (
        f"ArcLM {__version__} | device={device.type} ({device_note}) | "
        f"local_checkpoints={models}"
    )


def create_simple_interface_app():
    """Create the optional Flask app for ArcLM's simple web interface."""
    from .simple_interface import create_simple_interface_app as _create_app

    return _create_app()


def run_simple_interface(*args, **kwargs):
    """Run ArcLM's optional simple web interface."""
    from .simple_interface import run_simple_interface as _run_interface

    return _run_interface(*args, **kwargs)
