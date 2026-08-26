"""Intentional research extension points over the shared ArcLM engine."""

from __future__ import annotations

from ..architectures import Architecture, ArchitectureCapabilities, ArchitectureKind, ArchitectureRegistry, CapabilitySupport, architectures
from ..core.backend import BackendContract, TorchBackend
from ..datasets.torch_dataset import TextDataset, create_dataloader
from ..runtime import Runtime
from ..tokenizers import Tokenizer, TokenizerEngine
from ..training import (
    AdapterStrategy,
    BaseStrategy,
    CallbackManager,
    CheckpointManager,
    ConsoleProgress,
    EvaluationResult,
    Evaluator,
    FineTuningConfig,
    FullFineTuneStrategy,
    LossFunction,
    NextTokenLoss,
    PretrainStrategy,
    StepMetrics,
    Trainer,
    TrainingCallback,
    TrainingConfig,
    TrainingEngine,
    TrainingEvent,
    TrainingPlan,
    TrainingResult,
    TrainingStrategy,
    configure_trainable_parameters,
)

__all__ = [
    "AdapterStrategy",
    "Architecture",
    "ArchitectureCapabilities",
    "ArchitectureKind",
    "ArchitectureRegistry",
    "BackendContract",
    "BaseStrategy",
    "CallbackManager",
    "CapabilitySupport",
    "CheckpointManager",
    "ConsoleProgress",
    "EvaluationResult",
    "Evaluator",
    "FineTuningConfig",
    "FullFineTuneStrategy",
    "LossFunction",
    "NextTokenLoss",
    "PretrainStrategy",
    "Runtime",
    "StepMetrics",
    "TextDataset",
    "Tokenizer",
    "TokenizerEngine",
    "TorchBackend",
    "Trainer",
    "TrainingCallback",
    "TrainingConfig",
    "TrainingEngine",
    "TrainingEvent",
    "TrainingPlan",
    "TrainingResult",
    "TrainingStrategy",
    "architectures",
    "configure_trainable_parameters",
    "create_dataloader",
]
