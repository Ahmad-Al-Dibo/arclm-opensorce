"""Intentional research extension points over the shared ArcLM engine."""

from __future__ import annotations

from ..architectures import Architecture, ArchitectureCapabilities, ArchitectureKind, ArchitectureRegistry, CapabilitySupport, architectures
from ..core.backend import BackendContract, TorchBackend
from ..datasets import DataEngine, DataSource, FieldMapTransform, FunctionTransform, IngestionResult, ModelInputPreparer, Transform
from ..evaluation import EvaluationContext, EvaluationEngine, EvaluationReport, MetricEvaluator
from ..experiments import Experiment
from ..inspection import ArtifactInspector, DatasetInspector, ModelInspector, RuntimeInspector, TrainingInspector
from ..datasets.torch_dataset import TextDataset, create_dataloader
from ..runtime import Runtime
from ..tokenizers import ChatFormatter, ChatTemplate, Tokenizer, TokenizerEngine
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
    "ChatFormatter",
    "ChatTemplate",
    "ConsoleProgress",
    "DataEngine",
    "DataSource",
    "EvaluationResult",
    "EvaluationContext",
    "EvaluationEngine",
    "EvaluationReport",
    "Evaluator",
    "Experiment",
    "FieldMapTransform",
    "FineTuningConfig",
    "FunctionTransform",
    "FullFineTuneStrategy",
    "IngestionResult",
    "LossFunction",
    "MetricEvaluator",
    "ModelInputPreparer",
    "ModelInspector",
    "NextTokenLoss",
    "PretrainStrategy",
    "Runtime",
    "RuntimeInspector",
    "StepMetrics",
    "TextDataset",
    "Tokenizer",
    "TokenizerEngine",
    "TorchBackend",
    "Trainer",
    "Transform",
    "TrainingCallback",
    "TrainingConfig",
    "TrainingEngine",
    "TrainingEvent",
    "TrainingPlan",
    "TrainingResult",
    "TrainingStrategy",
    "TrainingInspector",
    "ArtifactInspector",
    "DatasetInspector",
    "architectures",
    "configure_trainable_parameters",
    "create_dataloader",
]
