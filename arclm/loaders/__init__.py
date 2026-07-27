"""External model loading and adaptation APIs."""

from .adapters import (
    adapt_for_training,
    config_from_checkpoint,
    validate_tokenizer_compatibility,
)
from .base import AdaptedModelBundle, CheckpointLoader, LoadedCheckpoint
from .registry import LoaderRegistry, create_default_registry, load_external_model
from .smart_loader import (
    DefaultModelInspector,
    LoadPlan,
    ModelInspector,
    SmartLoader,
    inspect_model_source,
    register_model_inspector,
)

__all__ = [
    "AdaptedModelBundle",
    "CheckpointLoader",
    "DefaultModelInspector",
    "LoadPlan",
    "LoadedCheckpoint",
    "LoaderRegistry",
    "ModelInspector",
    "SmartLoader",
    "adapt_for_training",
    "config_from_checkpoint",
    "create_default_registry",
    "inspect_model_source",
    "load_external_model",
    "register_model_inspector",
    "validate_tokenizer_compatibility",
]
