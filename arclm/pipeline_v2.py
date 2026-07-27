"""
Compatibility imports for the former unified pipeline module.
"""

from .training import ModelAdapter, PreTrainedModelLoader, StoppingCriteria, UnifiedPipeline
from .pipeline import build_model, build_optimizer_with_discriminative_lr, build_trainer

__all__ = [
    "ModelAdapter",
    "PreTrainedModelLoader",
    "StoppingCriteria",
    "UnifiedPipeline",
    "build_model",
    "build_optimizer_with_discriminative_lr",
    "build_trainer",
]
