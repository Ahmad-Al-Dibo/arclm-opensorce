"""ArcLM exception hierarchy."""

from __future__ import annotations


class ArcLMError(Exception):
    """Base class for ArcLM-specific errors."""


class ConfigurationError(ArcLMError, ValueError):
    """Raised when configuration values are invalid."""


class DatasetError(ArcLMError):
    """Base class for dataset errors."""


class DatasetValidationError(DatasetError, ValueError):
    """Raised when records do not satisfy a requested schema."""


class DatasetFormatError(DatasetError, ValueError):
    """Raised when dataset input cannot be parsed into records."""


class ModelError(ArcLMError):
    """Base class for model errors."""


class UnsupportedModelError(ModelError, ValueError):
    """Raised when a model is outside ArcLM's supported workflow."""


class ModelLoadError(ModelError, RuntimeError):
    """Raised when model loading fails."""


class ModelCompatibilityError(ModelError, ValueError):
    """Raised when a model is not compatible with a requested task."""


class TrainingError(ArcLMError, RuntimeError):
    """Raised when training cannot be completed."""


class CheckpointError(ArcLMError, RuntimeError):
    """Raised when checkpoint loading or saving fails."""


class OptionalDependencyError(ArcLMError, ImportError):
    """Raised when an optional dependency is required but missing."""


__all__ = [
    "ArcLMError",
    "CheckpointError",
    "ConfigurationError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetValidationError",
    "ModelCompatibilityError",
    "ModelError",
    "ModelLoadError",
    "OptionalDependencyError",
    "TrainingError",
    "UnsupportedModelError",
]
