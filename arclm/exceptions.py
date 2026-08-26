"""ArcLM exception hierarchy."""

from __future__ import annotations


class ArcLMError(Exception):
    """Base class for ArcLM-specific errors."""

    def __init__(self, message: str, *, subsystem: str = "core", action: str | None = None, cause: str | None = None):
        self.message = str(message)
        self.subsystem = str(subsystem)
        self.action = action
        self.cause = cause
        parts = [f"[ArcLM:{self.subsystem}] {self.message}"]
        if self.cause:
            parts.append(f"Cause: {self.cause}")
        if self.action:
            parts.append(f"Action: {self.action}")
        super().__init__(" ".join(parts))


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


class ArtifactError(ArcLMError, RuntimeError):
    """Raised when an ArcLM native artifact cannot be read or written."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when an ArcLM native artifact fails integrity validation."""


class ArtifactVersionError(ArtifactError):
    """Raised when an artifact version cannot be read safely."""


class OptionalDependencyError(ArcLMError, ImportError):
    """Raised when an optional dependency is required but missing."""


class RuntimePlanningError(ArcLMError, RuntimeError):
    """Raised when runtime discovery or planning fails."""


__all__ = [
    "ArcLMError",
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactVersionError",
    "CheckpointError",
    "ConfigurationError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetValidationError",
    "ModelCompatibilityError",
    "ModelError",
    "ModelLoadError",
    "OptionalDependencyError",
    "RuntimePlanningError",
    "TrainingError",
    "UnsupportedModelError",
]
