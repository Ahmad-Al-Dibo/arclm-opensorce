"""Checkpoint inspection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ._version import __version__
from .exceptions import CheckpointError


REQUIRED_NATIVE_KEYS = {"model_state_dict", "config", "vocab_size"}


@dataclass
class CheckpointInspectionReport:
    """Structured checkpoint reliability report."""

    path: str
    exists: bool
    load_attempted: bool = False
    trusted_pickle_required: bool = True
    is_native_arclm: bool = False
    has_model_state: bool = False
    has_optimizer_state: bool = False
    has_tokenizer_metadata: bool = False
    has_vocab_mappings: bool = False
    checkpoint_version: Optional[str] = None
    current_version: str = __version__
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_loadable(self) -> bool:
        """Return whether the checkpoint looks loadable."""

        return self.exists and self.has_model_state and not self.errors

    def summary(self) -> str:
        """Return a compact human-readable summary."""

        return (
            f"path={self.path} exists={self.exists} native={self.is_native_arclm} "
            f"model_state={self.has_model_state} tokenizer={self.has_tokenizer_metadata or self.has_vocab_mappings}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report."""

        data = asdict(self)
        data["is_loadable"] = self.is_loadable
        return data

    def raise_for_errors(self) -> None:
        """Raise :class:`CheckpointError` when inspection found errors."""

        if self.errors:
            raise CheckpointError("; ".join(self.errors))


def inspect_checkpoint(path: str | Path, *, trusted: bool = False, map_location: str = "cpu") -> CheckpointInspectionReport:
    """Inspect a native ArcLM/PyTorch checkpoint.

    Parameters:
        path: Checkpoint path.
        trusted: Whether ArcLM may deserialize the checkpoint with PyTorch.
            PyTorch checkpoint files may use pickle. Only set this for files
            from trusted sources.
        map_location: Device mapping passed to `torch.load` when trusted.
    """

    checkpoint_path = Path(path)
    report = CheckpointInspectionReport(path=str(checkpoint_path), exists=checkpoint_path.exists())
    if not checkpoint_path.exists():
        report.errors.append(f"Checkpoint does not exist: {checkpoint_path}")
        return report
    if not trusted:
        report.warnings.append(
            "Checkpoint exists but was not deserialized because trusted=False. "
            "Set trusted=True only for checkpoints from trusted sources."
        )
        return report

    report.load_attempted = True
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        report.errors.append(f"Could not deserialize checkpoint: {exc}")
        return report
    if not isinstance(checkpoint, dict):
        report.errors.append("Checkpoint payload is not a dictionary.")
        return report

    keys = set(checkpoint)
    report.has_model_state = "model_state_dict" in keys or all(hasattr(value, "shape") for value in checkpoint.values())
    report.is_native_arclm = REQUIRED_NATIVE_KEYS.issubset(keys)
    report.has_optimizer_state = checkpoint.get("optimizer_state_dict") is not None
    report.has_tokenizer_metadata = checkpoint.get("tokenizer_metadata") is not None or checkpoint.get("tokenizer") is not None
    report.has_vocab_mappings = checkpoint.get("stoi") is not None and checkpoint.get("itos") is not None
    report.config = dict(checkpoint.get("config") or {})
    report.checkpoint_version = (
        str(checkpoint.get("arclm_version"))
        if checkpoint.get("arclm_version") is not None
        else str(report.config.get("arclm_version")) if report.config.get("arclm_version") is not None else None
    )
    if report.is_native_arclm and not (report.has_tokenizer_metadata or report.has_vocab_mappings):
        report.warnings.append("Native checkpoint is missing tokenizer metadata or vocab mappings.")
    if report.checkpoint_version and report.checkpoint_version != __version__:
        report.warnings.append(
            f"Checkpoint was created by ArcLM {report.checkpoint_version}; current version is {__version__}."
        )
    missing = REQUIRED_NATIVE_KEYS - keys
    if "model_state_dict" not in keys and not report.has_model_state:
        report.errors.append("Checkpoint is missing model_state_dict.")
    elif missing and any(key in keys for key in REQUIRED_NATIVE_KEYS):
        report.warnings.append("Checkpoint is missing native key(s): " + ", ".join(sorted(missing)) + ".")
    return report


def load_trusted_checkpoint(path: str | Path, *, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a trusted checkpoint after structured inspection."""

    report = inspect_checkpoint(path, trusted=True, map_location=map_location)
    report.raise_for_errors()
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise CheckpointError("Checkpoint payload is not a dictionary.")
    return checkpoint


__all__ = ["CheckpointInspectionReport", "inspect_checkpoint", "load_trusted_checkpoint"]
