"""Checkpoint inspection helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from ._version import __version__
from .exceptions import CheckpointError
from .security import LoadingPolicy


REQUIRED_NATIVE_KEYS = {"model_state_dict", "config", "vocab_size"}
CHECKPOINT_FORMAT_VERSION = "1"
CHECKPOINT_MANIFEST = "manifest.json"


@dataclass
class CheckpointInspectionReport:
    """Structured checkpoint reliability report."""

    path: str
    exists: bool
    load_attempted: bool = False
    loading_policy: str = "safe"
    format_version: Optional[str] = None
    trusted_pickle_required: bool = False
    is_directory_checkpoint: bool = False
    is_native_arclm: bool = False
    has_model_state: bool = False
    model_weight_format: Optional[str] = None
    has_optimizer_state: bool = False
    optimizer_pickle_required: bool = False
    has_tokenizer_metadata: bool = False
    has_vocab_mappings: bool = False
    checkpoint_version: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    integrity_hashes: Dict[str, str] = field(default_factory=dict)
    unexpected_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    current_version: str = __version__
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_loadable(self) -> bool:
        """Return whether the checkpoint looks loadable."""

        return self.exists and self.has_model_state and not self.errors

    @property
    def is_verified(self) -> bool:
        """Return whether integrity checks passed."""

        return self.exists and not self.errors and not self.missing_files

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


def inspect_checkpoint(
    path: str | Path,
    *,
    trusted: bool = False,
    map_location: str = "cpu",
    trust: str | LoadingPolicy | None = None,
) -> CheckpointInspectionReport:
    """Inspect an ArcLM checkpoint without unsafe loading by default.

    Parameters:
        path: Checkpoint path.
        trusted: Whether ArcLM may deserialize the checkpoint with PyTorch.
            PyTorch checkpoint files may use pickle. Only set this for files
            from trusted sources.
        map_location: Device mapping passed to `torch.load` when trusted.
        trust: Explicit loading policy. ``safe`` is the default.
    """

    policy = LoadingPolicy.from_value("trusted_local" if trusted and trust is None else trust or "safe")
    checkpoint_path = Path(path)
    report = CheckpointInspectionReport(path=str(checkpoint_path), exists=checkpoint_path.exists(), loading_policy=policy.mode)
    if not checkpoint_path.exists():
        report.errors.append(f"Checkpoint does not exist: {checkpoint_path}")
        return report
    if checkpoint_path.is_dir():
        return _inspect_directory_checkpoint(checkpoint_path, report, policy)
    if not trusted:
        report.trusted_pickle_required = checkpoint_path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin"}
        if report.trusted_pickle_required:
            report.warnings.append(
                "Checkpoint exists but was not deserialized because the safe loading policy rejects legacy PyTorch pickle files."
            )
            report.errors.append(
                "Legacy PyTorch checkpoint files require pickle deserialization and are rejected in safe mode. "
                "Use trust='trusted_local' only for trusted local files."
            )
            return report
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


def verify_checkpoint(path: str | Path, *, trust: str | LoadingPolicy = "safe") -> CheckpointInspectionReport:
    """Inspect a checkpoint and raise when it is incomplete or unsafe."""

    report = inspect_checkpoint(path, trust=trust)
    report.raise_for_errors()
    return report


def write_checkpoint_manifest(
    checkpoint_dir: str | Path,
    *,
    model_config: Optional[Mapping[str, Any]] = None,
    source_model: Optional[str] = None,
    source_revision: Optional[str] = None,
    security_classification: str = "trusted_local",
) -> Path:
    """Create or refresh a versioned ArcLM checkpoint manifest and hashes."""

    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "arclm-checkpoint",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "arclm_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "architecture": (model_config or {}).get("architecture", "unknown"),
            "config_path": "model/config.json",
            "weights_path": "model/model.safetensors",
            "weights_format": "safetensors",
            "source_model": source_model,
            "source_revision": source_revision,
        },
        "training": {
            "state_path": "training/state.json",
            "optimizer_path": "training/optimizer.pt",
            "scheduler_path": "training/scheduler.pt",
            "optimizer_pickle_required": True,
        },
        "tokenizer": {"path": "tokenizer"},
        "security": {"classification": security_classification},
    }
    hashes: dict[str, str] = {}
    for relative in ["model/config.json", "model/model.safetensors", "training/state.json"]:
        target = root / relative
        if target.exists() and target.is_file():
            hashes[relative] = _sha256_file(target)
    manifest["hashes_path"] = "hashes.json"
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp_name:
        tmp = Path(tmp_name)
        (tmp / CHECKPOINT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (tmp / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
        for item in tmp.iterdir():
            shutil.move(str(item), str(root / item.name))
    return root / CHECKPOINT_MANIFEST


def load_trusted_checkpoint(path: str | Path, *, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a trusted checkpoint after structured inspection."""

    report = inspect_checkpoint(path, trusted=True, map_location=map_location)
    report.raise_for_errors()
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise CheckpointError("Checkpoint payload is not a dictionary.")
    return checkpoint


def _inspect_directory_checkpoint(root: Path, report: CheckpointInspectionReport, policy: LoadingPolicy) -> CheckpointInspectionReport:
    report.is_directory_checkpoint = True
    manifest_path = root / CHECKPOINT_MANIFEST
    if not manifest_path.exists():
        report.errors.append("Directory checkpoint is missing manifest.json.")
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.errors.append(f"Could not parse checkpoint manifest: {exc}")
        return report
    report.manifest = manifest
    report.format_version = str(manifest.get("format_version", ""))
    if manifest.get("format") != "arclm-checkpoint":
        report.errors.append("Unsupported checkpoint manifest format.")
    if report.format_version != CHECKPOINT_FORMAT_VERSION:
        report.errors.append(f"Unsupported checkpoint format version: {report.format_version!r}.")
    model_section = dict(manifest.get("model") or {})
    training_section = dict(manifest.get("training") or {})
    weights_path = model_section.get("weights_path")
    weights_format = model_section.get("weights_format")
    report.model_weight_format = weights_format
    report.checkpoint_version = str(manifest.get("arclm_version")) if manifest.get("arclm_version") else None
    if weights_format == "safetensors":
        report.has_model_state = True
        report.trusted_pickle_required = False
    elif weights_path:
        report.trusted_pickle_required = True
        if not policy.allow_pickle:
            report.errors.append("Model weights are not safetensors and are rejected by safe loading policy.")
    for key in ["config_path", "weights_path"]:
        relative = model_section.get(key)
        if relative and not (root / relative).exists():
            report.missing_files.append(str(relative))
    optimizer_path = training_section.get("optimizer_path")
    if optimizer_path and (root / optimizer_path).exists():
        report.has_optimizer_state = True
        report.optimizer_pickle_required = bool(training_section.get("optimizer_pickle_required", True))
    tokenizer_path = manifest.get("tokenizer", {}).get("path") if isinstance(manifest.get("tokenizer"), dict) else None
    report.has_tokenizer_metadata = bool(tokenizer_path and (root / tokenizer_path).exists())
    hashes_path = root / str(manifest.get("hashes_path", "hashes.json"))
    if hashes_path.exists():
        try:
            report.integrity_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            if policy.verify_hashes:
                _verify_hashes(root, report)
        except Exception as exc:
            report.errors.append(f"Could not verify checkpoint hashes: {exc}")
    expected = {CHECKPOINT_MANIFEST, "hashes.json", "model", "tokenizer", "training"}
    report.unexpected_files = sorted(item.name for item in root.iterdir() if item.name not in expected)
    if report.unexpected_files:
        report.warnings.append("Unexpected checkpoint file(s): " + ", ".join(report.unexpected_files))
    if report.missing_files:
        report.errors.append("Checkpoint is missing required file(s): " + ", ".join(report.missing_files))
    return report


def _verify_hashes(root: Path, report: CheckpointInspectionReport) -> None:
    for relative, expected in report.integrity_hashes.items():
        target = root / relative
        if not target.exists():
            report.errors.append(f"Hash entry references missing file: {relative}")
            continue
        actual = _sha256_file(target)
        if actual != expected:
            report.errors.append(f"Hash mismatch for {relative}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointInspectionReport",
    "inspect_checkpoint",
    "load_trusted_checkpoint",
    "verify_checkpoint",
    "write_checkpoint_manifest",
]
