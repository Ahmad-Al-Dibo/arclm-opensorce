"""Experimental `.arcmodel` artifact support."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import __version__


@dataclass(frozen=True)
class ArcModelManifest:
    """Versioned manifest for a native ArcLM model artifact."""

    artifact_id: str
    format: str
    format_version: str
    schema_version: str
    minimum_reader_version: str
    architecture_id: str
    created_at: str
    dtype: str
    weights_file: str
    weights_hash: str
    tokenizer_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe manifest dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArcModelManifest":
        """Parse a manifest dictionary."""

        return cls(**data)


class ArcModelArtifact:
    """Reader/writer for `.arcmodel` directory artifacts."""

    MANIFEST = "manifest.json"
    CONFIG = "config.json"
    TOKENIZER = "tokenizer.json"
    WEIGHTS = "weights.pt"

    def __init__(self, path: str | Path):
        self.path = self.normalize_path(path)

    @staticmethod
    def normalize_path(path: str | Path) -> Path:
        """Normalize a user path to a `.arcmodel` directory path."""

        resolved = Path(path)
        if resolved.suffix != ".arcmodel":
            resolved = resolved.with_suffix(".arcmodel")
        return resolved

    @classmethod
    def save(
        cls,
        path: str | Path,
        *,
        model: Any,
        config: dict[str, Any],
        tokenizer: Any,
        architecture_id: str,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "ArcModelArtifact":
        """Write a native ArcLM model artifact."""

        artifact = cls(path)
        if artifact.path.exists():
            if not overwrite:
                raise FileExistsError(f"Artifact already exists: {artifact.path}")
            if artifact.path.is_dir():
                shutil.rmtree(artifact.path)
            else:
                artifact.path.unlink()
        artifact.path.mkdir(parents=True)

        config_path = artifact.path / cls.CONFIG
        tokenizer_path = artifact.path / cls.TOKENIZER
        weights_path = artifact.path / cls.WEIGHTS

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tokenizer_payload = tokenizer.to_json() if hasattr(tokenizer, "to_json") else {}
        tokenizer_path.write_text(json.dumps(tokenizer_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

        import torch

        torch.save(model.state_dict(), weights_path)
        manifest = ArcModelManifest(
            artifact_id=f"arcmodel-{uuid.uuid4()}",
            format="arcmodel",
            format_version="1",
            schema_version="1",
            minimum_reader_version=__version__,
            architecture_id=architecture_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            dtype=cls._model_dtype(model),
            weights_file=cls.WEIGHTS,
            weights_hash=cls._sha256(weights_path),
            tokenizer_hash=cls._sha256(tokenizer_path),
            metadata=dict(metadata or {}),
        )
        (artifact.path / cls.MANIFEST).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return artifact

    def manifest(self) -> ArcModelManifest:
        """Load and validate the artifact manifest."""

        manifest_path = self.path / self.MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ArcModelManifest.from_dict(data)
        self._validate_hashes(manifest)
        return manifest

    def read_config(self) -> dict[str, Any]:
        """Read model configuration."""

        return json.loads((self.path / self.CONFIG).read_text(encoding="utf-8"))

    def read_tokenizer(self) -> dict[str, Any]:
        """Read tokenizer payload."""

        return json.loads((self.path / self.TOKENIZER).read_text(encoding="utf-8"))

    def read_state_dict(self, map_location: Any = "cpu") -> Any:
        """Read model weights."""

        manifest = self.manifest()
        import torch

        try:
            return torch.load(self.path / manifest.weights_file, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(self.path / manifest.weights_file, map_location=map_location)

    def _validate_hashes(self, manifest: ArcModelManifest) -> None:
        weights_hash = self._sha256(self.path / manifest.weights_file)
        tokenizer_hash = self._sha256(self.path / self.TOKENIZER)
        if weights_hash != manifest.weights_hash:
            raise ValueError("Artifact weights hash does not match manifest.")
        if tokenizer_hash != manifest.tokenizer_hash:
            raise ValueError("Artifact tokenizer hash does not match manifest.")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _model_dtype(model: Any) -> str:
        try:
            return str(next(model.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            return "unknown"
