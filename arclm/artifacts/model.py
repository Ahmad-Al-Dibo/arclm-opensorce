"""Experimental `.arcmodel` artifact support."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import uuid
import zipfile
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
    layout: str = "directory"
    weight_files: list[str] = field(default_factory=list)
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
    WEIGHTS_DIR = "weights"
    SHARD_INDEX = "weights/index.json"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def normalize_path(path: str | Path, *, layout: str = "auto") -> Path:
        """Normalize a user path for the selected artifact layout."""

        resolved = Path(path)
        if layout in {"auto", "single"} and resolved.suffix != ".arcmodel":
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
        layout: str = "auto",
        shard_size: str | int | None = None,
        overwrite: bool = False,
    ) -> "ArcModelArtifact":
        """Write a native ArcLM model artifact."""

        selected_layout = cls._select_layout(model, path, layout)
        artifact_path = cls.normalize_path(path, layout=selected_layout)
        artifact = cls(artifact_path)
        if artifact_path.exists():
            if not overwrite:
                raise FileExistsError(f"Artifact already exists: {artifact_path}")
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
            else:
                artifact_path.unlink()

        if selected_layout == "single":
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_root = Path(tmpdir) / "artifact"
                cls._write_directory_payload(
                    temp_root,
                    model=model,
                    config=config,
                    tokenizer=tokenizer,
                    architecture_id=architecture_id,
                    metadata=metadata,
                    layout=selected_layout,
                    shard_size=shard_size,
                )
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(artifact_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for item in temp_root.rglob("*"):
                        if item.is_file():
                            archive.write(item, item.relative_to(temp_root).as_posix())
            return artifact

        cls._write_directory_payload(
            artifact_path,
            model=model,
            config=config,
            tokenizer=tokenizer,
            architecture_id=architecture_id,
            metadata=metadata,
            layout=selected_layout,
            shard_size=shard_size,
        )
        return artifact

    @classmethod
    def _write_directory_payload(
        cls,
        root: Path,
        *,
        model: Any,
        config: dict[str, Any],
        tokenizer: Any,
        architecture_id: str,
        metadata: dict[str, Any] | None,
        layout: str,
        shard_size: str | int | None,
    ) -> None:
        root.mkdir(parents=True)

        config_path = root / cls.CONFIG
        tokenizer_path = root / cls.TOKENIZER

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tokenizer_payload = tokenizer.to_json() if hasattr(tokenizer, "to_json") else {}
        tokenizer_path.write_text(json.dumps(tokenizer_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

        import torch

        state_dict = model.state_dict()
        if layout == "sharded":
            weights_root = root / cls.WEIGHTS_DIR
            weights_root.mkdir()
            shard_name = "shard-00001.pt"
            weights_path = weights_root / shard_name
            torch.save(state_dict, weights_path)
            index = {
                "format": "arcweights-index",
                "schema_version": "1",
                "shard_size": shard_size,
                "weight_map": {name: shard_name for name in state_dict},
                "shards": [shard_name],
            }
            (root / cls.SHARD_INDEX).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
            weights_file = cls.SHARD_INDEX
            weight_files = [f"{cls.WEIGHTS_DIR}/{shard_name}"]
            weights_hash = cls._sha256(weights_path)
        else:
            weights_path = root / cls.WEIGHTS
            torch.save(state_dict, weights_path)
            weights_file = cls.WEIGHTS
            weight_files = [cls.WEIGHTS]
            weights_hash = cls._sha256(weights_path)

        manifest = ArcModelManifest(
            artifact_id=f"arcmodel-{uuid.uuid4()}",
            format="arcmodel",
            format_version="1",
            schema_version="1",
            minimum_reader_version=__version__,
            architecture_id=architecture_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            dtype=cls._model_dtype(model),
            weights_file=weights_file,
            weights_hash=weights_hash,
            tokenizer_hash=cls._sha256(tokenizer_path),
            layout=layout,
            weight_files=weight_files,
            metadata=dict(metadata or {}),
        )
        (root / cls.MANIFEST).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def manifest(self) -> ArcModelManifest:
        """Load and validate the artifact manifest."""

        data = self._read_json(self.MANIFEST)
        if data is None:
            manifest_path = self.path / self.MANIFEST
            raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
        manifest = ArcModelManifest.from_dict(data)
        self._validate_hashes(manifest)
        return manifest

    def read_config(self) -> dict[str, Any]:
        """Read model configuration."""

        data = self._read_json(self.CONFIG)
        if data is None:
            raise FileNotFoundError(f"Artifact config not found: {self.CONFIG}")
        return data

    def read_tokenizer(self) -> dict[str, Any]:
        """Read tokenizer payload."""

        data = self._read_json(self.TOKENIZER)
        if data is None:
            raise FileNotFoundError(f"Artifact tokenizer not found: {self.TOKENIZER}")
        return data

    def read_state_dict(self, map_location: Any = "cpu") -> Any:
        """Read model weights."""

        manifest = self.manifest()
        import torch

        if manifest.layout == "sharded":
            index = self._read_json(manifest.weights_file)
            if not index:
                raise FileNotFoundError(f"Artifact shard index not found: {manifest.weights_file}")
            shards = index.get("shards") or []
            if len(shards) != 1:
                raise ValueError("This ArcLM reader currently supports one-shard artifacts only.")
            weights_path = f"{self.WEIGHTS_DIR}/{shards[0]}"
        else:
            weights_path = manifest.weights_file

        with self._open_binary(weights_path) as handle:
            try:
                return torch.load(handle, map_location=map_location, weights_only=True)
            except TypeError:
                handle.seek(0)
                return torch.load(handle, map_location=map_location)

    def _validate_hashes(self, manifest: ArcModelManifest) -> None:
        weight_file = manifest.weight_files[0] if manifest.weight_files else manifest.weights_file
        weights_hash = self._sha256_member(weight_file)
        tokenizer_hash = self._sha256_member(self.TOKENIZER)
        if weights_hash != manifest.weights_hash:
            raise ValueError("Artifact weights hash does not match manifest.")
        if tokenizer_hash != manifest.tokenizer_hash:
            raise ValueError("Artifact tokenizer hash does not match manifest.")

    def inspect(self) -> dict[str, Any]:
        """Return artifact metadata without exposing raw tensors."""

        manifest = self.manifest()
        return {
            "path": str(self.path),
            "format": manifest.format,
            "layout": manifest.layout,
            "architecture_id": manifest.architecture_id,
            "weights_hash": manifest.weights_hash,
            "tokenizer_hash": manifest.tokenizer_hash,
            "weight_files": list(manifest.weight_files),
            "metadata": dict(manifest.metadata),
        }

    def _read_json(self, relative_path: str) -> dict[str, Any] | None:
        if self.path.is_file():
            with zipfile.ZipFile(self.path) as archive:
                try:
                    with archive.open(relative_path) as handle:
                        return json.loads(handle.read().decode("utf-8"))
                except KeyError:
                    return None
        path = self.path / relative_path
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _open_binary(self, relative_path: str):
        if self.path.is_file():
            with zipfile.ZipFile(self.path) as archive:
                return _BytesHandle(io.BytesIO(archive.read(relative_path)))
        return (self.path / relative_path).open("rb")

    def _sha256_member(self, relative_path: str) -> str:
        if self.path.is_file():
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(relative_path) as handle:
                    return self._sha256_stream(handle)
        return self._sha256(self.path / relative_path)

    @staticmethod
    def _sha256(path: Path) -> str:
        with path.open("rb") as handle:
            return ArcModelArtifact._sha256_stream(handle)

    @staticmethod
    def _sha256_stream(handle: Any) -> str:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _model_dtype(model: Any) -> str:
        try:
            return str(next(model.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            return "unknown"

    @staticmethod
    def _select_layout(model: Any, path: str | Path, layout: str) -> str:
        normalized = str(layout or "auto").lower().strip()
        if normalized == "auto":
            return "single" if Path(path).suffix == ".arcmodel" else "directory"
        if normalized not in {"single", "directory", "sharded"}:
            raise ValueError("layout must be one of: auto, single, directory, sharded.")
        return normalized


class _BytesHandle:
    """Context manager for in-memory artifact members."""

    def __init__(self, handle: Any):
        self._handle = handle

    def __enter__(self):
        return self._handle

    def __exit__(self, exc_type, exc, traceback):
        self._handle.close()
