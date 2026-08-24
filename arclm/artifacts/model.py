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

from safetensors.torch import load, load_file, save_file

from .._version import __version__
from ..exceptions import ArtifactIntegrityError


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
    tensor_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    tensor_index_hash: str | None = None
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
    WEIGHTS = "weights/model.safetensors"
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

        state_dict = model.state_dict()
        if layout == "sharded":
            weights_root = root / cls.WEIGHTS_DIR
            weights_root.mkdir()
            index = cls._write_shards(weights_root, state_dict, shard_size=shard_size)
            (root / cls.SHARD_INDEX).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
            weights_file = cls.SHARD_INDEX
            weight_files = [f"{cls.WEIGHTS_DIR}/{item['file']}" for item in index["shards"]]
            weights_hash = cls._combined_hash(root, weight_files)
            tensor_index_hash = cls._sha256(root / cls.SHARD_INDEX)
        else:
            weights_path = root / cls.WEIGHTS
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(cls._cpu_state_dict(state_dict), str(weights_path))
            weights_file = cls.WEIGHTS
            weight_files = [cls.WEIGHTS]
            weights_hash = cls._sha256(weights_path)
            tensor_index_hash = None

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
            tensor_metadata=cls._tensor_metadata(state_dict),
            tensor_index_hash=tensor_index_hash,
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
        if manifest.layout == "sharded":
            index = self._read_json(manifest.weights_file)
            if not index:
                raise FileNotFoundError(f"Artifact shard index not found: {manifest.weights_file}")
            self._validate_index(index, manifest)
            state: dict[str, Any] = {}
            for shard in index.get("shards", []):
                relative = f"{self.WEIGHTS_DIR}/{shard['file']}"
                state.update(self._load_safetensors_member(relative, map_location=map_location))
        else:
            state = self._load_safetensors_member(manifest.weights_file, map_location=map_location)
        self._validate_state_dict(state, manifest)
        return state

    def _validate_hashes(self, manifest: ArcModelManifest) -> None:
        if manifest.layout == "sharded":
            weights_hash = self._combined_hash_members(manifest.weight_files)
        else:
            weights_hash = self._sha256_member(manifest.weights_file)
        tokenizer_hash = self._sha256_member(self.TOKENIZER)
        if weights_hash != manifest.weights_hash:
            raise ArtifactIntegrityError("Artifact weights hash does not match manifest.")
        if tokenizer_hash != manifest.tokenizer_hash:
            raise ArtifactIntegrityError("Artifact tokenizer hash does not match manifest.")
        if manifest.layout == "sharded" and manifest.tensor_index_hash:
            if self._sha256_member(manifest.weights_file) != manifest.tensor_index_hash:
                raise ArtifactIntegrityError("Artifact shard index hash does not match manifest.")

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
            "tensor_count": len(manifest.tensor_metadata),
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

    def _combined_hash_members(self, relative_paths: list[str]) -> str:
        digest = hashlib.sha256()
        for relative_path in sorted(relative_paths):
            digest.update(relative_path.encode("utf-8"))
            digest.update(self._sha256_member(relative_path).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _combined_hash(root: Path, relative_paths: list[str]) -> str:
        digest = hashlib.sha256()
        for relative_path in sorted(relative_paths):
            digest.update(relative_path.encode("utf-8"))
            digest.update(ArcModelArtifact._sha256(root / relative_path).encode("utf-8"))
        return digest.hexdigest()

    def _load_safetensors_member(self, relative_path: str, *, map_location: Any = "cpu") -> dict[str, Any]:
        device = str(map_location)
        if device.startswith("cuda"):
            device = "cuda"
        elif device != "cpu":
            device = "cpu"
        if self.path.is_file():
            with zipfile.ZipFile(self.path) as archive:
                return load(archive.read(relative_path))
        return load_file(str(self.path / relative_path), device=device)

    @classmethod
    def _write_shards(cls, weights_root: Path, state_dict: dict[str, Any], *, shard_size: str | int | None) -> dict[str, Any]:
        max_bytes = cls._parse_shard_size(shard_size)
        shards: list[dict[str, Any]] = []
        weight_map: dict[str, str] = {}
        current: dict[str, Any] = {}
        current_bytes = 0
        shard_id = 1

        def flush() -> None:
            nonlocal current, current_bytes, shard_id
            if not current:
                return
            filename = f"shard-{shard_id:05d}.safetensors"
            save_file(cls._cpu_state_dict(current), str(weights_root / filename))
            tensor_names = sorted(current)
            byte_count = sum(cls._tensor_nbytes(tensor) for tensor in current.values())
            shards.append({"file": filename, "hash": cls._sha256(weights_root / filename), "bytes": byte_count, "tensors": tensor_names})
            for tensor_name in tensor_names:
                weight_map[tensor_name] = filename
            current = {}
            current_bytes = 0
            shard_id += 1

        for name, tensor in state_dict.items():
            tensor_bytes = cls._tensor_nbytes(tensor)
            if current and current_bytes + tensor_bytes > max_bytes:
                flush()
            current[name] = tensor
            current_bytes += tensor_bytes
            if tensor_bytes >= max_bytes:
                flush()
        flush()
        return {
            "format": "arcweights-index",
            "format_version": "1",
            "schema_version": "1",
            "encoding": "safetensors",
            "max_shard_size_bytes": max_bytes,
            "weight_map": weight_map,
            "shards": shards,
        }

    @staticmethod
    def _parse_shard_size(value: str | int | None) -> int:
        if value is None:
            return 1024 * 1024 * 1024
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("shard_size must be positive.")
            return value
        text = str(value).strip().lower().replace(" ", "")
        units = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
        for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
            if text.endswith(suffix):
                number = float(text[: -len(suffix)])
                return max(1, int(number * multiplier))
        return int(text)

    @staticmethod
    def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        return {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}

    @staticmethod
    def _tensor_nbytes(tensor: Any) -> int:
        return int(tensor.numel() * tensor.element_size())

    @staticmethod
    def _tensor_metadata(state_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}
            for name, tensor in sorted(state_dict.items())
        }

    def _validate_state_dict(self, state: dict[str, Any], manifest: ArcModelManifest) -> None:
        expected = manifest.tensor_metadata or {}
        if expected and set(state) != set(expected):
            raise ArtifactIntegrityError("Artifact tensor names do not match manifest.")
        for name, meta in expected.items():
            tensor = state[name]
            if list(tensor.shape) != list(meta.get("shape", [])):
                raise ArtifactIntegrityError(f"Artifact tensor shape mismatch for {name}.")

    def _validate_index(self, index: dict[str, Any], manifest: ArcModelManifest) -> None:
        if index.get("format") != "arcweights-index" or str(index.get("schema_version")) != "1":
            raise ArtifactIntegrityError("Artifact shard index format is unsupported.")
        weight_map = index.get("weight_map") or {}
        expected_names = set(manifest.tensor_metadata)
        if expected_names and set(weight_map) != expected_names:
            raise ArtifactIntegrityError("Artifact shard index tensor names do not match manifest.")
        for shard in index.get("shards", []):
            relative = f"{self.WEIGHTS_DIR}/{shard['file']}"
            if self._sha256_member(relative) != shard.get("hash"):
                raise ArtifactIntegrityError(f"Artifact shard hash does not match index for {shard['file']}.")

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
