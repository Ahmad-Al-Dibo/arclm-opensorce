"""ArcLM training checkpoint support."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import __version__


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata stored next to checkpoint state."""

    format: str
    format_version: str
    checkpoint_id: str
    created_at: str
    created_with: str
    epoch: int
    step: int
    optimizer_step: int
    strategy: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Checkpoint payload returned by the checkpoint manager."""

    path: Path
    metadata: CheckpointMetadata
    state: dict[str, Any]


class CheckpointManager:
    """Save and restore training state in an ArcLM-owned directory layout."""

    METADATA = "metadata.json"
    STATE = "state.pt"
    LATEST = "latest.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(
        self,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        epoch: int,
        step: int,
        optimizer_step: int,
        strategy: str,
        metrics: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> Path:
        import torch

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"checkpoint-{step:06d}.arcckpt"
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Checkpoint already exists: {path}")
            shutil.rmtree(path)
        path.mkdir()
        metadata = CheckpointMetadata(
            format="arcckpt",
            format_version="1",
            checkpoint_id=f"arcckpt-{uuid.uuid4()}",
            created_at=datetime.now(timezone.utc).isoformat(),
            created_with=f"arclm {__version__}",
            epoch=epoch,
            step=step,
            optimizer_step=optimizer_step,
            strategy=strategy,
            metrics=dict(metrics or {}),
        )
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "metadata": metadata.to_dict(),
        }
        torch.save(state, path / self.STATE)
        (path / self.METADATA).write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (self.root / self.LATEST).write_text(json.dumps({"checkpoint": path.name}, indent=2), encoding="utf-8")
        return path

    def load(self, checkpoint: str | Path | None = None, *, map_location: Any = "cpu") -> LoadedCheckpoint:
        import torch

        path = self.resolve(checkpoint)
        metadata = CheckpointMetadata.from_dict(json.loads((path / self.METADATA).read_text(encoding="utf-8")))
        state = torch.load(path / self.STATE, map_location=map_location)
        return LoadedCheckpoint(path=path, metadata=metadata, state=state)

    def resolve(self, checkpoint: str | Path | None = None) -> Path:
        if checkpoint is not None:
            path = Path(checkpoint)
            return path if path.is_absolute() else self.root / path
        latest = self.root / self.LATEST
        if not latest.exists():
            raise FileNotFoundError(f"No latest checkpoint found in {self.root}")
        payload = json.loads(latest.read_text(encoding="utf-8"))
        return self.root / payload["checkpoint"]


__all__ = ["CheckpointManager", "CheckpointMetadata", "LoadedCheckpoint"]
