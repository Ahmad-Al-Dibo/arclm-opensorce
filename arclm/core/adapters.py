"""Native ArcLM adapter support."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from .._version import __version__
from ..exceptions import ArtifactIntegrityError, ModelCompatibilityError


@dataclass(frozen=True)
class LoRAConfig:
    """ArcLM-owned LoRA configuration."""

    rank: int = 4
    alpha: float = 8.0
    target_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterManifest:
    """Versioned `.arcadapter` manifest."""

    format: str
    format_version: str
    schema_version: str
    adapter_id: str
    adapter_type: str
    base_model_id: str
    base_model_fingerprint: str
    architecture_id: str
    target_modules: list[str]
    rank: int
    alpha: float
    adapter_tensors: dict[str, dict[str, Any]]
    tensor_file: str
    tensor_hash: str
    created_at: str
    created_with: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterManifest":
        return cls(**data)


class LoRALinear:
    """Factory namespace for native LoRA linear wrappers."""


def apply_lora(model: Any, *, rank: int = 4, alpha: float = 8.0, target_modules: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Attach LoRA modules to selected linear layers and return attached names."""

    import math
    import torch
    import torch.nn as nn

    class _LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad = False
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
            self.scale = float(alpha) / float(rank)
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, x: Any) -> Any:
            return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    requested = tuple(target_modules)
    attached: list[str] = []
    for parameter in model.parameters():
        parameter.requires_grad = False

    def should_wrap(name: str, module: Any) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if hasattr(module, "lora_a") or module.__class__.__name__ == "_LoRALinear":
            return False
        if not requested:
            return True
        return name in requested or any(name.endswith(target) for target in requested)

    for name, module in list(model.named_modules()):
        if not name or not should_wrap(name, module):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, _LoRALinear(module))
        attached.append(name)

    if not attached:
        raise ValueError("No matching linear modules were found for LoRA attachment.")
    model._arclm_lora = {"rank": rank, "alpha": alpha, "target_modules": tuple(attached)}
    return tuple(attached)


def base_fingerprint(model: Any) -> str:
    """Hash base model tensors while excluding adapter tensors."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if ".lora_" in name:
            continue
        normalized_name = name.replace(".base.", ".")
        digest.update(normalized_name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapter_state_dict(model: Any) -> dict[str, Any]:
    """Return only adapter tensors."""

    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if ".lora_" in name}


def save_adapter(
    path: str | Path,
    *,
    model: Any,
    architecture_id: str,
    base_model_id: str,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> AdapterManifest:
    """Save attached LoRA tensors as an ArcLM-native `.arcadapter`."""

    target = Path(path)
    if target.suffix != ".arcadapter":
        target = target.with_suffix(".arcadapter")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Adapter already exists: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.mkdir(parents=True)

    state = adapter_state_dict(model)
    if not state:
        raise ValueError("No LoRA adapter tensors are attached to this model.")

    tensor_file = "adapter.safetensors"
    save_file(state, str(target / tensor_file))
    tensor_hash = _sha256(target / tensor_file)
    lora = getattr(model, "_arclm_lora", {})
    manifest = AdapterManifest(
        format="arcadapter",
        format_version="1",
        schema_version="1",
        adapter_id=f"arcadapter-{uuid.uuid4()}",
        adapter_type="lora",
        base_model_id=base_model_id,
        base_model_fingerprint=base_fingerprint(model),
        architecture_id=architecture_id,
        target_modules=list(lora.get("target_modules", ())),
        rank=int(lora.get("rank", 0)),
        alpha=float(lora.get("alpha", 0.0)),
        adapter_tensors={name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")} for name, tensor in state.items()},
        tensor_file=tensor_file,
        tensor_hash=tensor_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        created_with=f"arclm {__version__}",
        metadata=dict(metadata or {}),
    )
    (target / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_adapter(path: str | Path, *, model: Any, architecture_id: str, base_model_id: str) -> AdapterManifest:
    """Load a `.arcadapter`, attach LoRA modules, and verify compatibility."""

    root = Path(path)
    manifest = AdapterManifest.from_dict(json.loads((root / "manifest.json").read_text(encoding="utf-8")))
    if manifest.format != "arcadapter" or manifest.adapter_type != "lora":
        raise ValueError(f"Unsupported adapter format: {manifest.format}/{manifest.adapter_type}")
    if manifest.architecture_id != architecture_id:
        raise ModelCompatibilityError("Adapter architecture does not match the base model.")
    if manifest.base_model_id != base_model_id:
        raise ModelCompatibilityError("Adapter base model id does not match the target model.")
    if base_fingerprint(model) != manifest.base_model_fingerprint:
        raise ModelCompatibilityError("Adapter base model fingerprint is incompatible.")

    tensor_path = root / manifest.tensor_file
    if _sha256(tensor_path) != manifest.tensor_hash:
        raise ArtifactIntegrityError("Adapter tensor hash does not match manifest.")
    targets = tuple(manifest.target_modules)
    apply_lora(model, rank=manifest.rank, alpha=manifest.alpha, target_modules=targets)
    tensors = load_file(str(tensor_path), device="cpu")
    _validate_adapter_tensors(manifest, tensors)
    model.load_state_dict(tensors, strict=False)
    return manifest


def _validate_adapter_tensors(manifest: AdapterManifest, tensors: dict[str, Any]) -> None:
    expected = manifest.adapter_tensors
    if set(tensors) != set(expected):
        raise ArtifactIntegrityError("Adapter tensor names do not match manifest.")
    for name, tensor in tensors.items():
        if list(tensor.shape) != list(expected[name]["shape"]):
            raise ArtifactIntegrityError(f"Adapter tensor shape mismatch for {name}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
