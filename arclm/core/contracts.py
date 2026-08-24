"""Typed boundaries for ArcLM Core.

These protocols define ownership boundaries without forcing every existing
legacy implementation into inheritance. Core code may depend on these contracts;
public API wrappers and backend implementations satisfy them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable


@runtime_checkable
class RuntimeContract(Protocol):
    """Runtime owns backend/device/precision selection."""

    backend: str
    device_name: str
    precision: str

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class ModelSpecContract(Protocol):
    """ModelSpec owns architecture metadata, support, requirements and mappings."""

    architecture_id: str
    display_name: str
    task: str
    capabilities: Any
    configuration: dict[str, Any]
    weight_mapping: dict[str, str]
    tokenizer_requirements: dict[str, Any]
    adapter_targets: tuple[str, ...]
    runtime_requirements: dict[str, Any]

    def supports(self, capability: str) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...


class ModelRegistryContract(Protocol):
    """Registry owns resolution from config/metadata/reference to ModelSpec."""

    @classmethod
    def resolve(cls, config: dict[str, Any] | str) -> ModelSpecContract: ...
    @classmethod
    def supports(cls, model: dict[str, Any] | str, capabilities: Iterable[str] | None = None) -> dict[str, str]: ...


@runtime_checkable
class ModelContract(Protocol):
    """Model owns architecture instance, config, runtime, tokenizer and lifecycle."""

    model: Any
    config: Any
    spec: ModelSpecContract
    runtime: RuntimeContract
    tokenizer: Any

    def generate(self, prompt: str, *, max_new_tokens: int = 20, temperature: float = 0.0) -> str: ...
    def save(self, path: str | Path, **kwargs: Any) -> Any: ...
    def inspect(self) -> dict[str, Any]: ...


@runtime_checkable
class DatasetContract(Protocol):
    """Dataset owns source records, inspection and preparation into batches."""

    def prepare(self, **kwargs: Any) -> Any: ...
    def inspect(self) -> Any: ...


@runtime_checkable
class BackendContract(Protocol):
    """Backend owns implementation-specific tensor/autograd/optimizer calls."""

    name: str

    def create_optimizer(self, parameters: Iterable[Any], *, learning_rate: float, weight_decay: float = 0.0) -> Any: ...
    def zero_grad(self, optimizer: Any) -> None: ...
    def backward(self, loss: Any) -> None: ...
    def optimizer_step(self, optimizer: Any) -> None: ...
    def scheduler_step(self, scheduler: Any, metric: float | None = None) -> None: ...
    def cross_entropy_next_token_loss(self, logits: Any, targets: Any) -> Any: ...
    def scalar(self, value: Any) -> float: ...


@runtime_checkable
class TrainingEngineContract(Protocol):
    """TrainingEngine owns lifecycle, execution loop, metrics and hooks."""

    backend: BackendContract

    def fit(self, *, model: Any, dataloader: Any, strategy: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ArtifactContract(Protocol):
    """Artifact owns manifest/config/metadata/layout/integrity and weight IO."""

    path: Path

    def manifest(self) -> Any: ...
    def read_config(self) -> dict[str, Any]: ...
    def read_state_dict(self, map_location: Any = "cpu") -> Any: ...
    def inspect(self) -> dict[str, Any]: ...


class WeightReaderContract(Protocol):
    """WeightReader returns implementation tensors by ArcLM tensor name."""

    def read_state_dict(self, map_location: Any = "cpu") -> Any: ...


class WeightWriterContract(Protocol):
    """WeightWriter stores implementation tensors under an ArcLM-owned layout."""

    @classmethod
    def save(cls, path: str | Path, **kwargs: Any) -> Any: ...


class WeightMapperContract(Protocol):
    """WeightMapper owns family-specific external-to-ArcLM tensor names."""

    def map_name(self, external_name: str) -> str: ...
    def map_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class AdapterContract(Protocol):
    """Adapter owns trainable overlay parameters and compatibility metadata."""

    adapter_type: str
    target_modules: tuple[str, ...]

    def state_dict(self) -> dict[str, Any]: ...
