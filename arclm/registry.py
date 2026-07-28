"""Simple extension registries for ArcLM components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    object: Any
    kind: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["object"] = repr(self.object)
        return data


class Registry:
    """Named extension registry with explicit override semantics."""

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: Dict[str, RegistryEntry] = {}

    def register(self, name: str, obj: Any, *, override: bool = False, **metadata: Any) -> RegistryEntry:
        normalized = name.strip()
        if not normalized:
            raise ConfigurationError("Registry name cannot be empty.")
        if normalized in self._entries and not override:
            raise ConfigurationError(f"{self.kind} registry entry {normalized!r} already exists.")
        entry = RegistryEntry(normalized, obj, self.kind, metadata)
        self._entries[normalized] = entry
        return entry

    def get(self, name: str) -> Any:
        try:
            return self._entries[name].object
        except KeyError as exc:
            raise ConfigurationError(f"Unknown {self.kind} registry entry: {name}") from exc

    def list(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in sorted(self._entries.values(), key=lambda item: item.name)]


dataset_loaders = Registry("dataset_loader")
transforms = Registry("transform")
model_adapters = Registry("model_adapter")
metrics = Registry("metric")
plugins = Registry("plugin")


def register_dataset_loader(name: str, loader: Callable[..., Any], *, override: bool = False, **metadata: Any) -> RegistryEntry:
    return dataset_loaders.register(name, loader, override=override, **metadata)


def register_transform(name: str, transform: Callable[..., Any], *, override: bool = False, **metadata: Any) -> RegistryEntry:
    return transforms.register(name, transform, override=override, **metadata)


def register_metric(name: str, metric: Callable[..., Any], *, override: bool = False, **metadata: Any) -> RegistryEntry:
    return metrics.register(name, metric, override=override, **metadata)


def list_plugins() -> list[dict[str, Any]]:
    return plugins.list()


__all__ = [
    "Registry",
    "RegistryEntry",
    "dataset_loaders",
    "list_plugins",
    "metrics",
    "model_adapters",
    "plugins",
    "register_dataset_loader",
    "register_metric",
    "register_transform",
    "transforms",
]
