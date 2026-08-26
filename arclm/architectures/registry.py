"""Cohesive architecture registry used by ArcLM model loading and creation."""

from __future__ import annotations

from typing import Any

from .base import Architecture
from .compatibility import GemmaTransformersCausalLM
from .native import ArcLMNativeCausalLM


class ArchitectureRegistry:
    """Registry/manager for built-in and user-defined architectures."""

    def __init__(self):
        self._architectures: dict[str, Architecture] = {}
        self._aliases: dict[str, str] = {}

    def register(self, architecture: Architecture | type[Architecture], *, aliases: tuple[str, ...] = (), replace: bool = False) -> Architecture:
        """Register an architecture object or class."""

        instance = architecture() if isinstance(architecture, type) else architecture
        architecture_id = self._normalize(instance.architecture_id)
        if architecture_id in self._architectures and not replace:
            raise ValueError(f"Architecture already registered: {instance.architecture_id!r}.")
        self._architectures[architecture_id] = instance
        self._aliases[architecture_id] = architecture_id
        for alias in aliases:
            self._aliases[self._normalize(alias)] = architecture_id
        return instance

    def get(self, architecture_id: str) -> Architecture:
        """Return one architecture by ID or alias."""

        key = self._aliases.get(self._normalize(architecture_id), self._normalize(architecture_id))
        try:
            return self._architectures[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported architecture: {architecture_id!r}.") from exc

    def resolve(self, reference: dict[str, Any] | str | Architecture) -> Architecture:
        """Resolve an architecture reference from config, metadata, ID, or object."""

        if isinstance(reference, Architecture):
            return reference
        if isinstance(reference, str):
            return self.get(reference)
        key = (
            reference.get("architecture_id")
            or reference.get("architecture")
            or reference.get("model_type")
            or "arclm-native-causal-lm"
        )
        return self.get(str(key))

    def available(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        """Return documentation-ready metadata for registered architectures."""

        architectures = self._architectures.values()
        if kind is not None:
            architectures = [architecture for architecture in architectures if architecture.kind == kind]
        return [architecture.to_dict() for architecture in sorted(architectures, key=lambda item: item.architecture_id)]

    def capabilities(self, architecture_id: str) -> dict[str, str]:
        """Return capability support for one architecture."""

        return self.get(architecture_id).capabilities.to_dict()

    def external_candidates(self) -> list[Architecture]:
        """Return architectures that can resolve external model references."""

        return [architecture for architecture in self._architectures.values() if hasattr(architecture, "can_load_source")]

    @staticmethod
    def _normalize(value: str) -> str:
        return str(value).lower().strip().replace("_", "-")


architectures = ArchitectureRegistry()
architectures.register(
    ArcLMNativeCausalLM,
    aliases=("arclm", "arclm-native", "native", "transformer", "arclm_native_causal_lm"),
)
architectures.register(
    GemmaTransformersCausalLM,
    aliases=("gemma", "hf-gemma", "transformers-gemma", "gemma_transformers_causal_lm"),
)

__all__ = ["ArchitectureRegistry", "architectures"]
