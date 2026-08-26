"""ArcLM-owned architecture contracts and metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


class ArchitectureKind:
    """Architecture origin labels."""

    NATIVE = "native"
    COMPATIBILITY = "compatibility"
    CUSTOM = "custom"


class CapabilitySupport:
    """Capability support labels exposed to users and docs."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ArchitectureCapabilities:
    """What ArcLM can currently do with one architecture."""

    load: str = CapabilitySupport.UNSUPPORTED
    inference: str = CapabilitySupport.UNSUPPORTED
    generation: str = CapabilitySupport.UNSUPPORTED
    training: str = CapabilitySupport.UNSUPPORTED
    training_from_scratch: str = CapabilitySupport.UNSUPPORTED
    fine_tuning: str = CapabilitySupport.UNSUPPORTED
    configuration_editing: str = CapabilitySupport.UNSUPPORTED
    layer_replacement: str = CapabilitySupport.UNSUPPORTED
    custom_forward: str = CapabilitySupport.UNSUPPORTED
    custom_attention: str = CapabilitySupport.UNSUPPORTED
    export: str = CapabilitySupport.UNSUPPORTED

    def supports(self, capability: str) -> str:
        return self.to_dict().get(capability, CapabilitySupport.UNSUPPORTED)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureMetadata:
    """Documentation-ready architecture metadata."""

    architecture_id: str
    name: str
    version: str
    kind: str
    task: str
    capabilities: dict[str, str]
    config_schema: dict[str, Any]
    component_slots: list[str]
    adapter_targets: list[str]
    runtime_requirements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IdentityWeightMapper:
    """Native ArcLM tensor names already use ArcLM-owned names."""

    def map_name(self, external_name: str) -> str:
        return external_name

    def map_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return dict(state_dict)


class Architecture:
    """Base class for ArcLM model-family architecture definitions."""

    architecture_id: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str] = "1"
    kind: ClassVar[str] = ArchitectureKind.CUSTOM
    task: ClassVar[str] = "causal-lm"
    capabilities: ClassVar[ArchitectureCapabilities] = ArchitectureCapabilities()
    config_schema: ClassVar[dict[str, Any]] = {}
    component_slots: ClassVar[tuple[str, ...]] = ()
    adapter_targets: ClassVar[tuple[str, ...]] = ()
    runtime_requirements: ClassVar[dict[str, Any]] = {}
    weight_mapper: ClassVar[Any] = IdentityWeightMapper()

    def validate_config(self, config: Any) -> None:
        """Validate a config object for this architecture."""

        required = self.config_schema.get("required", ())
        missing = [name for name in required if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f"{self.architecture_id} config is missing required field(s): {', '.join(missing)}")

    def build(self, config: Any, runtime: Any) -> Any:
        """Build a runtime model instance."""

        raise NotImplementedError

    def supports_component(self, component_name: str) -> bool:
        """Return whether this architecture advertises a replaceable component slot."""

        return component_name in self.component_slots

    def generate(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: Any,
        runtime: Any,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ) -> str:
        """Default next-token generation for causal language models."""

        import torch

        token_ids = tokenizer.encode(prompt.lower())
        if not token_ids:
            token_ids = [tokenizer.get_unknown_index()]
        tokens = torch.tensor([token_ids], device=runtime.torch_device(), dtype=torch.long)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                window = tokens[:, -int(config.block_size):]
                logits = model(window)[:, -1, :]
                if temperature <= 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probabilities, 1)
                tokens = torch.cat([tokens, next_token], dim=1)
        return tokenizer.decode([int(item) for item in tokens[0].detach().cpu().tolist()])

    def metadata(self) -> ArchitectureMetadata:
        """Return documentation and artifact-ready architecture metadata."""

        return ArchitectureMetadata(
            architecture_id=self.architecture_id,
            name=self.name,
            version=self.version,
            kind=self.kind,
            task=self.task,
            capabilities=self.capabilities.to_dict(),
            config_schema=dict(self.config_schema),
            component_slots=list(self.component_slots),
            adapter_targets=list(self.adapter_targets),
            runtime_requirements=dict(self.runtime_requirements),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.metadata().to_dict()


__all__ = [
    "Architecture",
    "ArchitectureCapabilities",
    "ArchitectureKind",
    "ArchitectureMetadata",
    "CapabilitySupport",
    "IdentityWeightMapper",
]
