"""Capability-aware model registry for the vNext slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ..core.architectures import ArcLMNativeArchitecture


class SupportLevel:
    """Capability support labels."""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    EXPERIMENTAL = "EXPERIMENTAL"
    PLANNED = "PLANNED"
    UNTESTED = "UNTESTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class CapabilitySet:
    """Capability-specific model support."""

    inference: str = SupportLevel.UNSUPPORTED
    pretraining: str = SupportLevel.UNSUPPORTED
    full_finetune: str = SupportLevel.UNSUPPORTED
    lora: str = SupportLevel.UNSUPPORTED
    save: str = SupportLevel.UNSUPPORTED
    load: str = SupportLevel.UNSUPPORTED
    qlora: str = SupportLevel.UNSUPPORTED
    cpu: str = SupportLevel.UNTESTED
    cuda: str = SupportLevel.UNTESTED

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe capability mapping."""

        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    """ArcLM-owned description of an architecture and its capabilities."""

    architecture_id: str
    display_name: str
    tasks: frozenset[str]
    capabilities: CapabilitySet
    architecture: Any | None = None
    task: str = "causal-lm"
    configuration: dict[str, Any] = field(default_factory=dict)
    weight_mapping: dict[str, str] = field(default_factory=dict)
    weight_mapper: Any | None = None
    tokenizer_requirements: dict[str, Any] = field(default_factory=dict)
    adapter_targets: tuple[str, ...] = ()
    runtime_requirements: dict[str, Any] = field(default_factory=dict)
    config_keys: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, capability: str) -> str:
        """Return the support level for one capability."""

        return self.capabilities.to_dict().get(capability, SupportLevel.UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe spec report."""

        return {
            "architecture_id": self.architecture_id,
            "display_name": self.display_name,
            "task": self.task,
            "tasks": sorted(self.tasks),
            "capabilities": self.capabilities.to_dict(),
            "configuration": dict(self.configuration),
            "weight_mapping": dict(self.weight_mapping),
            "tokenizer_requirements": dict(self.tokenizer_requirements),
            "adapter_targets": list(self.adapter_targets),
            "runtime_requirements": dict(self.runtime_requirements),
            "config_keys": sorted(self.config_keys),
            "metadata": dict(self.metadata),
        }


class IdentityWeightMapper:
    """Native ArcLM tensor names are already ArcLM-owned."""

    def map_name(self, external_name: str) -> str:
        return external_name

    def map_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return dict(state_dict)


class ModelRegistry:
    """Authoritative vNext model registry."""

    _specs: dict[str, ModelSpec] = {}

    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        """Register or replace a model spec."""

        cls._specs[spec.architecture_id] = spec

    @classmethod
    def get(cls, architecture_id: str) -> ModelSpec:
        """Return a registered spec by ID."""

        try:
            return cls._specs[architecture_id]
        except KeyError as exc:
            raise ValueError(f"Unsupported architecture: {architecture_id!r}.") from exc

    @classmethod
    def resolve(cls, config: dict[str, Any] | str) -> ModelSpec:
        """Resolve a config or architecture name to a registered spec."""

        if isinstance(config, str):
            key = config
        else:
            key = (
                config.get("architecture_id")
                or config.get("architecture")
                or config.get("model_type")
                or "arclm-native-causal-lm"
            )
        aliases = {
            "arclm": "arclm-native-causal-lm",
            "arclm-native": "arclm-native-causal-lm",
            "native": "arclm-native-causal-lm",
            "transformer": "arclm-native-causal-lm",
            "arclm_native_causal_lm": "arclm-native-causal-lm",
        }
        return cls.get(aliases.get(str(key), str(key)))

    @classmethod
    def supported(cls) -> list[dict[str, Any]]:
        """Return all registered specs as support reports."""

        return [spec.to_dict() for spec in cls._specs.values()]

    @classmethod
    def supports(cls, model: dict[str, Any] | str, capabilities: Iterable[str] | None = None) -> dict[str, str]:
        """Return capability support for a model/config."""

        spec = cls.resolve(model)
        report = spec.capabilities.to_dict()
        if capabilities is None:
            return report
        return {capability: report.get(capability, SupportLevel.UNSUPPORTED) for capability in capabilities}


ModelRegistry.register(
    ModelSpec(
        architecture_id="arclm-native-causal-lm",
        display_name="ArcLM Native Causal LM",
        tasks=frozenset({"causal-lm", "generation"}),
        capabilities=CapabilitySet(
            inference=SupportLevel.SUPPORTED,
            pretraining=SupportLevel.SUPPORTED,
            full_finetune=SupportLevel.EXPERIMENTAL,
            lora=SupportLevel.SUPPORTED,
            save=SupportLevel.SUPPORTED,
            load=SupportLevel.SUPPORTED,
            qlora=SupportLevel.UNSUPPORTED,
            cpu=SupportLevel.SUPPORTED,
            cuda=SupportLevel.UNTESTED,
        ),
        architecture=ArcLMNativeArchitecture(),
        task="causal-lm",
        configuration={
            "required": ["vocab_size"],
            "defaults": {"embed_dim": 64, "block_size": 8, "num_blocks": 2, "dropout": 0.0},
        },
        weight_mapping={"type": "identity"},
        weight_mapper=IdentityWeightMapper(),
        tokenizer_requirements={"kind": "arclm-tokenizer", "required": True},
        adapter_targets=("blocks.0.attn.query", "blocks.0.attn.value", "head"),
        runtime_requirements={"backend": "torch", "devices": ["cpu", "cuda"]},
        config_keys=frozenset({"vocab_size", "embed_dim", "block_size", "num_blocks", "dropout"}),
        metadata={"pretrained_family": "ArcLM native causal LM artifacts"},
    )
)
