"""Internal ArcLM model loading pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..architectures import Architecture, ArchitectureRegistry, CapabilitySupport, architectures
from ..artifacts import ArcModelArtifact, ArcModelManifest
from ..config import Config
from ..runtime import Runtime
from ..tokenizers import Tokenizer


@dataclass(frozen=True)
class LoadedModelComponents:
    """Resolved pieces needed to construct a public Model."""

    model: Any
    config: Config
    architecture: Architecture
    runtime: Runtime
    tokenizer: Tokenizer
    artifact: ArcModelArtifact
    manifest: ArcModelManifest


class ModelLoader:
    """Resolve ArcLM-owned model artifacts into runtime components."""

    def __init__(self, *, registry: ArchitectureRegistry = architectures):
        self.registry = registry

    def load(self, source: str | Path, *, runtime: Runtime | None = None) -> LoadedModelComponents:
        runtime = runtime or Runtime.auto()
        artifact = ArcModelArtifact(source)
        manifest = artifact.manifest()
        architecture = self.registry.get(manifest.architecture_id)
        self._validate_required_capabilities(architecture, manifest)
        config = self._read_config(artifact, runtime)
        tokenizer = Tokenizer.from_json(artifact.read_tokenizer())
        architecture.validate_config(config)
        model = architecture.build(config, runtime)
        state = artifact.read_state_dict(map_location=runtime.device_name)
        mapper = getattr(architecture, "weight_mapper", None)
        if mapper is not None:
            state = mapper.map_state_dict(state)
        model.load_state_dict(state, strict=True)
        model.eval()
        return LoadedModelComponents(
            model=model,
            config=config,
            architecture=architecture,
            runtime=runtime,
            tokenizer=tokenizer,
            artifact=artifact,
            manifest=manifest,
        )

    @staticmethod
    def _read_config(artifact: ArcModelArtifact, runtime: Runtime) -> Config:
        payload = dict(artifact.read_config())
        payload["device"] = runtime.device_name
        return Config(**payload)

    @staticmethod
    def _validate_required_capabilities(architecture: Architecture, manifest: ArcModelManifest) -> None:
        unsupported = [
            capability
            for capability in manifest.required_capabilities
            if architecture.capabilities.supports(capability) == CapabilitySupport.UNSUPPORTED
        ]
        if unsupported:
            raise ValueError(
                f"Artifact requires unsupported architecture capability/capabilities for "
                f"{architecture.architecture_id!r}: {', '.join(unsupported)}"
            )


__all__ = ["LoadedModelComponents", "ModelLoader"]
