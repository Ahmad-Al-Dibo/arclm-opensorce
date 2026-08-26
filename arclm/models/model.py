"""Public model lifecycle API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..architectures import Architecture, architectures
from ..config import Config
from ..runtime import Runtime
from ..tokenizers import Tokenizer


@dataclass
class Model:
    """ArcLM-owned model wrapper for native causal language-model workflows."""

    model: Any
    config: Config
    architecture: Architecture
    runtime: Runtime
    tokenizer: Tokenizer | None = None
    artifact: Any | None = None
    base_model_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        architecture: str = "arclm-native",
        tokenizer: Tokenizer | None = None,
        runtime: Runtime | None = None,
        **config_values: Any,
    ) -> "Model":
        """Create a native ArcLM model."""

        architecture_definition = architectures.resolve(architecture)
        runtime = runtime or Runtime.auto(prefer=str(config_values.pop("device", "auto")))
        if tokenizer is not None:
            config_values["vocab_size"] = tokenizer.get_vocab_size()
        if not config_values.get("vocab_size"):
            raise ValueError("Model.create() requires vocab_size or a built tokenizer.")
        config = Config(**{**config_values, "device": runtime.device_name})
        architecture_definition.validate_config(config)
        model = architecture_definition.build(config, runtime)
        model.eval()
        return cls(
            model=model,
            config=config,
            architecture=architecture_definition,
            runtime=runtime,
            tokenizer=tokenizer,
            base_model_id=architecture_definition.architecture_id,
        )

    @classmethod
    def load(cls, source: str | Path, *, runtime: Runtime | None = None) -> "Model":
        """Load an ArcLM `.arcmodel` artifact."""

        from .loading import ModelLoader

        loaded = ModelLoader().load(source, runtime=runtime)
        return cls(
            model=loaded.model,
            config=loaded.config,
            architecture=loaded.architecture,
            runtime=loaded.runtime,
            tokenizer=loaded.tokenizer,
            artifact=loaded.artifact,
            base_model_id=loaded.manifest.artifact_id,
        )

    def save(
        self,
        path: str | Path,
        *,
        layout: str = "auto",
        shard_size: str | int | None = None,
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Save this model as an ArcLM-native `.arcmodel` artifact."""

        if self.tokenizer is None:
            raise ValueError("Model.save() requires an attached tokenizer.")
        from ..artifacts import ArcModelArtifact

        artifact = ArcModelArtifact.save(
            path,
            model=self.model,
            config=self.config.to_dict(),
            tokenizer=self.tokenizer,
            architecture_id=self.architecture.architecture_id,
            architecture_metadata=self.architecture.metadata().to_dict(),
            required_capabilities=["load", "generation"],
            metadata={"runtime": self.runtime.to_dict(), **dict(metadata or {})},
            layout=layout,
            shard_size=shard_size,
            overwrite=overwrite,
        )
        self.artifact = artifact
        return artifact

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ) -> str:
        """Generate text using the native model."""

        if self.tokenizer is None:
            raise ValueError("Model.generate() requires an attached tokenizer.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        return self.architecture.generate(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            runtime=self.runtime,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def inspect(self) -> dict[str, Any]:
        """Return a structured model report."""

        from ..finetuning.adapters import base_fingerprint

        return {
            "architecture_id": self.architecture.architecture_id,
            "architecture_kind": self.architecture.kind,
            "architecture_version": self.architecture.version,
            "capabilities": self.architecture.capabilities.to_dict(),
            "config": self.config.to_dict(),
            "runtime": self.runtime.to_dict(),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "artifact": str(self.artifact.path) if self.artifact is not None else None,
            "tokenizer": self.tokenizer.strategy if self.tokenizer is not None else None,
            "base_model_id": self.base_model_id or self.architecture.architecture_id,
            "base_model_fingerprint": base_fingerprint(self.model),
        }

    def get_tensor(self, name: str):
        """Return a model tensor by state-dict name."""

        state = self.model.state_dict()
        if name not in state:
            raise KeyError(f"Unknown tensor: {name}")
        return state[name]

    def set_tensor(self, name: str, value: Any) -> None:
        """Replace one tensor in the model state dict."""

        state = self.model.state_dict()
        if name not in state:
            raise KeyError(f"Unknown tensor: {name}")
        if tuple(state[name].shape) != tuple(value.shape):
            raise ValueError(f"Tensor shape mismatch for {name}: expected {tuple(state[name].shape)}, got {tuple(value.shape)}.")
        state[name].copy_(value)

    def attach_lora(
        self,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        target_modules: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Attach a native LoRA adapter to this model."""

        from ..finetuning.adapters import apply_lora

        targets = tuple(target_modules or self.architecture.adapter_targets)
        return apply_lora(self.model, rank=rank, alpha=alpha, target_modules=targets)

    def save_adapter(self, path: str | Path, *, overwrite: bool = False) -> Any:
        """Save attached native LoRA tensors as `.arcadapter`."""

        from ..finetuning.adapters import save_adapter

        return save_adapter(
            path,
            model=self.model,
            architecture_id=self.architecture.architecture_id,
            base_model_id=self.base_model_id or self.architecture.architecture_id,
            metadata={"runtime": self.runtime.to_dict()},
            overwrite=overwrite,
        )

    def load_adapter(self, path: str | Path) -> Any:
        """Attach and load a native `.arcadapter`."""

        from ..finetuning.adapters import load_adapter

        return load_adapter(
            path,
            model=self.model,
            architecture_id=self.architecture.architecture_id,
            base_model_id=self.base_model_id or self.architecture.architecture_id,
        )

    def finetune(self, *, data: Any, method: str = "lora", **kwargs: Any) -> dict[str, Any]:
        """Fine-tune this model through the public Trainer."""

        if method != "lora":
            raise ValueError("Model.finetune() currently supports method='lora'.")
        if not any(".lora_" in name for name, _ in self.model.named_parameters()):
            self.attach_lora()
        from ..training import Trainer

        trainer = Trainer(model=self, dataset=data, **kwargs)
        return trainer.train(mode="adapter")

    def summary(self) -> None:
        """Print a model summary."""

        from ..utils import print_model_summary

        print_model_summary(self.model, self.architecture, self.config, self.runtime)