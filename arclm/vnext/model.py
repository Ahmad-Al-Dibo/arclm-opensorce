"""Transitional vNext model lifecycle API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.adapters import apply_lora, base_fingerprint, load_adapter, save_adapter
from ..core.architectures import ArcLMNativeArchitecture
from ..artifacts import ArcModelArtifact
from ..config import Config
from ..model import ArcLM
from ..runtime import Runtime
from ..tokenizer import SentencePieceTokenizer, Tokenizer
from .registry import ModelRegistry, ModelSpec


@dataclass
class Model:
    """ArcLM-owned model wrapper for the first vNext vertical slice."""

    model: ArcLM
    config: Config
    spec: ModelSpec
    runtime: Runtime
    tokenizer: Tokenizer | SentencePieceTokenizer | None = None
    artifact: ArcModelArtifact | None = None
    base_model_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        architecture: str = "arclm-native",
        tokenizer: Tokenizer | SentencePieceTokenizer | None = None,
        runtime: Runtime | None = None,
        **config_values: Any,
    ) -> "Model":
        """Create a native ArcLM model through vNext lifecycle semantics."""

        spec = ModelRegistry.resolve(architecture)
        runtime = runtime or Runtime.auto(prefer=str(config_values.pop("device", "auto")))
        if tokenizer is not None:
            config_values["vocab_size"] = tokenizer.get_vocab_size()
        if not config_values.get("vocab_size"):
            raise ValueError("Model.create() requires vocab_size or a built tokenizer.")
        config = Config(**{**config_values, "device": runtime.device_name})
        architecture = spec.architecture or ArcLMNativeArchitecture()
        model = architecture.build(config, runtime)
        model.eval()
        return cls(model=model, config=config, spec=spec, runtime=runtime, tokenizer=tokenizer, base_model_id=spec.architecture_id)

    @classmethod
    def load(cls, source: str | Path, *, runtime: Runtime | None = None) -> "Model":
        """Load an ArcLM `.arcmodel` artifact."""

        runtime = runtime or Runtime.auto()
        artifact = ArcModelArtifact(source)
        try:
            manifest = artifact.manifest()
            spec = ModelRegistry.get(manifest.architecture_id)
            config = Config(**{**artifact.read_config(), "device": runtime.device_name})
            tokenizer = cls._load_tokenizer(artifact.read_tokenizer())
            architecture = spec.architecture or ArcLMNativeArchitecture()
            model = architecture.build(config, runtime)
            state = artifact.read_state_dict(map_location=runtime.device_name)
            mapper = getattr(spec, "weight_mapper", None)
            if mapper is not None:
                state = mapper.map_state_dict(state)
            model.load_state_dict(state, strict=True)
            model.eval()
            return cls(
                model=model,
                config=config,
                spec=spec,
                runtime=runtime,
                tokenizer=tokenizer,
                artifact=artifact,
                base_model_id=manifest.artifact_id,
            )
        except Exception as native_error:
            path = Path(source)
            if not path.is_file():
                raise
            try:
                return cls._load_legacy_checkpoint(path, runtime=runtime)
            except Exception as legacy_error:
                raise ValueError(
                    f"Could not load {path} as a native .arcmodel or legacy ArcLM checkpoint. "
                    f"Native error: {native_error}. Legacy error: {legacy_error}."
                ) from legacy_error

    def save(
        self,
        path: str | Path,
        *,
        layout: str = "auto",
        shard_size: str | int | None = None,
        overwrite: bool = False,
    ) -> ArcModelArtifact:
        """Save this model as an ArcLM-native `.arcmodel` artifact."""

        if self.tokenizer is None:
            raise ValueError("Model.save() requires an attached tokenizer in the current vNext slice.")
        artifact = ArcModelArtifact.save(
            path,
            model=self.model,
            config=self.config.to_dict(),
            tokenizer=self.tokenizer,
            architecture_id=self.spec.architecture_id,
            metadata={"runtime": self.runtime.to_dict()},
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
            raise ValueError("Model.generate() requires an attached tokenizer in the current vNext slice.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        import torch

        ids = self.tokenizer.encode_text(prompt.lower())
        if not ids:
            ids = [self.tokenizer.get_unknown_index()]
        tokens = torch.tensor([ids], device=self.runtime.torch_device(), dtype=torch.long)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                window = tokens[:, -int(self.config.block_size):]
                logits = self.model(window)[:, -1, :]
                if temperature <= 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token], dim=1)
        return self.tokenizer.decode_string([int(item) for item in tokens[0].detach().cpu().tolist()])

    def inspect(self) -> dict[str, Any]:
        """Return a structured vNext model report."""

        return {
            "architecture_id": self.spec.architecture_id,
            "capabilities": self.spec.capabilities.to_dict(),
            "config": self.config.to_dict(),
            "runtime": self.runtime.to_dict(),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "artifact": str(self.artifact.path) if self.artifact is not None else None,
            "tokenizer": type(self.tokenizer).__name__ if self.tokenizer is not None else None,
            "base_model_id": self.base_model_id or self.spec.architecture_id,
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

        targets = tuple(target_modules or self.spec.adapter_targets)
        return apply_lora(self.model, rank=rank, alpha=alpha, target_modules=targets)

    def save_adapter(self, path: str | Path, *, overwrite: bool = False) -> Any:
        """Save attached native LoRA tensors as `.arcadapter`."""

        return save_adapter(
            path,
            model=self.model,
            architecture_id=self.spec.architecture_id,
            base_model_id=self.base_model_id or self.spec.architecture_id,
            metadata={"runtime": self.runtime.to_dict()},
            overwrite=overwrite,
        )

    def load_adapter(self, path: str | Path) -> Any:
        """Attach and load a native `.arcadapter`."""

        return load_adapter(
            path,
            model=self.model,
            architecture_id=self.spec.architecture_id,
            base_model_id=self.base_model_id or self.spec.architecture_id,
        )

    def finetune(self, *, data: Any, method: str = "lora", **kwargs: Any) -> dict[str, Any]:
        """Fine-tune this model through the public Trainer."""

        if method != "lora":
            raise ValueError("Model.finetune() currently supports method='lora'.")
        if not any(".lora_" in name for name, _ in self.model.named_parameters()):
            self.attach_lora()
        from .trainer import Trainer

        trainer = Trainer(model=self, dataset=data, **kwargs)
        return trainer.train(mode="adapter")

    @staticmethod
    def _load_tokenizer(payload: dict[str, Any]) -> Tokenizer | SentencePieceTokenizer:
        tokenizer_type = payload.get("tokenizer_type", "word")
        if tokenizer_type == "word":
            return Tokenizer.from_json(payload)
        if tokenizer_type == "sentencepiece":
            return SentencePieceTokenizer.from_json(payload)
        raise ValueError(f"Unsupported tokenizer type in artifact: {tokenizer_type!r}.")

    @classmethod
    def _load_legacy_checkpoint(cls, path: Path, *, runtime: Runtime) -> "Model":
        """Import a trusted local pre-vNext ArcLM PyTorch checkpoint."""

        import torch

        checkpoint = torch.load(path, map_location=runtime.device_name, weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Legacy checkpoint must be a dictionary with model_state_dict.")

        config_values = dict(checkpoint.get("config") or {})
        vocab_size = int(
            checkpoint.get("vocab_size")
            or config_values.get("vocab_size")
            or len(checkpoint.get("stoi") or {})
        )
        if vocab_size <= 0:
            raise ValueError("Legacy checkpoint is missing vocab_size.")

        if "num_epochs" not in config_values and "epochs" in config_values:
            config_values["num_epochs"] = config_values["epochs"]
        config_values.update(
            {
                "vocab_size": vocab_size,
                "device": runtime.device_name,
                "embed_dim": int(config_values.get("embed_dim", 64)),
                "block_size": int(config_values.get("block_size") or checkpoint.get("block_size") or 8),
                "num_blocks": int(config_values.get("num_blocks", 2)),
                "dropout": float(config_values.get("dropout", 0.0)),
            }
        )

        spec = ModelRegistry.resolve({"architecture": "arclm-native"})
        config = Config(**config_values)
        architecture = spec.architecture or ArcLMNativeArchitecture()
        model = architecture.build(config, runtime)
        state = checkpoint.get("best_model_state_dict") or checkpoint["model_state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()
        tokenizer = cls._load_legacy_tokenizer(checkpoint)
        return cls(
            model=model,
            config=config,
            spec=spec,
            runtime=runtime,
            tokenizer=tokenizer,
            artifact=None,
            base_model_id=f"legacy:{path.name}",
        )

    @staticmethod
    def _load_legacy_tokenizer(checkpoint: dict[str, Any]) -> Tokenizer | SentencePieceTokenizer:
        metadata = dict(checkpoint.get("tokenizer_metadata") or {})
        if metadata.get("tokenizer_type") == "sentencepiece":
            return SentencePieceTokenizer.from_checkpoint(metadata)

        tokenizer = Tokenizer(max_vocab=int(metadata.get("max_vocab") or checkpoint.get("vocab_size") or 50000))
        vocab = checkpoint.get("vocab")
        stoi = checkpoint.get("stoi") or {}
        itos = checkpoint.get("itos") or {}
        if vocab:
            tokenizer.vocab = list(vocab)
        elif itos:
            tokenizer.vocab = [itos[index] for index in sorted(itos)]
        elif stoi:
            tokenizer.vocab = [token for token, _ in sorted(stoi.items(), key=lambda item: item[1])]
        else:
            raise ValueError("Legacy checkpoint is missing tokenizer vocabulary.")

        tokenizer.vocab_size = len(tokenizer.vocab)
        tokenizer.stoi = {token: index for index, token in enumerate(tokenizer.vocab)}
        tokenizer.itos = {index: token for token, index in tokenizer.stoi.items()}
        return tokenizer
