"""Transitional vNext model lifecycle API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        model = ArcLM(
            vocab_size=int(config.vocab_size),
            embed_dim=int(config.embed_dim),
            block_size=int(config.block_size),
            num_blocks=int(config.num_blocks),
            dropout=float(config.dropout),
        ).to(runtime.torch_device())
        model.eval()
        return cls(model=model, config=config, spec=spec, runtime=runtime, tokenizer=tokenizer)

    @classmethod
    def load(cls, source: str | Path, *, runtime: Runtime | None = None) -> "Model":
        """Load an ArcLM `.arcmodel` artifact."""

        artifact = ArcModelArtifact(source)
        manifest = artifact.manifest()
        spec = ModelRegistry.get(manifest.architecture_id)
        runtime = runtime or Runtime.auto()
        config = Config(**{**artifact.read_config(), "device": runtime.device_name})
        tokenizer = cls._load_tokenizer(artifact.read_tokenizer())
        model = ArcLM(
            vocab_size=int(config.vocab_size),
            embed_dim=int(config.embed_dim),
            block_size=int(config.block_size),
            num_blocks=int(config.num_blocks),
            dropout=float(config.dropout),
        ).to(runtime.torch_device())
        model.load_state_dict(artifact.read_state_dict(map_location=runtime.torch_device()), strict=True)
        model.eval()
        return cls(model=model, config=config, spec=spec, runtime=runtime, tokenizer=tokenizer, artifact=artifact)

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

    @staticmethod
    def _load_tokenizer(payload: dict[str, Any]) -> Tokenizer | SentencePieceTokenizer:
        tokenizer_type = payload.get("tokenizer_type", "word")
        if tokenizer_type == "word":
            return Tokenizer.from_json(payload)
        if tokenizer_type == "sentencepiece":
            return SentencePieceTokenizer.from_json(payload)
        raise ValueError(f"Unsupported tokenizer type in artifact: {tokenizer_type!r}.")
