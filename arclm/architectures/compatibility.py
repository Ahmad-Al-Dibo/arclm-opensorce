"""Compatibility architectures for external model ecosystems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Architecture, ArchitectureCapabilities, ArchitectureKind, CapabilitySupport


class GemmaTransformersCausalLM(Architecture):
    """First external compatibility vertical slice: Gemma via Transformers."""

    architecture_id = "gemma-transformers-causal-lm"
    name = "Gemma Transformers Compatibility Causal LM"
    version = "1"
    kind = ArchitectureKind.COMPATIBILITY
    task = "causal-lm"
    capabilities = ArchitectureCapabilities(
        load=CapabilitySupport.SUPPORTED,
        inference=CapabilitySupport.EXPERIMENTAL,
        generation=CapabilitySupport.EXPERIMENTAL,
        training=CapabilitySupport.EXPERIMENTAL,
        fine_tuning=CapabilitySupport.EXPERIMENTAL,
        export=CapabilitySupport.EXPERIMENTAL,
    )
    config_schema = {
        "required": ["vocab_size", "external_model_id"],
        "fields": ["vocab_size", "block_size", "model_family", "external_model_id", "compatibility_backend"],
    }
    component_slots = ("model", "token_embedding", "layers", "attention", "mlp", "lm_head")
    adapter_targets = ("q_proj", "v_proj")
    runtime_requirements = {"backend": "torch", "compatibility_backend": "transformers", "devices": ["cpu", "cuda"]}

    def can_load_source(self, source: str | Path) -> bool:
        reference = str(source)
        normalized = reference.lower().strip()
        if normalized == "arclm://compat/gemma-tiny":
            return True
        if normalized.startswith("hf://"):
            normalized = normalized[5:]
        return "gemma" in normalized and not Path(reference).suffix == ".arcmodel"

    def load_external(self, source: str | Path, *, runtime: Any, **options: Any) -> dict[str, Any]:
        reference = str(source)
        if reference == "arclm://compat/gemma-tiny" or options.get("fixture"):
            return self._load_tiny_fixture(runtime=runtime)

        model_id = reference[5:] if reference.lower().startswith("hf://") else reference
        special_tokens = {
            "bos": "<bos>",
            "eos": "<eos>",
            "pad": "<pad>",
            "unk": "<unk>",
            "user": "<start_of_turn>user",
            "assistant": "<start_of_turn>model",
            "system": "<start_of_turn>system",
            "separator": "<end_of_turn>",
        }
        from ..compat.transformers import TransformersBackend

        loaded = TransformersBackend().load_causal_lm(
            model_id,
            runtime=runtime,
            family="gemma",
            revision=options.get("revision"),
            trust_remote_code=bool(options.get("trust_remote_code", False)),
            torch_dtype=options.get("torch_dtype"),
            device_map=options.get("device_map"),
            special_tokens=special_tokens,
            chat_template=options.get("chat_template"),
        )
        from ..config import Config

        return {
            "model": loaded.model,
            "tokenizer": loaded.tokenizer,
            "config": Config(**{key: value for key, value in loaded.config.items() if key in Config.__dataclass_fields__}),
            "base_model_id": model_id,
        }

    def build(self, config: Any, runtime: Any) -> Any:
        self.validate_config(config)
        if getattr(config, "external_model_id", None) == "arclm://compat/gemma-tiny":
            return self._tiny_model(
                vocab_size=int(config.vocab_size),
                embed_dim=int(getattr(config, "embed_dim", 16)),
                block_size=int(getattr(config, "block_size", 16)),
            ).to(runtime.torch_device())
        return self.load_external(
            getattr(config, "external_model_id"),
            runtime=runtime,
            revision=getattr(config, "revision", None),
            trust_remote_code=getattr(config, "trust_remote_code", False),
            torch_dtype=getattr(config, "torch_dtype", None),
        )["model"]

    def generate(self, *, model: Any, tokenizer: Any, config: Any, runtime: Any, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0) -> str:
        if hasattr(model, "generate"):
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            import torch

            inputs = torch.tensor([token_ids], device=runtime.torch_device(), dtype=torch.long)
            generated = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-6))
            return tokenizer.decode([int(item) for item in generated[0].detach().cpu().tolist()])
        return super().generate(model=model, tokenizer=tokenizer, config=config, runtime=runtime, prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def _load_tiny_fixture(self, *, runtime: Any) -> dict[str, Any]:
        from ..config import Config
        from ..tokenizers import ChatTemplate, Tokenizer

        tokenizer = Tokenizer(
            strategy="word",
            max_vocab=128,
            chat_template=ChatTemplate.default(),
            special_tokens={
                "user": "<start_of_turn>user",
                "assistant": "<start_of_turn>model",
                "separator": "<end_of_turn>",
            },
        ).build(
            "alpha beta gamma delta epsilon zeta eta theta hello hi there ping pong "
            "<start_of_turn>user <start_of_turn>model <end_of_turn>"
        )
        config = Config(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=16,
            block_size=16,
            num_blocks=1,
            model_family="gemma",
            external_model_id="arclm://compat/gemma-tiny",
            compatibility_backend="fixture",
            device=runtime.device_name,
        )
        return {
            "model": self._tiny_model(vocab_size=config.vocab_size, embed_dim=config.embed_dim, block_size=config.block_size).to(runtime.torch_device()),
            "tokenizer": tokenizer,
            "config": config,
            "base_model_id": "arclm://compat/gemma-tiny",
        }

    @staticmethod
    def _tiny_model(*, vocab_size: int, embed_dim: int, block_size: int) -> Any:
        import torch
        import torch.nn as nn

        class _TinyGemmaAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.o_proj = nn.Linear(embed_dim, embed_dim)

            def forward(self, hidden):
                return self.o_proj(torch.tanh(self.q_proj(hidden)) + self.v_proj(hidden))

        class _TinyGemmaLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _TinyGemmaAttention()
                self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, hidden):
                hidden = self.norm(hidden + self.self_attn(hidden))
                return hidden + self.mlp(hidden)

        class _TinyGemmaForCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
                self.position_embedding = nn.Embedding(block_size, embed_dim)
                self.layers = nn.ModuleList([_TinyGemmaLayer()])
                self.lm_head = nn.Linear(embed_dim, vocab_size)

            def forward(self, input_ids):
                positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                hidden = self.embed_tokens(input_ids) + self.position_embedding(positions.clamp(max=block_size - 1))
                for layer in self.layers:
                    hidden = layer(hidden)
                return self.lm_head(hidden)

        return _TinyGemmaForCausalLM()


__all__ = ["GemmaTransformersCausalLM"]
