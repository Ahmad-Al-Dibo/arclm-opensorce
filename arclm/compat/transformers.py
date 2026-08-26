"""Optional Hugging Face Transformers compatibility backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tokenizers import ChatTemplate


class TransformersCausalLMWrapper:
    """Thin module wrapper that returns logits for ArcLM's training engine."""

    def __init__(self, model: Any):
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def __call__(self, input_ids: Any, *args: Any, **kwargs: Any) -> Any:
        return self.forward(input_ids, *args, **kwargs)

    def forward(self, input_ids: Any, *args: Any, **kwargs: Any) -> Any:
        output = self.model(input_ids=input_ids, *args, **kwargs)
        return getattr(output, "logits", output[0] if isinstance(output, (tuple, list)) else output)


@dataclass
class TransformersTokenizerAdapter:
    """Internal adapter exposing the ArcLM tokenizer protocol over HF tokenizers."""

    tokenizer: Any
    special_tokens: dict[str, str]
    chat_template: Any = None
    model_id: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    strategy: str = "transformers"

    @property
    def is_built(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        if hasattr(self.tokenizer, "__len__"):
            return int(len(self.tokenizer))
        return int(getattr(self.tokenizer, "vocab_size", 0))

    def build(self, text: str) -> "TransformersTokenizerAdapter":
        return self

    def tokenize(self, text: str) -> list[str]:
        if hasattr(self.tokenizer, "tokenize"):
            return list(self.tokenizer.tokenize(str(text)))
        return str(text).split()

    def encode(self, text: str | list[str], *, add_special_tokens: bool = False) -> list[int]:
        value = " ".join(text) if isinstance(text, list) else str(text)
        encoded = self.tokenizer.encode(value, add_special_tokens=add_special_tokens)
        return [int(item) for item in encoded]

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode([int(item) for item in token_ids]))

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_unknown_index(self) -> int:
        return int(getattr(self.tokenizer, "unk_token_id", 0) or 0)

    def special_token(self, name: str) -> str:
        return self.special_tokens[str(name).lower().strip()]

    def special_token_id(self, name: str) -> int:
        token = self.special_token(name)
        attr_name = f"{name.lower().strip()}_token_id"
        attr_value = getattr(self.tokenizer, attr_name, None)
        if attr_value is not None:
            return int(attr_value)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            return int(self.tokenizer.convert_tokens_to_ids(token))
        encoded = self.encode(token, add_special_tokens=False)
        return int(encoded[0]) if encoded else self.get_unknown_index()

    def format_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = False, template: Any | None = None) -> str:
        active = template or self.chat_template or ChatTemplate.default()
        return active.format(messages, add_generation_prompt=add_generation_prompt)

    def to_json(self) -> dict[str, Any]:
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        payload = {
            "format": "arclm-tokenizer",
            "schema_version": "1",
            "strategy": "transformers-compat",
            "vocab_size": self.vocab_size,
            "special_tokens": dict(self.special_tokens),
            "backend": "transformers",
            "external_model_id": self.model_id,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "vocab_preview": dict(list(vocab.items())[:128]),
        }
        if isinstance(self.chat_template, ChatTemplate):
            payload["chat_template"] = self.chat_template.to_dict()
        return payload

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TransformersTokenizerAdapter":
        model_id = data.get("external_model_id")
        if not model_id:
            raise ValueError("transformers-compat tokenizer metadata is missing external_model_id.")
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("Loading a transformers-compat tokenizer requires transformers.") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=data.get("revision"),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
        )
        return cls(
            tokenizer=tokenizer,
            special_tokens=dict(data.get("special_tokens") or {}),
            chat_template=ChatTemplate.from_dict(data.get("chat_template")),
            model_id=model_id,
            revision=data.get("revision"),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
        )


@dataclass(frozen=True)
class TransformersLoadResult:
    model: Any
    tokenizer: TransformersTokenizerAdapter
    config: dict[str, Any]


class TransformersBackend:
    """Loads one causal LM through Transformers when explicitly requested."""

    def load_causal_lm(
        self,
        model_id: str,
        *,
        runtime: Any,
        family: str,
        revision: str | None = None,
        trust_remote_code: bool = False,
        torch_dtype: str | None = None,
        device_map: Any | None = None,
        special_tokens: dict[str, str] | None = None,
        chat_template: Any | None = None,
    ) -> TransformersLoadResult:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised only without optional deps
            raise RuntimeError("External compatibility loading requires transformers and torch.") from exc

        dtype = self._resolve_dtype(torch, torch_dtype, runtime)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs = {"revision": revision, "trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if device_map is None:
            model.to(runtime.torch_device())
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code).to_dict()
        return TransformersLoadResult(
            model=TransformersCausalLMWrapper(model),
            tokenizer=TransformersTokenizerAdapter(
                tokenizer=tokenizer,
                special_tokens=dict(special_tokens or {}),
                chat_template=chat_template or ChatTemplate.default(),
                model_id=model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
            ),
            config={
                "vocab_size": int(getattr(model.config, "vocab_size", len(tokenizer))),
                "block_size": int(getattr(model.config, "max_position_embeddings", 2048) or 2048),
                "model_family": family,
                "external_model_id": model_id,
                "compatibility_backend": "transformers",
                "revision": revision,
                "trust_remote_code": trust_remote_code,
                "torch_dtype": str(dtype).replace("torch.", "") if dtype is not None else None,
                "backend_config": config,
            },
        )

    @staticmethod
    def _resolve_dtype(torch: Any, requested: str | None, runtime: Any) -> Any | None:
        value = str(requested or getattr(runtime, "precision", "") or "").lower().replace("torch.", "").strip()
        if value in {"", "auto", "float32", "fp32"}:
            return None
        if value in {"float16", "fp16"}:
            return torch.float16
        if value in {"bfloat16", "bf16"}:
            return torch.bfloat16
        raise ValueError("torch_dtype must be one of: auto, float32, float16, bfloat16.")


__all__ = ["TransformersBackend", "TransformersCausalLMWrapper", "TransformersLoadResult", "TransformersTokenizerAdapter"]
