"""Loader for Hugging Face causal language model folders or IDs."""

from pathlib import Path
from typing import Any

from .base import CheckpointLoader, LoadedCheckpoint


class HuggingFaceLoader(CheckpointLoader):
    """Normalize a Hugging Face model into a ``LoadedCheckpoint``."""

    source_type = "huggingface"

    def can_load(self, source: Any) -> bool:
        source_str = str(source)
        path = Path(source_str)
        if path.is_file():
            return False
        if path.is_dir():
            return (path / "config.json").exists()
        if source_str.endswith((".pth", ".pt", ".bin", ".ckpt", ".safetensors")):
            return False
        return bool(source_str.strip())

    def load(self, source: Any, map_location: str = "cpu") -> LoadedCheckpoint:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Loading Hugging Face sources requires transformers. "
                "Install it with: pip install transformers"
            ) from exc

        source_str = str(source)
        hf_config = AutoConfig.from_pretrained(source_str, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            source_str,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.to(map_location)
        state_dict = model.state_dict()

        tokenizer_metadata = {"tokenizer_type": "huggingface", "model_id": source_str}
        vocab = None
        stoi = None
        itos = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(source_str, trust_remote_code=True)
            vocab_data = tokenizer.get_vocab()
            stoi = {str(token): int(index) for token, index in vocab_data.items()}
            itos = {index: token for token, index in stoi.items()}
            vocab = [itos[index] for index in sorted(itos)]
            tokenizer_metadata["vocab_size"] = len(stoi)
        except Exception as exc:
            tokenizer_metadata["tokenizer_warning"] = str(exc)

        return LoadedCheckpoint(
            source=source_str,
            source_type=self.source_type,
            state_dict=dict(state_dict),
            config=hf_config.to_dict(),
            vocab_size=getattr(hf_config, "vocab_size", None),
            tokenizer_metadata=tokenizer_metadata,
            vocab=vocab,
            stoi=stoi,
            itos=itos,
            metadata={"model_id": source_str, "architectures": getattr(hf_config, "architectures", None)},
        )
