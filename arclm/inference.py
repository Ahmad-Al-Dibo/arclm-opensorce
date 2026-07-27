"""Reusable inference interface for trained ArcLM checkpoints."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import torch

from .config import Config
from .generator import Generator
from .pipeline import build_model
from .tokenizer import SentencePieceTokenizer


DEFAULT_MODEL_PATH = Path("models/arclm.pth")


class CheckpointTokenizer:
    """Fallback tokenizer reconstructed from checkpoint vocabulary data."""

    def __init__(self, stoi, itos, tokenizer_type="word"):
        self.stoi = stoi
        self.itos = itos
        self.tokenizer_type = tokenizer_type
        self.unk_idx = self.stoi.get("<UNK>", self.stoi.get("<unk>", 0))

    def encode_text(self, text):
        tokens = text.lower().split()
        return [self.stoi.get(token, self.unk_idx) for token in tokens if token]

    def decode_string(self, indices):
        tokens = [self.itos[idx] for idx in indices if idx in self.itos]
        sentencepiece_space = "\u2581"
        if self.tokenizer_type == "sentencepiece" or any(
            token.startswith(sentencepiece_space) for token in tokens
        ):
            return "".join(tokens).replace(sentencepiece_space, " ").strip()
        return " ".join(tokens)


@dataclass
class LoadedModel:
    """Loaded model bundle with a minimal prediction method."""

    model: torch.nn.Module
    generator: Generator
    config: Config
    model_path: Path
    device: torch.device

    def predict(
        self,
        input_text,
        max_new_tokens=80,
        temperature=0.9,
        repetition_penalty=1.2,
        top_k=None,
        top_p=0.9,
    ):
        """Generate a prediction from raw input text."""
        if not isinstance(input_text, str):
            raise TypeError("input_text must be a string.")

        input_text = input_text.strip()
        if not input_text:
            raise ValueError("input_text cannot be empty.")

        return self.generator.generate_string(
            input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
        )
    
    # to use loadedModel.train from model, we can add a method to call model it self by any not defined method in LoadedModel
    def __getattr__(self, name):
        return getattr(self.model, name)

    # TypeError: 'LoadedModel' object is not callable
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

_cached_model = None
_cached_key = None


def load_model(model_path=None, device=None, prefer_best=True):
    """
    Load a trained ArcLM checkpoint for inference.

    Returns a LoadedModel object with a .predict(input_text) method.

    This can not load external Hugging Face models; use load_any_model() for that.
    """
    resolved_path = Path(model_path or DEFAULT_MODEL_PATH)
    resolved_device = _resolve_device(device)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {resolved_path}. "
            "Finish training first or pass model_path explicitly."
            "or check if the model path is correct."
        )

    checkpoint = _load_checkpoint(resolved_path, resolved_device)
    config = _build_inference_config(checkpoint, resolved_path, resolved_device)
    vocab_size = _get_required(checkpoint, "vocab_size")
    stoi = _normalize_stoi(_get_required(checkpoint, "stoi"))
    itos = _normalize_itos(_get_required(checkpoint, "itos"))

    model = build_model(config, vocab_size)
    state_dict = _select_state_dict(checkpoint, prefer_best=prefer_best)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tokenizer = _build_tokenizer(checkpoint, config, stoi, itos)
    generator = Generator(
        model=model,
        stoi=stoi,
        itos=itos,
        block_size=config.block_size,
        device=resolved_device,
        tokenizer=tokenizer,
    )

    return LoadedModel(
        model=model,
        generator=generator,
        config=config,
        model_path=resolved_path,
        device=resolved_device,
    )


def predict(input_text, model_path=None, device=None, reload=False, **generation_options):
    """Predict with a cached model using the same defaults as load_model()."""
    global _cached_key, _cached_model

    resolved_path = Path(model_path or DEFAULT_MODEL_PATH)
    resolved_device = _resolve_device(device)
    cache_key = (resolved_path.resolve(), str(resolved_device))

    if reload or _cached_model is None or _cached_key != cache_key:
        _cached_model = load_model(resolved_path, resolved_device)
        _cached_key = cache_key

    return _cached_model.predict(input_text, **generation_options)


def _resolve_device(device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def _build_inference_config(checkpoint, model_path, device):
    config_data = dict(checkpoint.get("config") or {})
    config_data["model_path"] = str(model_path)
    config_data["device"] = str(device)
    config_data["batch_size"] = 1
    config_data["num_epochs"] = 1

    if "block_size" not in config_data and "block_size" in checkpoint:
        config_data["block_size"] = checkpoint["block_size"]

    return Config(**config_data)


def _select_state_dict(checkpoint, prefer_best=True):
    if prefer_best and checkpoint.get("best_model_state_dict") is not None:
        return checkpoint["best_model_state_dict"]

    for key in ("model_state_dict", "model", "state_dict"):
        if key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]

    raise ValueError("Checkpoint does not contain model weights.")


def _build_tokenizer(checkpoint, config, stoi, itos):
    metadata = checkpoint.get("tokenizer_metadata") or {}
    tokenizer_type = metadata.get(
        "tokenizer_type",
        getattr(config, "tokenizer_type", "word"),
    )

    if tokenizer_type == "sentencepiece" and metadata.get("model_proto"):
        return SentencePieceTokenizer.from_checkpoint(metadata)

    if tokenizer_type == "sentencepiece":
        warnings.warn(
            "Checkpoint uses sentencepiece but has no tokenizer_metadata.model_proto. "
            "Falling back to stoi/itos token lookup; retrain or resave the checkpoint "
            "with the updated training code for exact tokenizer parity.",
            RuntimeWarning,
            stacklevel=2,
        )

    return CheckpointTokenizer(
        stoi=stoi,
        itos=itos,
        tokenizer_type=tokenizer_type,
    )


def _get_required(checkpoint, key):
    value = checkpoint.get(key)
    if value is None:
        raise ValueError(f"Checkpoint is missing required field: {key}.")
    return value


def _normalize_stoi(stoi):
    return {str(token): int(index) for token, index in stoi.items()}


def _normalize_itos(itos):
    return {int(index): str(token) for index, token in itos.items()}
