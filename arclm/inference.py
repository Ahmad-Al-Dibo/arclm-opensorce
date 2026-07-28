"""Reusable inference interface for trained ArcLM checkpoints."""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import warnings
from typing import Any, Iterable, Optional

import torch

from .config import Config
from .generator import Generator
from .pipeline import build_model
from .tokenizer import SentencePieceTokenizer


DEFAULT_MODEL_PATH = Path("models/arclm.pth")


@dataclass(frozen=True)
class GenerationConfig:
    """Validated text-generation configuration."""

    max_new_tokens: int = 20
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    repetition_penalty: float = 1.0
    num_beams: int = 1
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero.")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Structured result returned by batched generation."""

    outputs: list[str]
    prompt_tokens: list[int]
    generated_tokens: list[int]
    latency_seconds: float
    tokens_per_second: float
    errors: list[dict[str, Any]] = field(default_factory=list)
    report_type: str = "generation_result"
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StreamEvent:
    """Library-level streaming generation event."""

    type: str
    text: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    This cannot load external Hugging Face models; use
    ``arclm.models.load_model`` for the consolidated model-loading facade.
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


def generate(
    model_bundle: Any,
    *,
    prompts: Iterable[str],
    config: GenerationConfig | None = None,
    batch_size: int = 1,
) -> GenerationResult:
    """Generate text for multiple prompts while preserving input order."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    generation_config = config or GenerationConfig()
    if generation_config.seed is not None:
        torch.manual_seed(generation_config.seed)
    prompt_list = list(prompts)
    outputs: list[str] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset in range(0, len(prompt_list), batch_size):
        for index, prompt_text in enumerate(prompt_list[offset:offset + batch_size], start=offset):
            try:
                before_tokens = _count_prompt_tokens(model_bundle, prompt_text)
                output = _predict_with_bundle(model_bundle, prompt_text, generation_config)
                output = _apply_stop_strings(output, generation_config.stop)
                after_tokens = max(0, _count_text_tokens(model_bundle, output) - before_tokens)
                outputs.append(output)
                prompt_tokens.append(before_tokens)
                generated_tokens.append(after_tokens)
            except Exception as exc:
                outputs.append("")
                prompt_tokens.append(0)
                generated_tokens.append(0)
                errors.append({"index": index, "type": type(exc).__name__, "message": str(exc)})
    latency = time.perf_counter() - started
    generated_total = sum(generated_tokens)
    return GenerationResult(
        outputs=outputs,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        latency_seconds=latency,
        tokens_per_second=(generated_total / latency if latency > 0 else 0.0),
        errors=errors,
    )


def stream_generate(
    model_bundle: Any,
    *,
    prompt: str,
    config: GenerationConfig | None = None,
):
    """Yield start, delta, and completion events for supported prediction APIs."""

    yield StreamEvent(type="start")
    try:
        result = generate(model_bundle, prompts=[prompt], config=config, batch_size=1)
        if result.errors:
            yield StreamEvent(type="error", error=result.errors[0]["message"])
            return
        yield StreamEvent(type="delta", text=result.outputs[0])
        yield StreamEvent(type="completed")
    except Exception as exc:
        yield StreamEvent(type="error", error=str(exc))


def _resolve_device(device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _predict_with_bundle(model_bundle: Any, prompt: str, config: GenerationConfig) -> str:
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
    }
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.do_sample is not None:
        kwargs["do_sample"] = config.do_sample
    if hasattr(model_bundle, "predict"):
        return str(model_bundle.predict(prompt, **kwargs))
    raise TypeError("model_bundle must expose a predict(prompt, **kwargs) method.")


def _count_prompt_tokens(model_bundle: Any, text: str) -> int:
    tokenizer = getattr(model_bundle, "tokenizer", None)
    if tokenizer is None and hasattr(model_bundle, "generator"):
        tokenizer = getattr(model_bundle.generator, "tokenizer", None)
    if tokenizer is None:
        return len(text.split())
    if hasattr(tokenizer, "encode_text"):
        return len(tokenizer.encode_text(text))
    encoded = tokenizer(text)
    if isinstance(encoded, dict):
        return len(encoded.get("input_ids", []))
    return len(getattr(encoded, "input_ids", []))


def _count_text_tokens(model_bundle: Any, text: str) -> int:
    return _count_prompt_tokens(model_bundle, text)


def _apply_stop_strings(text: str, stops: list[str]) -> str:
    output = text
    for stop in stops:
        if stop and stop in output:
            output = output.split(stop, 1)[0]
    return output


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
