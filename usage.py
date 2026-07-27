"""Use a Qwen model fine-tuned by train.py.

The training script saves either a PEFT LoRA adapter or a full Hugging Face
model directory. This file loads the correct format automatically.

example usage:
    python usage.py --model-dir models/qwen_medical_o1_lora --question "A patient has sudden left-sided weakness after a long flight and a swollen tender calf. Which cardiac abnormality could explain these findings?"
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_DIR = Path("models") / "qwen_medical_o1_lora"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful medical question-answering assistant. Answer accurately "
    "from the provided question. This model is for research and must not replace "
    "professional medical advice."
)
DEFAULT_QUESTION = (
    "A patient has sudden left-sided weakness after a long flight and a swollen "
    "tender calf. Which cardiac abnormality could explain these findings?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an answer with the fine-tuned Qwen model.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--question-file", type=Path, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_finetuned_model(args)

    if args.interactive:
        run_repl(tokenizer, model, args)
        return

    question = read_question(args)
    answer = generate_answer(tokenizer, model, question, args)
    print(answer)


def load_finetuned_model(args: argparse.Namespace):
    if not args.model_dir.exists():
        raise SystemExit(
            f"Model directory not found: {args.model_dir}\n"
            "Run train.py first, or pass --model-dir to an existing adapter/model directory."
        )

    model_dir = resolve_model_dir(args.model_dir)
    if model_dir != args.model_dir:
        print(f"Using checkpoint: {model_dir}")

    if (model_dir / "adapter_config.json").exists():
        tokenizer, base_model = load_hf_model(args.base_model, args)
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Loading a LoRA adapter requires peft. Install it with: pip install peft") from exc
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        return tokenizer, model

    return load_hf_model(str(model_dir), args)


def resolve_model_dir(model_dir: Path) -> Path:
    if is_hf_model_dir(model_dir):
        return model_dir

    checkpoints_dir = model_dir / "checkpoints"
    if checkpoints_dir.exists():
        candidates = [path for path in checkpoints_dir.iterdir() if path.is_dir() and is_hf_model_dir(path)]
        if candidates:
            return max(candidates, key=checkpoint_sort_key)

    raise SystemExit(
        f"No loadable model or adapter found in: {model_dir}\n"
        "Expected adapter_config.json or a full Hugging Face model in that directory. "
        "If training stopped before the final save, pass a checkpoint directory such as "
        "--model-dir models/qwen_medical_o1_lora/checkpoints/step-000600."
    )


def is_hf_model_dir(path: Path) -> bool:
    if (path / "adapter_config.json").exists():
        return True
    has_config = (path / "config.json").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return has_config and has_weights


def checkpoint_sort_key(path: Path) -> tuple[int, float]:
    match = re.search(r"(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    return step, path.stat().st_mtime


def load_hf_model(model_name_or_path: str, args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    dtype = resolve_dtype(args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map and args.device_map.lower() not in {"none", "null"}:
        model_kwargs["device_map"] = args.device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except TypeError:
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if "device_map" not in model_kwargs:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer: Any, model: Any, question: str, args: argparse.Namespace) -> str:
    messages = build_messages(question, args.system_prompt)
    input_ids = render_prompt(tokenizer, messages, args.enable_thinking).to(model_input_device(model))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]], enable_thinking: bool) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors="pt",
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        return tensor_from_encoded_prompt(encoded)

    text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    text += "\nassistant:"
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def tensor_from_encoded_prompt(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if not torch.is_tensor(encoded):
        encoded = torch.tensor(encoded, dtype=torch.long)
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    return encoded


def read_question(args: argparse.Namespace) -> str:
    if args.question_file is not None:
        return args.question_file.read_text(encoding="utf-8").strip()
    return args.question


def run_repl(tokenizer: Any, model: Any, args: argparse.Namespace) -> None:
    print("Interactive mode. Press Ctrl+C or submit an empty question to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            break
        print(generate_answer(tokenizer, model, question, args))


def resolve_dtype(dtype: str):
    if dtype in (None, "auto"):
        return "auto"
    normalized = str(dtype).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError("dtype must be one of: auto, fp16, float16, bf16, bfloat16, fp32, float32.")
    return mapping[normalized]


def model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


if __name__ == "__main__":
    main()
