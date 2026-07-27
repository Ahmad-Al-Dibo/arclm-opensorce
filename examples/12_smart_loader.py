"""SmartLoader: inspect a local Hugging Face-style model folder."""

from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import SmartLoader


def main():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "tiny_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "_name_or_path": "tiny_model",
                    "model_type": "gpt2",
                    "architectures": ["GPT2LMHeadModel"],
                    "torch_dtype": "float32",
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_text("", encoding="utf-8")

        plan = SmartLoader.inspect(model_dir)
        print(plan.format_report())


if __name__ == "__main__":
    main()
