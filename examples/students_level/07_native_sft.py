"""Native adapter fine-tuning: train a tiny LoRA adapter with Model.finetune."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime, Trainer


def main():
    records = [
        {"instruction": "Explain ArcLM in one sentence.", "output": "ArcLM is a compact language-model toolkit."},
        {"instruction": "What does adapter training do?", "output": "It trains small extra weights while preserving the base model."},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        adapter_path = root / "student_lora.arcadapter"

        runtime = Runtime.auto(prefer="cpu")
        dataset = Dataset.load(records)
        prepared = dataset.prepare(block_size=8, batch_size=2)
        model = Model.create(
            architecture="arclm-native",
            tokenizer=prepared.tokenizer,
            runtime=runtime,
            embed_dim=16,
            block_size=8,
            num_blocks=1,
            dropout=0.0,
        )

        Trainer(model=model, dataset=dataset, epochs=1, batch_size=2, learning_rate=1e-3, block_size=8).train()
        model.attach_lora(rank=2, alpha=4.0)
        history = model.finetune(data=dataset, method="lora", epochs=1, batch_size=2, learning_rate=1e-3, block_size=8)
        manifest = model.save_adapter(adapter_path, overwrite=True)

        print(f"Adapter losses: {history['train_losses']}")
        print(f"Saved adapter type: {manifest.adapter_type}")
        print(f"Saved adapter: {adapter_path}")


if __name__ == "__main__":
    main()
