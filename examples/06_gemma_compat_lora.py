"""Gemma compatibility LoRA fine-tuning workflow.

Set ARCLM_GEMMA_MODEL to a real HF model reference such as
hf://google/gemma-2b-it to run against external weights.
"""

from __future__ import annotations

import os
from pathlib import Path


def run(output_dir: str | Path = "outputs/gemma-compat") -> dict:
    from arclm import ChatTemplate, Dataset, Model, Runtime, Trainer

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_source = os.environ.get("ARCLM_GEMMA_MODEL", "arclm://compat/gemma-tiny")

    template = ChatTemplate(
        name="gemma-ish",
        user="<start_of_turn>user\n{content}<end_of_turn>",
        assistant="<start_of_turn>model\n{content}<end_of_turn>",
        separator="\n",
    )
    dataset = Dataset.load(
        [
            {
                "messages": [
                    {"role": "user", "content": "Summarize ArcLM in one sentence."},
                    {"role": "assistant", "content": "ArcLM is a small owned framework for language-model experiments."},
                ]
            },
            {
                "prompt": "Name the training style used here.",
                "completion": "LoRA fine-tuning.",
            },
        ]
    ).clean(remove_empty=True, normalize_whitespace=True)
    validation = Dataset.load([{"prompt": "Say hello.", "completion": "Hello from ArcLM."}])

    model = Model.load(model_source, runtime=Runtime.auto(prefer="cpu"))
    model.tokenizer.chat_template = template
    trainer = Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        method="lora",
        rank=2,
        alpha=4.0,
        epochs=1,
        steps=1,
        batch_size=1,
        block_size=4,
        checkpoint_interval=1,
        checkpoint_dir=root / "checkpoints",
        save_adapter=True,
        adapter_path=root / "gemma-lora",
        save_artifact=True,
        artifact_path=root / "gemma-finetuned.arcmodel",
        shuffle=False,
    )
    prepared_preview = dataset.to_training_text(chat_template=template)
    memory = trainer.memory_plan()
    history = trainer.fine_tune()
    return {
        "model_source": model_source,
        "formatted_preview": prepared_preview,
        "memory_plan": memory,
        "history": history,
    }


if __name__ == "__main__":
    print(run())
