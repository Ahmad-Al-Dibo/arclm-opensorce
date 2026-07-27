"""Small functional benchmark for the Qwen3 SFT example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS = EXAMPLE_DIR / "data" / "benchmark_prompts.jsonl"
DEFAULT_BASE = EXAMPLE_DIR / "output" / "base_outputs.jsonl"
DEFAULT_FINETUNED = EXAMPLE_DIR / "output" / "finetuned_outputs.jsonl"
DEFAULT_OUTPUT = EXAMPLE_DIR / "output" / "benchmark_results.json"


def main():
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned example outputs.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--base-outputs", default=str(DEFAULT_BASE))
    parser.add_argument("--finetuned-outputs", default=str(DEFAULT_FINETUNED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    print("This is a small functional benchmark, not a full model evaluation.")
    results = compare_outputs(
        prompts_path=Path(args.prompts),
        base_outputs_path=Path(args.base_outputs),
        finetuned_outputs_path=Path(args.finetuned_outputs),
    )
    print_table(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results: {output_path}")


def compare_outputs(prompts_path: Path, base_outputs_path: Path, finetuned_outputs_path: Path):
    prompts = {row["id"]: row for row in read_jsonl(prompts_path)}
    base = {row["id"]: row for row in read_jsonl_required(base_outputs_path)}
    finetuned = {row["id"]: row for row in read_jsonl_required(finetuned_outputs_path)}
    results = []
    for prompt_id, prompt in prompts.items():
        expected = prompt.get("expected_keywords", [])
        base_text = base.get(prompt_id, {}).get("generated_text", "")
        finetuned_text = finetuned.get(prompt_id, {}).get("generated_text", "")
        results.append({
            "id": prompt_id,
            "prompt": prompt["prompt"],
            "expected_keywords": expected,
            "base_output": base_text,
            "finetuned_output": finetuned_text,
            "base_score": keyword_score(base_text, expected),
            "finetuned_score": keyword_score(finetuned_text, expected),
            "base_answer_length": len(base_text.split()),
            "finetuned_answer_length": len(finetuned_text.split()),
            "base_follows_instruction": follows_instruction(prompt["prompt"], base_text),
            "finetuned_follows_instruction": follows_instruction(prompt["prompt"], finetuned_text),
        })
    return results


def keyword_score(text: str, expected_keywords):
    if not expected_keywords:
        return 0.0
    normalized = text.lower()
    found = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return found / len(expected_keywords)


def follows_instruction(prompt: str, answer: str):
    if not answer.strip():
        return False
    if "one sentence" in prompt.lower():
        sentence_marks = sum(answer.count(mark) for mark in ".!?")
        return sentence_marks <= 2
    return len(answer.split()) <= 160


def print_table(results):
    print()
    print(f"{'Prompt ID':<24} {'Base Score':<14} {'Fine-tuned Score':<18}")
    for row in results:
        print(f"{row['id']:<24} {row['base_score']:<14.2f} {row['finetuned_score']:<18.2f}")


def read_jsonl_required(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Missing output file: {path}\n"
            "Run test_base_model.py and test_finetuned_model.py before benchmarking."
        )
    return list(read_jsonl(path))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
