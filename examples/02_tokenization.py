"""Tokenization: build a word tokenizer and round-trip text."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import Tokenizer


def main():
    tokenizer = Tokenizer(
        max_vocab=32,
        user_defined_symbols=["<|instruction|>", "<|response|>"],
    )
    tokenizer.build("ArcLM trains small models. ArcLM examples stay readable.")

    ids = tokenizer.encode_text("<|instruction|> ArcLM trains models")
    text = tokenizer.decode_string(ids)

    print("Token IDs:", ids)
    print("Decoded:", text)
    print("Vocab size:", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
