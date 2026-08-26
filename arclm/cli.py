"""Small command-line entry point for the current ArcLM architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .architectures import architectures
from .datasets import Dataset
from .runtime import Runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arclm", description="ArcLM architecture utility commands.")
    parser.add_argument("--version", action="store_true", help="Print the ArcLM version.")
    subparsers = parser.add_subparsers(dest="command")

    runtime_parser = subparsers.add_parser("runtime", help="Inspect the local ArcLM runtime.")
    runtime_parser.add_argument("--prefer", default="auto", help="Preferred device: auto, cpu, cuda, or cuda:<index>.")
    runtime_parser.add_argument("--precision", default="auto", help="Preferred precision.")

    dataset_parser = subparsers.add_parser("inspect-dataset", help="Inspect a local txt/json/jsonl dataset.")
    dataset_parser.add_argument("path", type=Path)

    subparsers.add_parser("models", help="List registered ArcLM model architectures.")

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "runtime":
        return _print_json(Runtime.auto(prefer=args.prefer, precision=args.precision).to_dict())
    if args.command == "inspect-dataset":
        return _print_json(Dataset.load(args.path).inspect().to_dict())
    if args.command == "models":
        return _print_json(architectures.available())

    parser.print_help()
    return 0


def _print_json(payload: Any) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = ["main"]
