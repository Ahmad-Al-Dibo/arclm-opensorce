"""Load a saved `.arcmodel` from output/ and generate text."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Model, Runtime


def discover_arcmodels(root: Path) -> list[Path]:
    """Return `.arcmodel` candidates newest-first."""

    return sorted(
        root.glob("*.arcmodel"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def print_artifacts(root: Path, artifacts: list[Path]) -> None:
    print(f"Found {len(artifacts)} `.arcmodel` artifacts in {root}:")
    print(f"{'Path':<50} {'Size (bytes)':>15} {'Modified':>25}")
    print("-" * 90)
    for artifact in artifacts:
        stat = artifact.stat()
        print(f"{str(artifact):<50} {stat.st_size:>15} {stat.st_mtime:>25}")


def load_latest_arcmodel(root: Path, runtime: Runtime) -> tuple[Path, Model, list[str]]:
    """Load the newest valid `.arcmodel` artifact under root."""

    artifacts = discover_arcmodels(root)
    if not artifacts:
        raise FileNotFoundError(
            f"No `.arcmodel` artifacts found in {root}/. "
            "Train or save a model first, or pass a path explicitly."
        )

    print_artifacts(root, artifacts)
    skipped: list[str] = []
    for artifact_path in artifacts:
        try:
            return artifact_path, Model.load(artifact_path, runtime=runtime), skipped
        except Exception as exc:  # noqa: BLE001 - discovery should keep trying candidates.
            skipped.append(f"{artifact_path}: {exc}")

    details = "\n".join(f"- {item}" for item in skipped)
    raise FileNotFoundError(f"No loadable ArcLM artifacts were found in {root}/.\n{details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and use an ArcLM .arcmodel from output/.")
    parser.add_argument("artifact", nargs="?", help="Optional path to a specific .arcmodel artifact.")
    parser.add_argument("--output-dir", default="output", help="Folder searched when no artifact path is passed.")
    parser.add_argument("--prompt", default="ArcLM", help="Prompt used for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Number of new tokens to generate.")
    return parser.parse_args()


def main():
    args = parse_args()

    runtime = Runtime.auto(prefer="cpu")
    skipped: list[str] = []
    if args.artifact:
        artifact_path = Path(args.artifact)
        model = Model.load(artifact_path, runtime=runtime)
    else:
        artifact_path, model, skipped = load_latest_arcmodel(Path(args.output_dir), runtime)

    response = model.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)

    print(f"Loaded: {artifact_path}")
    print(f"Architecture: {model.inspect()['architecture_id']}")
    print(f"Tokenizer: {type(model.tokenizer).__name__}")
    print(f"Prompt: {args.prompt}")
    print(f"Response: {response}")
    if skipped:
        print("\nSkipped older/unreadable artifacts:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
