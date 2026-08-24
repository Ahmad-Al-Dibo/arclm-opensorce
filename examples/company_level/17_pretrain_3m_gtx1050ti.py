"""Pretrain a 3M+ parameter ArcLM model on a GTX 1050 Ti class machine."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime, SentencePieceTokenizer
from arclm.dataset import create_dataloader


def describe_torch_cuda() -> str:
    """Return compact PyTorch/CUDA diagnostics for device selection messages."""

    try:
        import torch
    except Exception as exc:
        return f"PyTorch import failed: {exc}"

    version = getattr(torch, "__version__", "unknown")
    cuda = getattr(torch, "cuda", None)
    cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda is None:
        return f"PyTorch {version}; CUDA module unavailable."

    available = bool(cuda.is_available())
    device_count = int(cuda.device_count()) if available else 0
    devices = [str(cuda.get_device_name(index)) for index in range(device_count)]
    suffix = f"; devices={devices}" if devices else ""
    return f"PyTorch {version}; torch.version.cuda={cuda_runtime}; cuda_available={available}; device_count={device_count}{suffix}"


def resolve_runtime(device_preference: str) -> Runtime:
    runtime = Runtime.auto(prefer=device_preference)
    for warning in runtime.warnings:
        print(f"Runtime warning: {warning}")

    if device_preference == "cuda" and runtime.device.type != "cuda":
        raise SystemExit(
            "CUDA was requested, but ArcLM resolved to CPU.\n"
            f"{describe_torch_cuda()}\n"
            "Install a CUDA-enabled PyTorch build for this environment, or rerun with --device cpu/auto."
        )

    return runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a >3M parameter ArcLM model with conservative GTX 1050 Ti defaults.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Default GTX 1050 Ti run:
    python examples/company_level/17_pretrain_3m_gtx1050ti.py

  Use your own text file:
    python examples/company_level/17_pretrain_3m_gtx1050ti.py --data data/my_corpus.txt

  Safer if CUDA runs out of memory:
    python examples/company_level/17_pretrain_3m_gtx1050ti.py --batch-size 2 --block-size 32

  Longer real pretraining run with frequent checkpoints:
    python examples/company_level/17_pretrain_3m_gtx1050ti.py --epochs 20 --eval-interval 25 --checkpoint-interval 50
""",
    )
    parser.add_argument("--data", default="data/data.txt", help="Plain-text dataset used for pretraining.")
    parser.add_argument("--output", default="output/gtx1050ti-3m-pretrained.arcmodel", help="Native .arcmodel output path.")
    parser.add_argument("--checkpoint-dir", default="output/gtx1050ti-3m-checkpoints", help="Directory for step and best checkpoints.")
    parser.add_argument("--metrics-jsonl", default="output/gtx1050ti-3m-metrics.jsonl", help="Step metrics written as JSONL.")
    parser.add_argument("--metrics-csv", default="output/gtx1050ti-3m-metrics.csv", help="Step metrics written as CSV.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"], help="Runtime device preference.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of encoded tokens reserved for eval.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate every N optimizer steps.")
    parser.add_argument("--eval-batches", type=int, default=10, help="Maximum validation batches per eval.")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Save a checkpoint every N optimizer steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer steps; 0 means all epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Print every training step loss.")
    return parser.parse_args()


def split_encoded_tokens(encoded: list[int], *, val_split: float, block_size: int) -> tuple[list[int], list[int]]:
    """Split encoded text into train/eval tokens while keeping each side usable."""

    if len(encoded) <= block_size + 1:
        raise ValueError("Dataset is too small for the requested block_size.")

    requested_val = int(len(encoded) * max(0.0, min(float(val_split), 0.5)))
    min_eval = block_size + 2
    if requested_val < min_eval:
        return encoded, []

    split_at = len(encoded) - requested_val
    if split_at <= block_size + 1:
        return encoded, []
    return encoded[:split_at], encoded[split_at:]


def next_token_loss(logits, targets):
    import torch

    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def evaluate(model, loader, *, device, max_batches: int) -> float | None:
    import torch

    if loader is None:
        return None

    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = next_token_loss(model(inputs), targets)
            losses.append(float(loss.detach().cpu().item()))

    model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def write_metric(jsonl_path: Path, csv_path: Path, row: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(model: Model, checkpoint_dir: Path, name: str, metadata: dict) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.arcmodel"
    artifact = model.save(path, overwrite=True)
    (checkpoint_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return artifact.path


def train_pretraining_loop(
    *,
    model: Model,
    train_loader,
    eval_loader,
    args: argparse.Namespace,
) -> dict:
    import torch

    device = model.runtime.torch_device()
    model.model.train()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    jsonl_path = Path(args.metrics_jsonl)
    csv_path = Path(args.metrics_csv)
    checkpoint_dir = Path(args.checkpoint_dir)
    for metrics_path in (jsonl_path, csv_path):
        if metrics_path.exists():
            metrics_path.unlink()

    best_eval_loss = None
    best_checkpoint = None
    global_step = 0
    started = time.time()
    train_losses = []
    eval_losses = []

    initial_eval_loss = evaluate(model.model, eval_loader, device=device, max_batches=args.eval_batches)
    if initial_eval_loss is not None:
        eval_losses.append(initial_eval_loss)
        row = {
            "epoch": 0,
            "step": 0,
            "train_loss": None,
            "eval_loss": initial_eval_loss,
            "learning_rate": args.learning_rate,
            "grad_norm": None,
            "elapsed_sec": round(time.time() - started, 3),
        }
        write_metric(jsonl_path, csv_path, row)
        best_eval_loss = initial_eval_loss
        best_checkpoint = save_checkpoint(
            model,
            checkpoint_dir,
            "best",
            {**row, "kind": "best", "best_eval_loss": best_eval_loss},
        )
        print(f"step=0 eval_loss={initial_eval_loss:.4f}")
        print(f"Best checkpoint: {best_checkpoint} eval_loss={best_eval_loss:.4f}")

    for epoch in range(1, args.epochs + 1):
        for inputs, targets in train_loader:
            global_step += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = next_token_loss(model.model(inputs), targets)
            loss.backward()
            grad_norm = None
            if args.grad_clip is not None and args.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.model.parameters(), args.grad_clip).detach().cpu().item())
            optimizer.step()

            train_loss = float(loss.detach().cpu().item())
            train_losses.append(train_loss)
            should_eval = bool(eval_loader is not None and args.eval_interval > 0 and global_step % args.eval_interval == 0)
            should_checkpoint = bool(args.checkpoint_interval > 0 and global_step % args.checkpoint_interval == 0)
            eval_loss = evaluate(model.model, eval_loader, device=device, max_batches=args.eval_batches) if should_eval else None
            if eval_loss is not None:
                eval_losses.append(eval_loss)

            row = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "learning_rate": args.learning_rate,
                "grad_norm": grad_norm,
                "elapsed_sec": round(time.time() - started, 3),
            }
            write_metric(jsonl_path, csv_path, row)

            if args.debug or should_eval or should_checkpoint:
                eval_text = f" eval_loss={eval_loss:.4f}" if eval_loss is not None else ""
                grad_text = f" grad_norm={grad_norm:.3f}" if grad_norm is not None else ""
                print(f"epoch={epoch} step={global_step} train_loss={train_loss:.4f}{eval_text}{grad_text}")

            if eval_loss is not None and (best_eval_loss is None or eval_loss < best_eval_loss):
                best_eval_loss = eval_loss
                best_checkpoint = save_checkpoint(
                    model,
                    checkpoint_dir,
                    "best",
                    {**row, "kind": "best", "best_eval_loss": best_eval_loss},
                )
                print(f"Best checkpoint: {best_checkpoint} eval_loss={best_eval_loss:.4f}")

            if should_checkpoint:
                checkpoint = save_checkpoint(
                    model,
                    checkpoint_dir,
                    f"step-{global_step:06d}",
                    {**row, "kind": "step"},
                )
                print(f"Checkpoint: {checkpoint}")

            if args.max_steps and global_step >= args.max_steps:
                model.model.eval()
                return {
                    "train_losses": train_losses,
                    "eval_losses": eval_losses,
                    "best_eval_loss": best_eval_loss,
                    "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
                    "steps": global_step,
                }

    model.model.eval()
    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "best_eval_loss": best_eval_loss,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "steps": global_step,
    }


def main():
    args = parse_args()

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    runtime = resolve_runtime(args.device)
    dataset = Dataset.load(args.data, format="txt")
    inspection = dataset.inspect().to_dict()
    tokenizer = SentencePieceTokenizer(max_vocab=args.max_vocab, model_type="bpe")
    tokenizer.build(dataset.text())
    encoded = tokenizer.encode_text(dataset.text())
    train_ids, eval_ids = split_encoded_tokens(encoded, val_split=args.val_split, block_size=args.block_size)
    train_loader = create_dataloader(train_ids, block_size=args.block_size, batch_size=args.batch_size, shuffle=True)
    eval_loader = create_dataloader(eval_ids, block_size=args.block_size, batch_size=args.batch_size, shuffle=False) if eval_ids else None

    model = Model.create(
        architecture="arclm-native",
        tokenizer=tokenizer,
        runtime=runtime,
        embed_dim=args.embed_dim,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        dropout=0.1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    report = model.inspect()
    parameters = int(report["parameters"])

    print(f"Device: {runtime.device_name}")
    print(f"Precision: {runtime.precision}")
    print(describe_torch_cuda())
    print(f"Dataset records: {inspection['records']}, chars: {inspection['characters']}, encoded tokens: {len(encoded)}")
    print(f"Train tokens: {len(train_ids)}, eval tokens: {len(eval_ids)}")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print(f"Model parameters: {parameters:,}")
    print(f"Metrics JSONL: {args.metrics_jsonl}")
    print(f"Metrics CSV: {args.metrics_csv}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    if parameters < 3_000_000:
        print("Warning: this dataset/tokenizer produced fewer than 3M parameters. Increase --embed-dim or --num-blocks.")

    history = train_pretraining_loop(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        args=args,
    )
    artifact = model.save(args.output, overwrite=True)

    print(f"Steps: {history['steps']}")
    print(f"Final train loss: {history['train_losses'][-1] if history['train_losses'] else None}")
    print(f"Best eval loss: {history['best_eval_loss']}")
    print(f"Best checkpoint: {history['best_checkpoint']}")
    print(f"Saved: {artifact.path}")
    print("Sample:", model.generate("ArcLM", max_new_tokens=24, temperature=0.0))


if __name__ == "__main__":
    main()
