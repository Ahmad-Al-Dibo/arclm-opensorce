"""
ArcLM Command-Line Interface (CLI)

Provides command-line tools for training, evaluation, and generation.

Available Commands:
  - arclm train    : Train a model from scratch or fine-tune
  - arclm eval     : Evaluate model performance
  - arclm generate : Generate text from a trained model
  - arclm --run simple-interface : Start the simple web interface

Usage:
  arclm train --config config.yaml --data data.txt
  arclm eval --model models/model.pth --data test.txt
  arclm generate --model models/model.pth --prompt "Hello"
  python -m arclm --run simple-interface
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import torch

from arclm import (
    Config,
    UnifiedPipeline,
    calculate_metrics,
    export_metrics_to_json,
    export_metrics_to_markdown,
    load_model,
    create_tokenizer,
    TextDataset,
    create_dataloader,
)
from arclm.config_loader import load_config_yaml, load_config_json


def create_train_parser(subparsers):
    """Create parser for arclm train command"""
    parser = subparsers.add_parser(
        "train",
        help="Train a model from scratch or fine-tune",
        description="Train ArcLM model with configuration file or CLI arguments"
    )
    
    # Config input
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (YAML or JSON) - overrides CLI args"
    )
    
    # Model config
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--block-size", type=int, default=512, help="Context window size")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    
    # Training config
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--mode", type=str, default="pre_training",
                        choices=["pre_training", "fine_tuning", "instruction_tuning"],
                        help="Training mode")
    
    # Data & checkpoint
    parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--output", type=str, default="models/trained_model.pth",
                        help="Output model checkpoint path")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pre-trained model (for fine-tuning)")
    
    # Logging & tracking
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name (for tracking)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for logs and reports")
    
    parser.set_defaults(func=train_command)
    return parser


def create_eval_parser(subparsers):
    """Create parser for arclm eval command"""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate model performance",
        description="Evaluate trained model on validation/test data"
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to evaluation data")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (if not in model checkpoint)")
    parser.add_argument("--output", type=str, default="metrics_report.json",
                        help="Output metrics file")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "auto"],
                        help="Device to use for evaluation")
    
    parser.set_defaults(func=eval_command)
    return parser


def create_generate_parser(subparsers):
    """Create parser for arclm generate command"""
    parser = subparsers.add_parser(
        "generate",
        help="Generate text from trained model",
        description="Use trained model to generate text"
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to generate from")
    parser.add_argument("--length", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "auto"],
                        help="Device to use for generation")
    
    parser.set_defaults(func=generate_command)
    return parser


def train_command(args):
    """Execute training"""
    print("\n" + "="*70)
    print("ArcLM Training")
    print("="*70)
    
    # Load config
    if args.config:
        print(f"\n📂 Loading config from: {args.config}")
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = load_config_yaml(args.config)
        elif args.config.endswith(".json"):
            config = load_config_json(args.config)
        else:
            print(f"❌ Unknown config format: {args.config}")
            return 1
    else:
        print("\n⚙️ Creating config from CLI arguments")
        config = Config(
            embed_dim=args.embed_dim,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            num_heads=args.num_heads,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    
    print(f"✓ Config loaded: {config.embed_dim}D, {config.num_blocks} blocks")
    
    # Load data
    try:
        print(f"\n📊 Loading data from: {args.data}")
        with open(args.data, "r") as f:
            text = f.read()
        
        tokenizer = create_tokenizer("word", max_vocab=10000)
        dataset = TextDataset(text, tokenizer, seq_len=config.block_size)
        train_loader = create_dataloader(dataset, batch_size=config.batch_size)
        print(f"✓ Data loaded: {len(dataset)} samples")
    except FileNotFoundError:
        print(f"❌ Data file not found: {args.data}")
        return 1
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Create pipeline
    print(f"\n🔧 Creating pipeline (mode: {args.mode})")
    pipeline = UnifiedPipeline(config, mode=args.mode)
    
    # Load pretrained if needed
    if args.pretrained:
        print(f"📦 Loading pre-trained model: {args.pretrained}")
        from arclm import PreTrainedModelLoader
        loader = PreTrainedModelLoader(args.pretrained)
        model, _ = loader.load()
        pipeline.build(vocab_size=tokenizer.max_vocab, pretrained_model=model)
    else:
        pipeline.build(vocab_size=tokenizer.max_vocab)
    
    print("✓ Pipeline ready")
    
    # Train
    print(f"\n🚀 Starting training ({args.epochs} epochs)...")
    try:
        results = pipeline.train(train_loader, num_epochs=args.epochs)
        print(f"✓ Training complete!")
        print(f"  - Final loss: {results['final_loss']:.4f}")
    except Exception as e:
        print(f"❌ Training error: {e}")
        return 1
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pipeline.model.state_dict(), output_path)
    print(f"\n💾 Model saved to: {output_path}")
    
    # Save config
    config_path = output_path.parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"💾 Config saved to: {config_path}")
    
    # Log experiment
    if args.experiment:
        log_dir = Path(args.log_dir) / args.experiment
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "training_log.json"
        with open(log_file, "w") as f:
            json.dump({
                "model_path": str(output_path),
                "config_path": str(config_path),
                "final_loss": float(results["final_loss"]),
                "epochs": args.epochs,
            }, f, indent=2)
        print(f"📝 Experiment logged: {log_file}")
    
    print("\n" + "="*70 + "\n")
    return 0


def eval_command(args):
    """Execute evaluation"""
    print("\n" + "="*70)
    print("ArcLM Evaluation")
    print("="*70)
    
    # Load model
    try:
        print(f"\n📦 Loading model from: {args.model}")
        model = load_model(args.model, device=args.device)
        print("✓ Model loaded")
    except FileNotFoundError:
        print(f"❌ Model file not found: {args.model}")
        return 1
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Load config
    if args.config:
        if args.config.endswith(".yaml"):
            config = load_config_yaml(args.config)
        else:
            config = load_config_json(args.config)
    else:
        config_path = Path(args.model).parent / "config.json"
        if config_path.exists():
            config = load_config_json(str(config_path))
        else:
            print("⚠️ Config not found, using defaults")
            config = Config()
    
    # Load data
    try:
        print(f"\n📊 Loading data from: {args.data}")
        with open(args.data, "r") as f:
            text = f.read()
        
        tokenizer = create_tokenizer("word", max_vocab=10000)
        dataset = TextDataset(text, tokenizer, seq_len=config.block_size)
        val_loader = create_dataloader(dataset, batch_size=args.batch_size)
        print(f"✓ Data loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Evaluate
    print(f"\n📈 Evaluating...")
    try:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
        metrics = calculate_metrics(model.model, val_loader, config, device)
        
        print(f"✓ Evaluation complete!")
        print(f"  - Perplexity: {metrics.perplexity:.4f}")
        print(f"  - Loss: {metrics.avg_loss:.4f}")
        print(f"  - Accuracy: {metrics.accuracy:.4%}")
        print(f"  - Tokens: {metrics.total_tokens}")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return 1
    
    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_metrics_to_json(metrics, str(output_path))
    print(f"\n💾 Metrics saved to: {output_path}")
    
    # Also save markdown report
    report_path = output_path.with_suffix(".md")
    export_metrics_to_markdown(metrics, str(report_path))
    print(f"📝 Report saved to: {report_path}")
    
    print("\n" + "="*70 + "\n")
    return 0


def generate_command(args):
    """Execute text generation"""
    print("\n" + "="*70)
    print("ArcLM Text Generation")
    print("="*70)
    
    # Load model
    try:
        print(f"\n📦 Loading model from: {args.model}")
        model = load_model(args.model, device=args.device)
        print("✓ Model loaded")
    except FileNotFoundError:
        print(f"❌ Model file not found: {args.model}")
        return 1
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    print(f"\n✍️ Prompt: {args.prompt}")
    print(f"📝 Generating {args.length} tokens (temperature={args.temperature})...")
    
    try:
        for i in range(args.num_samples):
            result = model.predict(
                args.prompt,
                max_new_tokens=args.length,
                temperature=args.temperature
            )
            
            if args.num_samples > 1:
                print(f"\n--- Sample {i+1} ---")
            print(result)
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return 1
    
    print("\n" + "="*70 + "\n")
    return 0


def run_command(args):
    """Execute package-level runtime helpers."""
    if args.run in {"simple-interface", "simple_interface"}:
        try:
            from arclm.simple_interface import run_simple_interface
        except ModuleNotFoundError as exc:
            if exc.name == "flask":
                print(
                    "The simple interface requires Flask. "
                    "Install it with: pip install arclm[web]"
                )
                return 1
            raise

        run_simple_interface(
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
        return 0

    print(f"Unknown runtime target: {args.run}")
    return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ArcLM - command-line tools for training and inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arclm train --config config.yaml --data data.txt --output models/model.pth
  arclm eval --model models/model.pth --data test.txt
  arclm generate --model models/model.pth --prompt "The future is"
  python -m arclm --run simple-interface
        """
    )
    
    parser.add_argument("--version", action="version", version="{} ArcLM".format(__import__("arclm").__version__))
    parser.add_argument(
        "--run",
        choices=["simple-interface", "simple_interface"],
        help="Run an ArcLM runtime helper such as the simple web interface",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host for --run simple-interface, defaults to HOST or 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for --run simple-interface, defaults to PORT or 5000",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=None,
        help="Enable Flask debug mode for --run simple-interface",
    )
    debug_group.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable Flask debug mode for --run simple-interface",
    )
    
    subparsers = parser.add_subparsers(title="commands", dest="command")
    
    create_train_parser(subparsers)
    create_eval_parser(subparsers)
    create_generate_parser(subparsers)
    
    args = parser.parse_args()

    if args.run:
        return run_command(args)
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
