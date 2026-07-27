"""
Configuration Module - Centralized settings
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

import torch

class Config:
    
    """
        Central configuration for ArcLM.

        All parameters can be overridden via keyword arguments.

        -------------------------
         Core
        -------------------------
        embed_dim: embedding size
        block_size: context length
        num_blocks: number of transformer layers
        dropout: regularization rate

        -------------------------
         Training
        -------------------------
        batch_size: samples per batch
        num_epochs: training iterations
        learning_rate: optimizer step size
        weight_decay: L2 regularization
        grad_clip: gradient limit (optional)
        device: "cpu" | "cuda"

        -------------------------
         Data
        -------------------------
        data_path: training data file
        domain_data_path: optional extra dataset
        validation_split: validation ratio

        -------------------------
         Tokenizer
        -------------------------
        tokenizer_type: selects tokenization strategy

            "word" (default)
                Simple word-level tokenizer.
                Fast and easy, but limited understanding.

            "sentencepiece"
                Subword tokenizer (recommended).
                Better generalization and handles unknown words.
                user_defined_symbols: list of special tokens to include in the tokenizer
                    example: ["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"]

        sentencepiece_model_type:
            "bpe" (default) or "unigram"

        -------------------------
         Finetuning
        -------------------------
        freeze_backbone: freeze transformer layers
        freeze_embedding: freeze embeddings
        use_lr_scheduler: enable LR scheduling

        -------------------------
         Example
        -------------------------
        >>> config = Config(
        ...     embed_dim=128,
        ...     tokenizer_type="sentencepiece",
        ...     device="cuda"
        ... )
    """

    
    def __init__(self, **kwargs):
        # Default values
        self.embed_dim = kwargs.get("embed_dim", 64)
        self.block_size = kwargs.get("block_size", 8)
        self.batch_size = kwargs.get("batch_size", 64)
        self.num_epochs = kwargs.get("num_epochs", 100)
        self.vocab_size = kwargs.get("vocab_size", None)  # Will be set after tokenizer is built
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.dropout = kwargs.get("dropout", 0.0)
        self.grad_clip = kwargs.get("grad_clip", None)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.model_path = kwargs.get("model_path", "output/model.pth")
        self.tokenizer_path = kwargs.get("tokenizer_path", "output/tokenizer.model")
        self.user_defined_symbols = kwargs.get("user_defined_symbols", ["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"])
        self.data_path = kwargs.get("data_path", "data/data.txt")
        self.domain_data_path = kwargs.get("domain_data_path", None)
        self.domain_data_repeats = kwargs.get("domain_data_repeats", 1)
        self.tokenizer_type = kwargs.get("tokenizer_type", "word")
        self.sentencepiece_model_type = kwargs.get("sentencepiece_model_type", "bpe")
        self.sentencepiece_character_coverage = kwargs.get(
            "sentencepiece_character_coverage",
            1.0,
        )
        self.tokenizer_max_line_length = kwargs.get("tokenizer_max_line_length", 4000)
        self.max_vocab = kwargs.get("max_vocab", 50000)
        self.max_data_size = kwargs.get("max_data_size", 1000000)
        self.device = kwargs.get("device", "cpu")
        self.validation_split = kwargs.get("validation_split", 0.0)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", None)
        self.early_stopping_min_delta = kwargs.get("early_stopping_min_delta", 0.0)
        self.restore_best_model = kwargs.get("restore_best_model", True)
        self.seed = kwargs.get("seed", 42)
        self.diagnostic_top_k = kwargs.get("diagnostic_top_k", 5)
        self.concept_benchmark_top_k = kwargs.get("concept_benchmark_top_k", 10)
        self.diagnostic_prompts = kwargs.get(
            "diagnostic_prompts",
            ["machine learning", "donald trump"],
        )
        self.diagnostic_sample_tokens = kwargs.get("diagnostic_sample_tokens", 60)
        self.tokenizer_rare_threshold = kwargs.get("tokenizer_rare_threshold", 2)
        self.training_log_interval = kwargs.get("training_log_interval", 50)
        self.metrics_log_path = kwargs.get("metrics_log_path", None)
        self.run_long_context_evaluation = kwargs.get("run_long_context_evaluation", False)
        self.use_checkpoint_tokenizer = kwargs.get("use_checkpoint_tokenizer", False)
        self.long_context_block_sizes = kwargs.get(
            "long_context_block_sizes",
            [32, 64, 128],
        )
        
        # ===================== FINETUNING PARAMETERS =====================
        self.freeze_backbone = kwargs.get("freeze_backbone", False)
        self.freeze_embedding = kwargs.get("freeze_embedding", False)
        self.use_discriminative_lr = kwargs.get("use_discriminative_lr", False)
        self.lr_multiplier = kwargs.get("lr_multiplier", None)
        self.use_lr_scheduler = kwargs.get("use_lr_scheduler", False)
        self.lr_scheduler_strategy = kwargs.get("lr_scheduler_strategy", "cosine")
        self.warmup_epochs = kwargs.get("warmup_epochs", 1)
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 0)
        self.checkpoint_batch_interval = kwargs.get("checkpoint_batch_interval", 0)
        self.checkpoint_callback = kwargs.get("checkpoint_callback", None)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "embed_dim": self.embed_dim,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "vocab_size": self.vocab_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "grad_clip": self.grad_clip,
            "num_blocks": self.num_blocks,
            "model_path": self.model_path,
            "data_path": self.data_path,
            "tokenizer_path": self.tokenizer_path,
            "user_defined_symbols": self.user_defined_symbols,
            "domain_data_path": self.domain_data_path,
            "domain_data_repeats": self.domain_data_repeats,
            "tokenizer_type": self.tokenizer_type,
            "sentencepiece_model_type": self.sentencepiece_model_type,
            "sentencepiece_character_coverage": self.sentencepiece_character_coverage,
            "tokenizer_max_line_length": self.tokenizer_max_line_length,
            "max_vocab": self.max_vocab,
            "max_data_size": self.max_data_size,
            "device": self.device,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "restore_best_model": self.restore_best_model,
            "seed": self.seed,
            "diagnostic_top_k": self.diagnostic_top_k,
            "concept_benchmark_top_k": self.concept_benchmark_top_k,
            "diagnostic_prompts": self.diagnostic_prompts,
            "diagnostic_sample_tokens": self.diagnostic_sample_tokens,
            "tokenizer_rare_threshold": self.tokenizer_rare_threshold,
            "training_log_interval": self.training_log_interval,
            "metrics_log_path": self.metrics_log_path,
            "run_long_context_evaluation": self.run_long_context_evaluation,
            "use_checkpoint_tokenizer": self.use_checkpoint_tokenizer,
            "long_context_block_sizes": self.long_context_block_sizes,
            "freeze_backbone": self.freeze_backbone,
            "freeze_embedding": self.freeze_embedding,
            "use_discriminative_lr": self.use_discriminative_lr,
            "lr_multiplier": self.lr_multiplier,
            "use_lr_scheduler": self.use_lr_scheduler,
            "lr_scheduler_strategy": self.lr_scheduler_strategy,
            "warmup_epochs": self.warmup_epochs,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_batch_interval": self.checkpoint_batch_interval,
        }


    def load_config_from_model(self, model_path):
        pass

    def get_device(self):
        return torch.device(self.device)
    
    def __repr__(self):
        items = "\n".join(f"  {k}: {v}" for k, v in self.to_dict().items())
        return f"Config(\n{items}\n)"

    def set_safe(self, **kwargs):
        """Set config attributes safely, ignoring unknown keys."""
        validators = {
            "vocab_size": lambda v: min(v, self.max_vocab)
            if isinstance(v, int) else v
        }

        for key, value in kwargs.items():
            if not hasattr(self, key):
                print(f"Warning: Unknown config attribute '{key}' ignored.")
                continue

            if key in validators:
                new_value = validators[key](value)
                if new_value != value:
                    print(
                        f"Warning: {key} {value} exceeds allowed limit. "
                        f"Using {new_value} instead."
                    )
                value = new_value

            setattr(self, key, value)




def create_config(**kwargs) -> Config:
    """
    Create a Config object with sensible defaults that can be overridden.

    Parameters
    ----------
    **kwargs
        Any configuration field supported by the Config class.

    Common Parameters
    -----------------
    embed_dim : int
        Transformer embedding dimension.
    num_blocks : int
        Number of transformer blocks.
    vocab_size : int
        Size of the tokenizer vocabulary.
    block_size : int
        Maximum context length in tokens.
    batch_size : int
        Training batch size.
    learning_rate : float
        Optimizer learning rate.
    dropout : float
        Dropout probability.
    max_vocab : int
        Maximum tokenizer vocabulary size.
    num_epochs : int
        Number of training epochs.
    tokenizer_path : str
        Path to save or load the tokenizer model.
    user_defined_symbols: list
        List of special tokens to include in the tokenizer.
    model_path : str
        Path to save or load the model checkpoint.
    device : str
        Device to run training on ("cpu" or "cuda").

    Returns
    -------
    Config
        Initialized configuration object.

    Examples
    --------
    Create a default configuration:

    >>> cfg = create_config()

    Create a larger model:

    >>> cfg = create_config(
    ...     embed_dim=256,
    ...     num_blocks=6,
    ...     block_size=256
    ... )

    Customize training settings:

    >>> cfg = create_config(
    ...     batch_size=64,
    ...     learning_rate=1e-4,
    ...     num_epochs=20
    ... )

    Configure tokenizer settings:

    >>> cfg = create_config(
    ...     tokenizer_type="sentencepiece",
    ...     max_vocab=16000
    ... )

    Notes
    -----
    - Any keyword argument matching a Config field will override
      the default value.
    - Unknown arguments are forwarded directly to Config.
    - The device is automatically selected unless explicitly specified.
    - For multi-head attention models, embed_dim should generally be
      divisible by the number of attention heads.
    """
    defaults = {
        "embed_dim": 64,
        "block_size": 8,
        "batch_size": 64,
        "vocab_size": None,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "grad_clip": None,
        "num_blocks": 2,

        "model_path": "output/model.pth",
        "data_path": "data/data.txt",
        "tokenizer_path": "output/tokenizer.model",
        "user_defined_symbols": ["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"],

        "domain_data_path": None,
        "domain_data_repeats": 1,

        "tokenizer_type": "word",
        "sentencepiece_model_type": "bpe",
        "sentencepiece_character_coverage": 1.0,
        "tokenizer_max_line_length": 4000,
        "max_vocab": 50000,
        "max_data_size": 1_000_000,
        "tokenizer_rare_threshold": 2,

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "validation_split": 0.0,
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
        "restore_best_model": True,

        "seed": 42,

        "diagnostic_top_k": 5,
        "concept_benchmark_top_k": 10,
        "diagnostic_prompts": [
            "machine learning",
            "donald trump",
        ],
        "diagnostic_sample_tokens": 60,

        "training_log_interval": 50,
        "metrics_log_path": None,

        "run_long_context_evaluation": False,
        "use_checkpoint_tokenizer": False,
        "long_context_block_sizes": [32, 64, 128],
        
        # Finetuning parameters (added in Phase 1)
        "freeze_backbone": False,
        "freeze_embedding": False,
        "use_discriminative_lr": False,
        "lr_multiplier": None,
        "use_lr_scheduler": False,
        "lr_scheduler_strategy": "cosine",
        "warmup_epochs": 1,
        "checkpoint_interval": 0,
        "checkpoint_batch_interval": 0,
        "checkpoint_callback": None,
    }

    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown configuration parameters: {', '.join(sorted(unknown))}"
        )

    defaults.update(kwargs)

    torch.manual_seed(defaults.get("seed"))

    return Config(**defaults)


def get_finetuning_config(
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    freeze_backbone: bool = True,
    use_discriminative_lr: bool = True,
    tokenizer_type: str = "sentencepiece",
    vocab_size: int = None,
    user_defined_symbols= ["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"],
    **kwargs
) -> Config:
    """Get config optimized for finetuning pretrained models.
    
    Typical finetuning settings:
    - Small number of epochs (3-5)
    - Small learning rate (1e-5 to 5e-5)
    - Freeze backbone
    - Use discriminative learning rates
    
    Args:
        num_epochs: Number of training epochs (default 3)
        batch_size: Batch size (default 32)
        learning_rate: Base learning rate (default 2e-5)
        freeze_backbone: Whether to freeze transformer blocks (default True)
        use_discriminative_lr: Use different LRs for different layers (default True)
        user_defined_symbols: List of special tokens to include in the tokenizer (default ["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"])
        **kwargs: Additional config parameters
    
    Returns:
        Config object optimized for finetuning
    
    Examples:
        >>> config = get_finetuning_config()
        >>> config = get_finetuning_config(num_epochs=5, learning_rate=5e-5)
    """
    config = create_config(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        freeze_backbone=freeze_backbone,
        use_discriminative_lr=use_discriminative_lr,
        user_defined_symbols = user_defined_symbols,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        restore_best_model=True,
        lr_multiplier={
            'embeddings': 0.1,
            'blocks': 0.1,
            'head': 1.0
        } if use_discriminative_lr else None,
        **kwargs
    )
    return config


def get_instruction_tuning_config(
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    vocab_size: int = None,
    user_defined_symbols = ["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"],
    **kwargs
) -> Config:
    """Get config optimized for instruction tuning.
    
    Builds on finetuning config with additional optimizations for
    instruction-following model training.
    
    Args:
        num_epochs: Number of training epochs (default 5)
        batch_size: Batch size (default 16, smaller for smaller batch diversity)
        learning_rate: Base learning rate (default 5e-5)
        user_defined_symbols: List of special tokens to include in the tokenizer (default ["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"])
        **kwargs: Additional config parameters
    
    Returns:
        Config object optimized for instruction tuning
    
    Examples:
        >>> config = get_instruction_tuning_config()
        >>> config = get_instruction_tuning_config(num_epochs=3, batch_size=8)
    """
    config = get_finetuning_config(
        num_epochs=num_epochs,
        batch_size=batch_size,
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        user_defined_symbols = user_defined_symbols,
        freeze_backbone=True,
        use_discriminative_lr=True,
        **kwargs
    )
    return config
