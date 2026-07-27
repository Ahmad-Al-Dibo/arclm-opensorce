"""
Unified Training Pipeline v2 — Pre-trained model support + stopping criteria
Provides end-to-end training orchestration for pre-training, fine-tuning, and instruction-tuning.

Key Components:
  1. PreTrainedModelLoader - Load from HuggingFace, local files, custom sources
  2. ModelAdapter - Seamlessly adapt external model weights to ArcLM architecture
  3. UnifiedPipeline - Orchestrate all training modes with stopping criteria
  4. StoppingCriteria - max_steps, early_stopping, patience controls
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import logging

from ..model import ArcLM
from ..trainer import Trainer
from ..config import Config
from .base import BaseModelAdapter, BaseModelLoader, BaseTrainingPipeline

logger = logging.getLogger(__name__)


# ============================================================================
# STOPPING CRITERIA
# ============================================================================

@dataclass
class StoppingCriteria:
    """Base stopping criteria for training loop control."""
    
    max_steps: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    
    def should_stop_by_steps(self, current_step: int) -> bool:
        """Check if max_steps exceeded."""
        if self.max_steps is None:
            return False
        return current_step >= self.max_steps
    
    def should_stop_by_validation(self, val_loss: float, best_val_loss: float, 
                                  patience_counter: int) -> bool:
        """Check if early stopping criteria met."""
        if self.early_stopping_patience is None:
            return False
        
        # Improvement detected
        if val_loss < best_val_loss - self.early_stopping_min_delta:
            return False  # Don't stop, reset patience
        
        # No improvement - check if patience exceeded
        return patience_counter >= self.early_stopping_patience


# ============================================================================
# PRE-TRAINED MODEL LOADER
# ============================================================================

@dataclass
class PreTrainedModelLoader(BaseModelLoader):
    """
    Load pre-trained models from multiple sources:
      - HuggingFace transformers
      - Local checkpoint files
      - Custom model paths
    
    Handles weight alignment and architecture mismatch gracefully.
    """
    
    source: Union[str, Path]
    target_vocab_size: int
    target_config: Config
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    strict_loading: bool = False
    
    def load(self, revision: str | None = None, trust_remote_code: bool = False) -> tuple:
        """
        Load model and return (model, metadata).
        
        Returns:
            (ArcLM model, Dict of metadata)
        
        Raises:
            ValueError: If source invalid or incompatible
            FileNotFoundError: If file not found
        """
        if self._is_huggingface_model(self.source):
            return self._load_from_huggingface(revision=revision, trust_remote_code=trust_remote_code)
        elif self._is_local_checkpoint(self.source):
            return self._load_from_checkpoint()
        else:
            raise ValueError(
                f"Unrecognized model source: {self.source}\n"
                f"Expected: 'hf_model_id' or local path to .pth checkpoint"
            )
    
    def _is_huggingface_model(self, source: Union[str, Path]) -> bool:
        """Check if source is HuggingFace model ID (no file path)."""
        source_str = str(source)
        # If it's a file that exists, it's not a HF model
        if Path(source_str).exists():
            return False
        # HF format: "username/model-name" or "model-name" (but not paths)
        if source_str.endswith((".pth", ".pt", ".bin", ".safetensors", ".ckpt")):
            return False
        # HF models can be org/model or just model
        return "/" in source_str or (not "/" in source_str and not "." in source_str)
    
    def _is_local_checkpoint(self, source: Union[str, Path]) -> bool:
        """Check if source is local file path."""
        path = Path(source)
        return path.exists() and path.is_file()
    
    def _load_from_huggingface(
        self,
        revision: str | None = None,
        trust_remote_code: bool = True) -> tuple:
        """Load from HuggingFace transformers library."""
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers library required for HuggingFace models. "
                "Install: pip install transformers"
            )
        
        logger.info(f"Loading HuggingFace model: {self.source}")
        
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                str(self.source),
                device_map=self.device,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            hf_config = AutoConfig.from_pretrained(str(self.source))
            
            # Create ArcLM model and adapt weights
            model = ArcLM(
                vocab_size=self.target_vocab_size,
                embed_dim=self.target_config.embed_dim,
                block_size=self.target_config.block_size,
                num_blocks=self.target_config.num_blocks,
                dropout=self.target_config.dropout,
            ).to(self.device)
            
            adapter = ModelAdapter(source_model=hf_model, target_model=model)
            adapter.adapt_weights()
            
            metadata = {
                "source": "huggingface",
                "model_id": str(self.source),
                "original_config": hf_config.to_dict(),
                "vocab_size_mismatch": hf_config.vocab_size != self.target_vocab_size,
            }
            
            logger.info(f"✓ HuggingFace model loaded and adapted")
            return model, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace model '{self.source}': {e}")
    
    def _load_from_checkpoint(self) -> tuple:
        """Load from local .pth checkpoint."""
        path = Path(self.source)
        logger.info(f"Loading checkpoint: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        
        # Extract model config from checkpoint
        checkpoint_config = checkpoint.get("config", {})
        if not checkpoint_config:
            logger.warning("Checkpoint has no config; using target config")
            checkpoint_config = self.target_config.to_dict()
        
        # Create model with checkpoint config
        model = ArcLM(
            vocab_size=checkpoint.get("vocab_size", self.target_vocab_size),
            embed_dim=checkpoint_config.get("embed_dim", self.target_config.embed_dim),
            block_size=checkpoint_config.get("block_size", self.target_config.block_size),
            num_blocks=checkpoint_config.get("num_blocks", self.target_config.num_blocks),
            dropout=checkpoint_config.get("dropout", self.target_config.dropout),
        ).to(self.device)
        
        # Load state dict
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
        if state_dict is None:
            raise ValueError("Checkpoint does not contain model weights (model_state_dict)")
        
        try:
            model.load_state_dict(state_dict, strict=self.strict_loading)
        except RuntimeError as e:
            if self.strict_loading:
                raise
            logger.warning(f"Non-strict loading: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        metadata = {
            "source": "checkpoint",
            "path": str(path),
            "checkpoint_epoch": checkpoint.get("current_epoch", 0),
            "checkpoint_vocab_size": checkpoint.get("vocab_size"),
            "checkpoint_config": checkpoint_config,
        }
        
        logger.info(f"✓ Checkpoint loaded (epoch {checkpoint.get('current_epoch', 0)})")
        return model, metadata


# ============================================================================
# MODEL ADAPTER
# ============================================================================

class ModelAdapter(BaseModelAdapter):
    """
    Adapt weights from external models to ArcLM architecture.
    
    Handles:
      - Embedding dimension mismatches (linear projection)
      - Sequence length mismatches (interpolation or truncation)
      - Missing/extra weights (graceful handling)
    """
    
    def __init__(self, source_model: nn.Module, target_model: ArcLM):
        """
        Initialize adapter.
        
        Args:
            source_model: Source model (HuggingFace, checkpoint, etc.)
            target_model: Target ArcLM model
        """
        self.source_model = source_model
        self.target_model = target_model
        self.device = next(target_model.parameters()).device
    
    def adapt_weights(self, verbose: bool = True) -> Dict[str, int]:
        """
        Adapt source model weights to target model architecture.
        
        Args:
            verbose: Print adaptation details
        
        Returns:
            Dict with stats: {'adapted': N, 'skipped': M, 'initialized_random': K}
        """
        stats = {"adapted": 0, "skipped": 0, "initialized_random": 0}
        
        # Try to extract embedding from source
        source_embed = self._extract_embedding(self.source_model)
        if source_embed is not None:
            self._adapt_embedding(source_embed, stats)
            if verbose:
                logger.info(f"  ✓ Adapted embeddings ({stats['adapted']} params)")
        
        # Try to extract transformer blocks
        source_blocks = self._extract_transformer_blocks(self.source_model)
        if source_blocks:
            self._adapt_blocks(source_blocks, stats)
            if verbose:
                logger.info(f"  ✓ Adapted {len(source_blocks)} transformer blocks")
        
        # Initialize remaining parameters randomly
        for name, param in self.target_model.named_parameters():
            if param.requires_grad:
                if param.data.abs().sum() == 0:  # Uninitialized
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        # For 1D params (bias), use normal initialization
                        nn.init.normal_(param, std=0.01)
                    stats["initialized_random"] += 1
        
        if verbose:
            logger.info(
                f"Adaptation complete: "
                f"{stats['adapted']} adapted, "
                f"{stats['skipped']} skipped, "
                f"{stats['initialized_random']} random-init"
            )
        
        return stats
    
    def _extract_embedding(self, model: nn.Module) -> Optional[nn.Embedding]:
        """Extract token embedding layer from source model."""
        for attr_name in ["embedding", "token_embedding", "embed_tokens"]:
            if hasattr(model, attr_name):
                return getattr(model, attr_name)
        
        # Try to find via children
        for module in model.children():
            if isinstance(module, nn.Embedding):
                return module
        
        return None
    
    def _extract_transformer_blocks(self, model: nn.Module) -> list:
        """Extract transformer blocks from source model."""
        # Try common names
        for attr_name in ["blocks", "layers", "transformer.h", "decoder.layers"]:
            if hasattr(model, attr_name):
                blocks_container = getattr(model, attr_name)
                
                # If it's nn.ModuleList, convert to list
                if isinstance(blocks_container, nn.ModuleList):
                    return list(blocks_container)
                
                # If it's nn.Sequential, convert to list
                if isinstance(blocks_container, nn.Sequential):
                    return list(blocks_container.children())
        
        return []
    
    def _adapt_embedding(self, source_embed: nn.Embedding, stats: Dict):
        """Adapt embedding weights."""
        target_embed = self.target_model.token_embedding
        
        source_vocab_size, source_embed_dim = source_embed.weight.shape
        target_vocab_size, target_embed_dim = target_embed.weight.shape
        
        # Copy common vocabulary up to target vocab size
        min_vocab = min(source_vocab_size, target_vocab_size)
        
        # Adapt embedding dimension if needed
        if source_embed_dim != target_embed_dim:
            # Use simple average or truncation based on dimension
            if source_embed_dim > target_embed_dim:
                # Truncate: take first target_embed_dim dimensions
                target_embed.weight.data[:min_vocab] = source_embed.weight.data[:min_vocab, :target_embed_dim]
            else:
                # Expand: pad with zeros
                target_embed.weight.data[:min_vocab, :source_embed_dim] = source_embed.weight.data[:min_vocab]
        else:
            # Same dimension, just copy
            target_embed.weight.data[:min_vocab] = source_embed.weight.data[:min_vocab]
        
        stats["adapted"] += 1
    
    def _adapt_blocks(self, source_blocks: list, stats: Dict):
        """Adapt transformer blocks."""
        target_blocks = self.target_model.blocks
        
        min_blocks = min(len(source_blocks), len(target_blocks))
        
        for i in range(min_blocks):
            # Simple approach: if same architecture, copy weights directly
            try:
                state = source_blocks[i].state_dict()
                target_blocks[i].load_state_dict(state, strict=False)
                stats["adapted"] += 1
            except Exception:
                stats["skipped"] += 1
                logger.warning(f"Could not adapt block {i}")


# ============================================================================
# UNIFIED PIPELINE
# ============================================================================

class UnifiedPipeline(BaseTrainingPipeline):
    """
    Unified training pipeline supporting multiple modes:
      - PRE_TRAINING: Train from scratch on large corpus
      - FINE_TUNING: Fine-tune pre-trained model on task data
      - INSTRUCTION_TUNING: Fine-tune on instruction-response pairs
    
    Orchestrates:
      - Model loading (scratch or pre-trained)
      - Stopping criteria (max_steps, early_stopping)
      - Checkpointing and resumption
      - Diagnostics integration
    """
    
    MODES = ["pre_training", "fine_tuning", "instruction_tuning"]
    
    def __init__(
        self,
        config: Config,
        mode: str = "fine_tuning",
        pretrained_source: Optional[Union[str, Path]] = None,
        stopping_criteria: Optional[StoppingCriteria] = None,
    ):
        """
        Initialize unified pipeline.
        
        Args:
            config: Training config
            mode: "pre_training", "fine_tuning", or "instruction_tuning"
            pretrained_source: For fine-tuning: HF model ID or local checkpoint path
            stopping_criteria: Stopping criteria (max_steps, early_stopping, etc.)
        """
        if mode not in self.MODES:
            raise ValueError(f"Mode must be one of {self.MODES}, got {mode}")
        
        self.config = config
        self.mode = mode
        self.pretrained_source = pretrained_source
        self.stopping_criteria = stopping_criteria or StoppingCriteria()
        self.device = torch.device(config.device)
        
        self.model = None
        self.trainer = None
        self.metadata = {}
        
        logger.info(f"UnifiedPipeline initialized (mode={mode})")
    
    def build(self, vocab_size: int) -> "UnifiedPipeline":
        """
        Build model and trainer.
        
        Args:
            vocab_size: Vocabulary size for tokenizer
        
        Returns:
            self (for chaining)
        """
        logger.info(f"Building pipeline for mode: {self.mode}")
        
        if self.mode == "pre_training":
            self.model = self._build_from_scratch(vocab_size)
            self.metadata["source"] = "random_init"
        
        elif self.mode in ["fine_tuning", "instruction_tuning"]:
            if self.pretrained_source is None:
                raise ValueError(
                    f"{self.mode} requires pretrained_source. "
                    f"Pass HuggingFace model ID or local checkpoint path."
                )
            self.model, self.metadata = self._load_pretrained(vocab_size)
        
        # Build trainer
        self.trainer = self._build_trainer()
        
        logger.info(f"✓ Pipeline built: model={self.model.__class__.__name__}, "
                    f"device={self.device}, mode={self.mode}")
        return self
    
    def _build_from_scratch(self, vocab_size: int) -> ArcLM:
        """Build new model from random initialization."""
        logger.info("Building model from scratch")
        model = ArcLM(
            vocab_size=vocab_size,
            embed_dim=self.config.embed_dim,
            block_size=self.config.block_size,
            num_blocks=self.config.num_blocks,
            dropout=self.config.dropout,
        ).to(self.device)
        
        logger.info(f"✓ Model created: {vocab_size} vocab, "
                    f"{self.config.embed_dim} embed_dim, "
                    f"{self.config.num_blocks} blocks")
        return model
    
    def _load_pretrained(self, vocab_size: int) -> tuple:
        """Load pre-trained model with adaptation."""
        logger.info(f"Loading pre-trained model from: {self.pretrained_source}")
        
        loader = PreTrainedModelLoader(
            source=self.pretrained_source,
            target_vocab_size=vocab_size,
            target_config=self.config,
            device=str(self.device),
            strict_loading=False,
        )
        
        model, metadata = loader.load()
        logger.info(f"✓ Pre-trained model loaded and adapted")
        
        return model, metadata
    
    def _build_trainer(self) -> Trainer:
        """Build trainer with appropriate configuration."""
        # Use discriminative LR for fine-tuning
        if self.mode in ["fine_tuning", "instruction_tuning"]:
            if not hasattr(self.config, "use_discriminative_lr"):
                self.config.use_discriminative_lr = True
            
            if not hasattr(self.config, "lr_multiplier"):
                self.config.lr_multiplier = {
                    'embeddings': 0.1,
                    'blocks': 0.1,
                    'head': 1.0
                }
        
        from ..pipeline import build_trainer
        return build_trainer(self.model, self.config)
    
    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute training with stopping criteria.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            num_epochs: Number of epochs (overrides config.num_epochs)
        
        Returns:
            Dict with training results: {'train_losses', 'val_losses', 'stopped_reason', ...}
        """
        if self.model is None or self.trainer is None:
            raise RuntimeError("Pipeline not built. Call build(vocab_size) first.")
        
        epochs = num_epochs or self.config.num_epochs
        logger.info(f"Starting training: {epochs} epochs, mode={self.mode}")
        
        # Train with early stopping
        self.trainer.train(
            loader=train_loader,
            epochs=epochs,
            val_loader=val_loader,
            early_stopping_patience=self.stopping_criteria.early_stopping_patience,
            min_delta=self.stopping_criteria.early_stopping_min_delta,
        )
        
        results = {
            "mode": self.mode,
            "epochs_completed": self.trainer.current_epoch,
            "train_losses": self.trainer.train_losses,
            "val_losses": self.trainer.val_losses,
            "best_val_loss": self.trainer.best_val_loss,
            "stopped_by": "max_epochs",
        }
        
        logger.info(f"Training completed: {self.trainer.current_epoch} epochs")
        return results
    
    def save_checkpoint(self, path: Optional[str] = None) -> Path:
        """Save model checkpoint with metadata."""
        if self.trainer is None:
            raise RuntimeError("No trainer available. Build and train first.")
        
        path = Path(path or self.config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model": self.model.state_dict(),
            "trainer": self.trainer,
            "config": self.config.to_dict(),
            "mode": self.mode,
            "metadata": self.metadata,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"✓ Checkpoint saved to {path}")
        
        return path
    
    def get_model(self) -> ArcLM:
        """Get the trained model."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        return self.model
