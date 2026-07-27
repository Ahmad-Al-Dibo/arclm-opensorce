"""
Trainer Module - Training loop and model management
"""

import time
import copy
import inspect
import math
import os
import tempfile
import warnings
import torch
import torch.nn.functional as F
from pathlib import Path
from .config import Config
from .logging import AsyncTrainingLogger

from .logics import (
    Symbol,
    And,
    Or,
    Not,
    Implication,
    Biconditional,
    model_check
)


class Trainer:
    """Model trainer with save/load functionality and generalization monitoring"""
    
    def __init__(self, model, optimizer, criterion, config, event_logger=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = torch.device(config.device)
        self.event_logger = event_logger
        self._owns_event_logger = False
        if self.event_logger is None and getattr(config, "metrics_log_path", None):
            self.event_logger = AsyncTrainingLogger(config.metrics_log_path).start()
            self._owns_event_logger = True
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []
        self.val_token_accuracies = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state_dict = None
        self.current_epoch = 0
        self.current_batch = 0
        self.global_step = 0
    
    def train(
        self,
        loader,
        epochs,
        val_loader=None,
        early_stopping_patience=None,
        min_delta=None,
        checkpoint_callback=None,
        checkpoint_epoch_interval=None,
        checkpoint_batch_interval=None,
    ):
        """Train the model with optional validation and early stopping
        
        Args:
            loader: Training DataLoader
            epochs: Total target number of completed epochs
            val_loader: Validation DataLoader (optional)
            early_stopping_patience: Stop if val loss doesn't improve (optional)
            min_delta: Minimum validation loss improvement (optional)
            checkpoint_callback: Optional callable used for checkpoint saves
            checkpoint_epoch_interval: Save every N completed epochs (optional)
            checkpoint_batch_interval: Save every N training batches (optional)
        """
        if loader is None:
            warnings.warn(
                "No training loader was provided; training cannot start.",
                RuntimeWarning,
                stacklevel=2,
            )
            raise ValueError("loader is required for training.")

        if epochs is None:
            warnings.warn(
                "No epoch count was provided. Pass epochs or config.num_epochs.",
                RuntimeWarning,
                stacklevel=2,
            )
            raise ValueError("epochs is required for training.")

        if epochs <= 0:
            warnings.warn(
                f"epochs must be greater than zero; got {epochs}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        try:
            loader_batches = len(loader)
        except TypeError:
            loader_batches = None

        if loader_batches == 0:
            warnings.warn(
                "The training loader has no batches; nothing will be trained.",
                RuntimeWarning,
                stacklevel=2,
            )
            return


        checkpoint_epoch_interval = self._resolve_checkpoint_interval( 
            checkpoint_epoch_interval,
            getattr(self.config, "checkpoint_interval", None),
        )
        checkpoint_batch_interval = self._resolve_checkpoint_interval(
            checkpoint_batch_interval,
            getattr(self.config, "checkpoint_batch_interval", None),
        )

        if checkpoint_callback is None:
            checkpoint_callback = getattr(self.config, "checkpoint_callback", None)

        if (
            checkpoint_callback is not None
            and checkpoint_epoch_interval == 0
            and checkpoint_batch_interval == 0
        ):
            checkpoint_epoch_interval = 1

        if checkpoint_callback is None and (
            checkpoint_epoch_interval > 0 or checkpoint_batch_interval > 0
        ):
            checkpoint_callback = self._default_checkpoint_callback

        if early_stopping_patience is not None and val_loader is None:
            warnings.warn(
                "early_stopping_patience was set, but no val_loader was provided; "
                "early stopping will be ignored.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.current_epoch >= epochs:
            print(
                f"\n[OK] Training already complete "
                f"({self.current_epoch}/{epochs} epochs)."
            )
            if val_loader is not None and getattr(self.config, "restore_best_model", True):
                self.restore_best_model()
            if self._owns_event_logger and self.event_logger is not None:
                self.event_logger.close()
                self.event_logger = None
            return

        self.model.train()
        train_start = time.time()
        if self.event_logger is not None:
            self.event_logger.info(
                "training_started",
                epochs=epochs,
                current_epoch=self.current_epoch,
                device=str(self.device),
            )
        if min_delta is None:
            min_delta = getattr(self.config, "early_stopping_min_delta", 0.0)
        
        if self.current_epoch > 0:
            print(f"\n[OK] Resuming training from epoch {self.current_epoch + 1}/{epochs}.")

        for epoch in range(self.current_epoch, epochs):
            epoch_start = time.time()
            total_loss = 0
            log_interval = getattr(self.config, "training_log_interval", 50)
            total_batches = loader_batches if loader_batches is not None else "?"
            batches_seen = 0
            
            for batch_idx, batch in enumerate(loader, start=1):
                batches_seen = batch_idx
                x, y, loss_mask = self.unpack_batch(batch)
                
                logits = self.model(x)
                loss = self.compute_loss(logits, y, loss_mask)
                
                self.optimizer.zero_grad()
                loss.backward()
                grad_clip = getattr(self.config, "grad_clip", None)
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                total_loss += loss.item()
                self.current_batch = batch_idx
                self.global_step += 1

                if (
                    checkpoint_batch_interval > 0
                    and checkpoint_callback is not None
                    and self.global_step % checkpoint_batch_interval == 0
                ):
                    self._call_checkpoint_callback(
                        checkpoint_callback,
                        event="batch_checkpoint",
                        epoch=epoch + 1,
                        batch=batch_idx,
                        total_batches=total_batches,
                        global_step=self.global_step,
                    )

                if log_interval and batch_idx % log_interval == 0:
                    elapsed = time.time() - epoch_start
                    batches_per_second = batch_idx / elapsed if elapsed else 0.0
                    remaining_batches = (
                        total_batches - batch_idx
                        if isinstance(total_batches, int)
                        else 0
                    )
                    eta_seconds = (
                        remaining_batches / batches_per_second
                        if batches_per_second
                        else 0.0
                    )
                    print(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx}/{total_batches} | "
                        f"Loss = {loss.item():.4f} | "
                        f"Elapsed = {elapsed:.1f}s | "
                        f"ETA = {eta_seconds:.1f}s",
                        flush=True,
                    )
                    if self.event_logger is not None:
                        self.event_logger.metric(
                            "batch_completed",
                            epoch=epoch + 1,
                            batch=batch_idx,
                            total_batches=total_batches,
                            loss=float(loss.item()),
                            elapsed_seconds=elapsed,
                            eta_seconds=eta_seconds,
                        )
                    # try:
                    #     if checkpoint_callback is not None:
                    #         checkpoint_callback(self)
                    # except Exception as e:
                    #     pass # in the feaurue i want to add logging functions

            
            # Average training loss
            if batches_seen == 0:
                warnings.warn(
                    "The training loader yielded no batches; stopping training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            avg_train_loss = total_loss / batches_seen
            self.train_losses.append(avg_train_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Validation
            val_loss = None
            should_stop = False
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                val_loss = val_metrics["loss"]
                self.val_losses.append(val_loss)
                self.val_perplexities.append(val_metrics["perplexity"])
                self.val_token_accuracies.append(val_metrics["token_accuracy"])
                
                # Berechne generalisatie gap
                gen_gap = val_loss - avg_train_loss
                
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss = {avg_train_loss:.4f} | "
                    f"Val Loss = {val_loss:.4f} | "
                    f"Val PPL = {val_metrics['perplexity']:.2f} | "
                    f"Val Acc = {val_metrics['token_accuracy']*100:.2f}% | "
                    f"Gen Gap = {gen_gap:.4f} | "
                    f"Time = {epoch_time:.2f}s"
                )
                if self.event_logger is not None:
                    self.event_logger.metric(
                        "epoch_completed",
                        epoch=epoch + 1,
                        train_loss=float(avg_train_loss),
                        val_loss=float(val_loss),
                        val_perplexity=float(val_metrics["perplexity"]),
                        val_token_accuracy=float(val_metrics["token_accuracy"]),
                        generalization_gap=float(gen_gap),
                        epoch_seconds=epoch_time,
                    )
                
                # Early stopping
                if early_stopping_patience is not None:
                    if self._check_early_stopping(val_loss, early_stopping_patience, min_delta):
                        print(f"\n[WARNING]  Early stopping at epoch {epoch+1}")
                        print(f"   Best validation loss: {self.best_val_loss:.4f}")
                        should_stop = True
                else:
                    self._update_best_validation(val_loss, min_delta)
            else:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss = {avg_train_loss:.4f} | "
                    f"Time = {epoch_time:.2f}s"
                )
                if self.event_logger is not None:
                    self.event_logger.metric(
                        "epoch_completed",
                        epoch=epoch + 1,
                        train_loss=float(avg_train_loss),
                        epoch_seconds=epoch_time,
                    )

            self.current_epoch = epoch + 1
            if (
                checkpoint_epoch_interval > 0
                and checkpoint_callback is not None
                and self.current_epoch % checkpoint_epoch_interval == 0
            ):
                self._call_checkpoint_callback(
                    checkpoint_callback,
                    event="epoch_checkpoint",
                    epoch=self.current_epoch,
                    batch=self.current_batch,
                    total_batches=total_batches,
                    global_step=self.global_step,
                )

            if should_stop:
                break
        
        total_train_time = time.time() - train_start
        hours = int(total_train_time // 3600)
        minutes = int((total_train_time % 3600) // 60)
        seconds = int(total_train_time % 60)
        if val_loader is not None and getattr(self.config, "restore_best_model", True):
            self.restore_best_model()
        print(f"\n[OK] Training completed in {hours}h {minutes}m {seconds}s")
        if self.event_logger is not None:
            self.event_logger.info(
                "training_completed",
                epochs_completed=self.current_epoch,
                total_seconds=total_train_time,
            )
        if self._owns_event_logger and self.event_logger is not None:
            self.event_logger.close()
            self.event_logger = None
    
    def _validate(self, val_loader):
        """Bereken validation loss, perplexity, en token accuracy."""
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        log_interval = getattr(self.config, "training_log_interval", 50)
        total_batches = len(val_loader)
        validation_start = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, start=1):
                x, y, loss_mask = self.unpack_batch(batch)
                
                logits = self.model(x)
                loss = self.compute_loss(logits, y, loss_mask)
                
                total_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                if loss_mask is not None:
                    active = loss_mask.bool()
                    correct_tokens += ((predictions == y) & active).sum().item()
                    total_tokens += int(active.sum().item())
                else:
                    correct_tokens += (predictions == y).sum().item()
                    total_tokens += y.numel()

                if log_interval and batch_idx % log_interval == 0:
                    elapsed = time.time() - validation_start
                    batches_per_second = batch_idx / elapsed if elapsed else 0.0
                    remaining_batches = total_batches - batch_idx
                    eta_seconds = (
                        remaining_batches / batches_per_second
                        if batches_per_second
                        else 0.0
                    )
                    print(
                        f"Validation | "
                        f"Batch {batch_idx}/{total_batches} | "
                        f"Elapsed = {elapsed:.1f}s | "
                        f"ETA = {eta_seconds:.1f}s",
                        flush=True,
                    )
         
        self.model.train()
        avg_loss = total_loss / len(val_loader)
        return {
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20)),
            "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        }

    def unpack_batch(self, batch):
        """Normalize tuple and dict dataloader batches."""
        if isinstance(batch, dict):
            try:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
            except KeyError as exc:
                raise ValueError("Dictionary batches must contain 'x' and 'y'.") from exc
            loss_mask = batch.get("mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)
            return x, y, loss_mask

        x, y = batch
        return x.to(self.device), y.to(self.device), None

    def compute_loss(self, logits, y, loss_mask=None):
        """Compute next-token loss, optionally over masked label positions only."""
        B, T, V = logits.shape
        if loss_mask is None:
            return self.criterion(
                logits.reshape(B * T, V),
                y.reshape(B * T),
            )

        if tuple(loss_mask.shape) != tuple(y.shape):
            raise ValueError(
                "Loss mask shape must match labels: "
                f"mask={tuple(loss_mask.shape)}, labels={tuple(y.shape)}."
            )

        loss_per_token = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            reduction="none",
        ).reshape(B, T)
        loss_mask = loss_mask.to(dtype=loss_per_token.dtype)
        return (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    def _unpack_batch(self, batch):
        """Backward-compatible alias for unpack_batch()."""
        return self.unpack_batch(batch)

    def _compute_loss(self, logits, y, loss_mask=None):
        """Backward-compatible alias for compute_loss()."""
        return self.compute_loss(logits, y, loss_mask)
    
    def _check_early_stopping(self, val_loss, patience, min_delta=0.0):
        """Check if we should stop early based on validation loss
        
        Generalisatie eigenschap: Stop training als model niet meer verbetert
        """
        if self._update_best_validation(val_loss, min_delta):
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
        return False

    def _update_best_validation(self, val_loss, min_delta=0.0):
        """Track best validation loss and copy current weights."""
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            return True
        return False

    def _resolve_checkpoint_interval(self, explicit_value, config_value):
        """Normalize optional checkpoint intervals to non-negative integers."""
        value = explicit_value if explicit_value is not None else config_value
        if value in (None, False):
            return 0
        try:
            value = int(value)
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid checkpoint interval {value!r}; automatic checkpointing is disabled.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0
        if value < 0:
            warnings.warn(
                f"Checkpoint interval must be >= 0; got {value}. Disabling it.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0
        return value

    def _default_checkpoint_callback(self, trainer, **_context):
        """Save a bare trainer checkpoint when no custom callback is provided."""
        trainer.save(trainer.config)

    def _call_checkpoint_callback(self, checkpoint_callback, **context):
        """Call checkpoint callbacks while keeping old one-argument callbacks valid."""
        try:
            signature = inspect.signature(checkpoint_callback)
        except (TypeError, ValueError):
            checkpoint_callback(self, **context)
            return

        parameters = list(signature.parameters.values())
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        positional_params = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if accepts_kwargs:
            checkpoint_callback(self, **context)
        elif len(positional_params) >= 2:
            checkpoint_callback(self, context)
        else:
            checkpoint_callback(self)

    def restore_best_model(self):
        """Restore weights from the best validation loss, if available."""
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            print(f"[OK] Restored best validation model ({self.best_val_loss:.4f})")
    
    def get_generalization_gap(self):
        """Get generalization gap (val_loss - train_loss)"""
        if not self.train_losses or not self.val_losses:
            return None
        return self.val_losses[-1] - self.train_losses[-1]
    
    # ===================== FINETUNING SUPPORT =====================
    
    def freeze_layers(self, pattern: str = "blocks", verbose: bool = True) -> int:
        """Freeze layers matching pattern for finetuning.
        
        Args:
            pattern: Layer name pattern to freeze (e.g., "blocks", "token_embedding")
            verbose: Print frozen layer info
        
        Returns:
            Number of frozen parameters
        
        Examples:
            trainer.freeze_layers("blocks")  # Freeze all transformer blocks
            trainer.freeze_layers("embedding")  # Freeze embeddings
        """
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                if verbose:
                    print(f"  Froze: {name}")
        
        total = sum(1 for _ in self.model.parameters())
        if verbose:
            print(f"[OK] Frozen {frozen_count}/{total} parameters")
        return frozen_count
    
    def unfreeze_layers(self, pattern: str = None, verbose: bool = True) -> int:
        """Unfreeze all or specific layers.
        
        Args:
            pattern: Layer name pattern to unfreeze (None = all)
            verbose: Print unfrozen layer info
        
        Returns:
            Number of unfrozen parameters
        """
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if pattern is None or pattern in name:
                param.requires_grad = True
                unfrozen_count += 1
                if verbose:
                    print(f"  Unfroze: {name}")
        
        print(f"✓ Unfrozen {unfrozen_count} parameters")
        return unfrozen_count
    
    def get_frozen_layers_info(self) -> dict:
        """Return info about frozen/trainable layers.
        
        Returns:
            Dictionary with 'frozen', 'trainable', 'total', 'trainable_pct'
        """
        frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total = sum(1 for _ in self.model.parameters())
        trainable = total - frozen
        return {
            "frozen": frozen,
            "trainable": trainable,
            "total": total,
            "trainable_pct": (trainable / total * 100) if total > 0 else 0,
        }
    
    def get_train_history(self):
        """Get training history"""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_perplexities": self.val_perplexities,
            "val_token_accuracies": self.val_token_accuracies,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "global_step": self.global_step,
        }
    
    def save(
        self,
        config:Config,
        vocab=None,
        stoi=None,
        itos=None,
        tokenizer_metadata=None,
    ):
        """Save model checkpoint"""
        Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "vocab": vocab,
            "stoi": stoi,
            "itos": itos,
            "tokenizer_metadata": tokenizer_metadata,
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "global_step": self.global_step,
            "patience_counter": self.patience_counter,
            "best_val_loss": self.best_val_loss,
            "best_model_state_dict": self.best_model_state_dict,
            "train_history": self.get_train_history(),
        }
        model_path = Path(config.model_path)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=model_path.parent,
                prefix=f".{model_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, model_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        print(f"[OK] Model saved to {config.model_path}")
    
    def load(self, model_path):
        """Load model checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore training history
        if "train_history" in checkpoint:
            history = checkpoint["train_history"]
            self.train_losses = history.get("train_losses", [])
            self.val_losses = history.get("val_losses", [])
            self.val_perplexities = history.get("val_perplexities", [])
            self.val_token_accuracies = history.get("val_token_accuracies", [])
            self.best_val_loss = history.get("best_val_loss", float('inf'))
            self.patience_counter = history.get("patience_counter", 0)
            self.current_epoch = history.get("current_epoch", len(self.train_losses))
            self.current_batch = history.get("current_batch", 0)
            self.global_step = history.get("global_step", 0)

        self.current_epoch = checkpoint.get("current_epoch", self.current_epoch)
        self.current_batch = checkpoint.get("current_batch", self.current_batch)
        self.global_step = checkpoint.get("global_step", self.global_step)
        self.patience_counter = checkpoint.get("patience_counter", self.patience_counter)
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        self.best_model_state_dict = checkpoint.get("best_model_state_dict")
        
        print(f"[OK] Model loaded from {model_path} at epoch {self.current_epoch}")
        return checkpoint
    
    def exists(self, model_path):
        """Check if model checkpoint exists"""
        return Path(model_path).exists()
