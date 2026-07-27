"""
Regularization & Generalization Utilities
Technieken om model generalisatie te verbeteren
"""

import torch
import torch.nn as nn


class L1Regularization:
    """L1 Regularization - Sparsity inducing"""
    
    def __init__(self, lambda_l1=0.0001):
        self.lambda_l1 = lambda_l1
    
    def compute_loss(self, model):
        """Compute L1 regularization loss"""
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.lambda_l1 * l1_loss


class L2Regularization:
    """L2 Regularization (Weight Decay) - Smoothness inducing"""
    
    def __init__(self, lambda_l2=0.0001):
        self.lambda_l2 = lambda_l2
    
    def compute_loss(self, model):
        """Compute L2 regularization loss"""
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.lambda_l2 * l2_loss


class Dropout(nn.Module):
    """Dropout regularization"""
    
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        return self.dropout(x)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=1e-4):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_loss = None
    
    def check(self, val_loss):
        """Check if training should stop
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        return False
    
    def reset(self):
        """Reset early stopping"""
        self.patience_counter = 0
        self.best_loss = None


class LearningRateScheduler:
    """Learning rate scheduler for better generalization
    
    Reduce learning rate during training for better convergence
    """
    
    def __init__(self, optimizer, strategy="cosine", total_epochs=100):
        """
        Args:
            optimizer: PyTorch optimizer
            strategy: "cosine", "linear", or "exponential"
            total_epochs: Total number of training epochs
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.initial_lr = optimizer.defaults['lr']
    
    def step(self, epoch):
        """Update learning rate for current epoch"""
        if self.strategy == "cosine":
            lr = self.initial_lr * (1 + torch.cos(torch.tensor(torch.pi * epoch / self.total_epochs))) / 2
        elif self.strategy == "linear":
            lr = self.initial_lr * (1 - epoch / self.total_epochs)
        elif self.strategy == "exponential":
            lr = self.initial_lr * (0.1 ** (epoch / self.total_epochs))
        else:
            return
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class GeneralizationMonitor:
    """Monitor model generalization during training
    
    Traceert training vs validation loss om overfitting te detecteren
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    
    def update(self, train_loss, val_loss):
        """Update monitor with new losses"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def get_generalization_gap(self):
        """Get current generalization gap (val - train)"""
        if not self.train_losses or not self.val_losses:
            return None
        return self.val_losses[-1] - self.train_losses[-1]
    
    def get_average_gap(self, window=10):
        """Get average generalization gap over window"""
        if len(self.val_losses) < window:
            return None
        
        recent_train = sum(self.train_losses[-window:]) / window
        recent_val = sum(self.val_losses[-window:]) / window
        return recent_val - recent_train
    
    def is_overfitting(self, threshold=0.5):
        """Check if model is overfitting
        
        Args:
            threshold: Gap threshold to consider overfitting
        
        Returns:
            True if generalization gap > threshold
        """
        gap = self.get_generalization_gap()
        return gap is not None and gap > threshold
    
    def get_report(self):
        """Get generalization report"""
        if not self.train_losses or not self.val_losses:
            return "No data available"
        
        gap = self.get_generalization_gap()
        avg_gap = self.get_average_gap()
        avg_gap_str = f"{avg_gap:.4f}" if avg_gap is not None else "N/A"
        overfitting_status = "[WARNING]  OVERFITTING" if self.is_overfitting() else "[OK] OK"
        
        report = f"""
Generalization Report:
  Latest Train Loss: {self.train_losses[-1]:.4f}
  Latest Val Loss:   {self.val_losses[-1]:.4f}
  Generalization Gap: {gap:.4f}
  Avg Gap (10-epoch): {avg_gap_str}
  Status: {overfitting_status}
        """
        return report


class MixupAugmentation:
    """Mixup data augmentation for better generalization
    
    Combineert twee samples lineair om data augmentation te krijgen
    """
    
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, x, y):
        """Apply mixup augmentation
        
        Args:
            x: Input batch [batch_size, seq_len]
            y: Target batch [batch_size, seq_len]
        
        Returns:
            Mixed x and y
        """
        batch_size = x.size(0)
        
        # Random lambda
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y


class LabelSmoothing(nn.Module):
    """Label smoothing regularization
    
    Voorkomt overconfident voorspellingen door targets glad te maken
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (0-1)
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """Compute smoothed cross-entropy loss
        
        Args:
            pred: Model predictions [batch, num_classes]
            target: Target labels [batch]
        
        Returns:
            Smoothed cross-entropy loss
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
