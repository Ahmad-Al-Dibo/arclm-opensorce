"""
Base extension points for ArcLM training workflows.
"""

from abc import ABC, abstractmethod


class BaseModelLoader(ABC):
    """Base class for model loaders that return a model and metadata."""

    @abstractmethod
    def load(self):
        """Load a model source."""
        raise NotImplementedError


class BaseModelAdapter(ABC):
    """Base class for adapting one model implementation into another."""

    @abstractmethod
    def adapt_weights(self, verbose: bool = True):
        """Adapt source weights into the target model."""
        raise NotImplementedError


class BaseTrainingPipeline(ABC):
    """Base class for developer-customizable training pipelines."""

    @abstractmethod
    def build(self, vocab_size: int):
        """Build the pipeline for a vocabulary size."""
        raise NotImplementedError

    @abstractmethod
    def train(self, train_loader, val_loader=None, num_epochs=None):
        """Train the pipeline."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path=None):
        """Save pipeline state."""
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """Return the current model."""
        raise NotImplementedError
