"""
Experiment Tracking Integration

Provides unified interface for tracking experiments with multiple backends:
- Local file tracking (default, no dependencies)
- MLflow integration (optional)
- Weights & Biases integration (optional)
- Neptune integration (optional)

Example:
    from arclm.tracking import ExperimentTracker
    
    tracker = ExperimentTracker(name="my_experiment", backend="local")
    tracker.log_config(config)
    tracker.log_metrics({"loss": 2.5, "accuracy": 0.92})
    tracker.log_artifact("model.pth")
    tracker.end()
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import warnings

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment"""
    name: str
    experiment_id: str
    backend: str
    start_time: str
    config: Dict[str, Any]
    tags: Dict[str, str]


class ExperimentTracker:
    """
    Unified experiment tracking interface.
    
    Supports multiple backends:
    - "local": Local file-based tracking (default, no dependencies)
    - "mlflow": MLflow experiment tracking
    - "wandb": Weights & Biases tracking
    
    Example:
        # Local tracking (no setup required)
        tracker = ExperimentTracker(name="baseline")
        tracker.log_metrics({"loss": 2.5})
        tracker.end()
        
        # MLflow tracking
        tracker = ExperimentTracker(name="experiment1", backend="mlflow")
        tracker.log_config(config_dict)
        tracker.log_metrics({"epoch": 1, "loss": 2.5})
        tracker.end()
    """
    
    def __init__(
        self,
        name: str,
        backend: str = "local",
        experiment_dir: str = "experiments",
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize experiment tracker.
        
        Args:
            name: Experiment name
            backend: Tracking backend ("local", "mlflow", "wandb")
            experiment_dir: Directory for local tracking
            tags: Optional tags for experiment
            **kwargs: Backend-specific arguments
        """
        self.name = name
        self.backend = backend
        self.tags = tags or {}
        self.experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if backend == "local":
            self._init_local(experiment_dir)
        elif backend == "mlflow":
            self._init_mlflow(**kwargs)
        elif backend == "wandb":
            self._init_wandb(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_local(self, experiment_dir: str):
        """Initialize local file tracking"""
        self.exp_dir = Path(experiment_dir) / self.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        self.metadata = {
            "name": self.name,
            "experiment_id": self.experiment_id,
            "backend": "local",
            "start_time": datetime.now().isoformat(),
            "tags": self.tags,
        }
        
        self.config_file = self.exp_dir / "config.json"
        self.metrics_file = self.exp_dir / "metrics.jsonl"
        self.artifacts_dir = self.exp_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
    
    def _init_mlflow(self, **kwargs):
        """Initialize MLflow tracking"""
        if not MLFLOW_AVAILABLE:
            raise ValueError(
                "MLflow not installed. Install with: pip install mlflow"
            )
        
        # Set tracking URI if provided
        if "tracking_uri" in kwargs:
            mlflow.set_tracking_uri(kwargs["tracking_uri"])
        
        # Set experiment
        mlflow.set_experiment(self.name)
        
        # Start run
        self.mlflow_run = mlflow.start_run()
        
        # Set tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
    
    def _init_wandb(self, **kwargs):
        """Initialize Weights & Biases tracking"""
        if not WANDB_AVAILABLE:
            raise ValueError(
                "Weights & Biases not installed. Install with: pip install wandb"
            )
        
        wandb.init(
            project=kwargs.get("project", "arclm"),
            name=self.name,
            tags=list(self.tags.values()),
            config=kwargs.get("config", {}),
        )
    
    def log_config(self, config: Union[Dict[str, Any], Any]) -> None:
        """
        Log configuration.
        
        Args:
            config: Configuration object (dict or Config object)
        """
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        else:
            config_dict = dict(config) if isinstance(config, dict) else vars(config)
        
        if self.backend == "local":
            with open(self.config_file, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif self.backend == "mlflow":
            mlflow.log_params(config_dict)
        elif self.backend == "wandb":
            wandb.config.update(config_dict)
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step/epoch number
        """
        if self.backend == "local":
            entry = {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                **metrics
            }
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        
        elif self.backend == "mlflow":
            mlflow.log_metrics(metrics, step=step)
        
        elif self.backend == "wandb":
            if step is not None:
                metrics["step"] = step
            wandb.log(metrics)
    
    def log_artifact(self, filepath: Union[str, Path]) -> None:
        """
        Log artifact (file).
        
        Args:
            filepath: Path to file to log
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            warnings.warn(f"Artifact file not found: {filepath}")
            return
        
        if self.backend == "local":
            import shutil
            dest = self.artifacts_dir / filepath.name
            shutil.copy(filepath, dest)
        
        elif self.backend == "mlflow":
            mlflow.log_artifact(str(filepath))
        
        elif self.backend == "wandb":
            wandb.save(str(filepath))
    
    def log_text(self, text: str, filename: str = "notes.txt") -> None:
        """
        Log text content.
        
        Args:
            text: Text to log
            filename: Name for stored file
        """
        if self.backend == "local":
            filepath = self.artifacts_dir / filename
            with open(filepath, "w") as f:
                f.write(text)
        
        elif self.backend == "mlflow":
            filepath = Path(filename)
            filepath.write_text(text)
            mlflow.log_artifact(str(filepath))
            filepath.unlink()
        
        elif self.backend == "wandb":
            with open(filename, "w") as f:
                f.write(text)
            wandb.save(filename)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters (alternative to log_config).
        
        Args:
            params: Dictionary of parameters
        """
        if self.backend == "local":
            # Store in config
            with open(self.config_file, "a") as f:
                f.write(json.dumps({"params": params}, indent=2))
        
        elif self.backend == "mlflow":
            mlflow.log_params(params)
        
        elif self.backend == "wandb":
            wandb.config.update(params)
    
    def get_metric_history(self, metric_name: Optional[str] = None) -> List[Dict]:
        """
        Get metric history (local backend only).
        
        Args:
            metric_name: Optional specific metric to retrieve
        
        Returns:
            List of metric entries
        """
        if self.backend != "local":
            raise NotImplementedError(
                f"Metric history not available for backend: {self.backend}"
            )
        
        if not self.metrics_file.exists():
            return []
        
        history = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if metric_name is None or metric_name in entry:
                        history.append(entry)
        
        return history
    
    def end(self) -> str:
        """
        End experiment tracking.
        
        Returns:
            Experiment ID
        """
        if self.backend == "local":
            # Save final metadata
            metadata_file = self.exp_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"[OK] Experiment saved to: {self.exp_dir}")
        
        elif self.backend == "mlflow":
            mlflow.end_run()
        
        elif self.backend == "wandb":
            wandb.finish()
        
        return self.experiment_id
    
    def get_experiment_dir(self) -> Optional[Path]:
        """Get experiment directory (local backend only)"""
        if self.backend == "local":
            return self.exp_dir
        return None


# Convenience functions
def create_experiment(
    name: str,
    backend: str = "local",
    **kwargs
) -> ExperimentTracker:
    """
    Create and return an experiment tracker.
    
    Convenience function for quick experiment setup.
    
    Args:
        name: Experiment name
        backend: Tracking backend
        **kwargs: Backend-specific arguments
    
    Returns:
        ExperimentTracker instance
    
    Example:
        exp = create_experiment("baseline")
        exp.log_config({"lr": 0.001})
        exp.log_metrics({"loss": 2.5})
    """
    return ExperimentTracker(name, backend=backend, **kwargs)


def list_experiments(experiment_dir: str = "experiments") -> List[str]:
    """
    List all local experiments.
    
    Args:
        experiment_dir: Directory containing experiments
    
    Returns:
        List of experiment directories
    """
    exp_dir = Path(experiment_dir)
    if not exp_dir.exists():
        return []
    
    return [d.name for d in exp_dir.iterdir() if d.is_dir()]
