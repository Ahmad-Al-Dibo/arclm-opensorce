"""Backend boundary for ArcLM Core."""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable


@runtime_checkable
class BackendContract(Protocol):
    """Protocol re-export for backend implementations."""

    name: str

    def create_optimizer(self, parameters: Iterable[Any], *, learning_rate: float, weight_decay: float = 0.0) -> Any: ...
    def zero_grad(self, optimizer: Any) -> None: ...
    def backward(self, loss: Any) -> None: ...
    def optimizer_step(self, optimizer: Any) -> None: ...
    def scheduler_step(self, scheduler: Any, metric: float | None = None) -> None: ...
    def cross_entropy_next_token_loss(self, logits: Any, targets: Any) -> Any: ...
    def clip_grad_norm(self, parameters: Iterable[Any], max_norm: float | None) -> float | None: ...
    def scalar(self, value: Any) -> float: ...


class TorchBackend:
    """Torch implementation of the backend contract."""

    name = "torch"

    def create_optimizer(self, parameters: Iterable[Any], *, learning_rate: float, weight_decay: float = 0.0) -> Any:
        import torch

        trainable = [parameter for parameter in parameters if getattr(parameter, "requires_grad", False)]
        if not trainable:
            raise ValueError("No trainable parameters are available for optimization.")
        return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: Any) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: Any) -> None:
        optimizer.step()

    def scheduler_step(self, scheduler: Any, metric: float | None = None) -> None:
        if scheduler is None:
            return
        if metric is not None and scheduler.__class__.__name__ == "ReduceLROnPlateau":
            scheduler.step(metric)
            return
        scheduler.step()

    def cross_entropy_next_token_loss(self, logits: Any, targets: Any) -> Any:
        import torch

        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def clip_grad_norm(self, parameters: Iterable[Any], max_norm: float | None) -> float | None:
        if max_norm is None:
            return None
        import torch

        trainable = [parameter for parameter in parameters if getattr(parameter, "grad", None) is not None]
        if not trainable:
            return None
        return float(torch.nn.utils.clip_grad_norm_(trainable, max_norm).detach().cpu().item())

    def scalar(self, value: Any) -> float:
        if hasattr(value, "detach"):
            return float(value.detach().cpu().item())
        return float(value)
