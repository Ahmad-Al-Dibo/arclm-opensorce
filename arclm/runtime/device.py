"""Runtime and device inspection owned by ArcLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..exceptions import RuntimePlanningError


@dataclass(frozen=True)
class DeviceInfo:
    """ArcLM description of a compute device."""

    type: str
    name: str
    index: int | None = None
    total_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True)
class Runtime:
    """Resolved ArcLM runtime plan for local execution."""

    backend: str
    device: DeviceInfo
    precision: str
    available_devices: tuple[DeviceInfo, ...]
    warnings: tuple[str, ...] = ()
    backend_capabilities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def auto(cls, *, prefer: str = "auto", precision: str = "auto") -> "Runtime":
        """Inspect the local machine and choose a runtime."""

        try:
            import torch
        except Exception as exc:  # pragma: no cover - exercised only without torch
            raise RuntimePlanningError(
                "Runtime discovery failed because the torch backend is unavailable.",
                subsystem="runtime",
                action="Install PyTorch or request a future non-torch backend when available.",
                cause=repr(exc),
            ) from exc

        requested = str(prefer or "auto").lower().strip()
        devices = [DeviceInfo(type="cpu", name="CPU", capabilities={"precision": ["float32"]})]
        warnings: list[str] = []

        cuda_available = bool(getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
        if cuda_available:
            for index in range(int(torch.cuda.device_count())):
                props = torch.cuda.get_device_properties(index)
                free_memory = None
                try:
                    free_memory = int(torch.cuda.mem_get_info(index)[0])
                except Exception:
                    free_memory = None
                devices.append(
                    DeviceInfo(
                        type="cuda",
                        name=str(torch.cuda.get_device_name(index)),
                        index=index,
                        total_memory_bytes=int(getattr(props, "total_memory", 0)) or None,
                        available_memory_bytes=free_memory,
                        capabilities={
                            "compute_capability": tuple(torch.cuda.get_device_capability(index)),
                            "bf16": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
                            "precision": ["float32", "float16", "bfloat16"],
                        },
                    )
                )
        mps_available = bool(
            getattr(torch, "backends", None) is not None
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        if mps_available:
            devices.append(
                DeviceInfo(
                    type="mps",
                    name="Apple Metal Performance Shaders",
                    capabilities={"precision": ["float32", "float16"]},
                )
            )

        selected = devices[0]
        if requested == "auto":
            selected = devices[1] if len(devices) > 1 else devices[0]
        elif requested == "cpu":
            selected = devices[0]
        elif requested == "cuda":
            if not cuda_available:
                warnings.append("CUDA was requested but is not available; using CPU.")
            else:
                selected = devices[1]
        elif requested == "mps":
            matches = [device for device in devices if device.type == "mps"]
            if not matches:
                warnings.append("MPS was requested but is not available; using CPU.")
            else:
                selected = matches[0]
        elif requested.startswith("cuda:"):
            selected = cls._select_cuda_device(requested, devices, warnings)
        else:
            raise RuntimePlanningError(
                f"Unsupported runtime preference: {prefer!r}.",
                subsystem="runtime",
                action="Use one of: auto, cpu, cuda, cuda:<index>, mps.",
            )

        resolved_precision = cls._resolve_precision(precision, selected)
        return cls(
            backend="torch",
            device=selected,
            precision=resolved_precision,
            available_devices=tuple(devices),
            warnings=tuple(warnings),
            backend_capabilities={
                "name": "torch",
                "devices": sorted({device.type for device in devices}),
                "future_backends": ["jax", "mlx"],
                "distributed": "planned",
            },
        )

    @staticmethod
    def _select_cuda_device(requested: str, devices: list[DeviceInfo], warnings: list[str]) -> DeviceInfo:
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device reference: {requested!r}.") from exc
        for device in devices:
            if device.type == "cuda" and device.index == index:
                return device
        warnings.append(f"{requested} was requested but is not available; using CPU.")
        return devices[0]

    @staticmethod
    def _resolve_precision(precision: str, device: DeviceInfo) -> str:
        requested = str(precision or "auto").lower().replace("torch.", "").strip()
        aliases = {
            "fp32": "float32",
            "float": "float32",
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
        }
        requested = aliases.get(requested, requested)
        if requested == "auto":
            if device.type == "cuda":
                return "bfloat16" if device.capabilities.get("bf16") else "float16"
            return "float32"
        if requested not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be one of: auto, float32, fp32, float16, fp16, bfloat16, bf16.")
        if device.type == "cpu" and requested in {"float16", "bfloat16"}:
            return "float32"
        if device.type == "mps" and requested == "bfloat16":
            return "float32"
        return requested

    @property
    def device_name(self) -> str:
        """Return a backend device string for internal backend adapters."""

        if self.device.type == "cuda":
            return f"cuda:{self.device.index or 0}"
        return self.device.type

    def memory_report(self) -> dict[str, Any]:
        """Return memory information for selected and available devices."""

        return {
            "selected": self.device.to_dict(),
            "available_devices": [device.to_dict() for device in self.available_devices],
        }

    def compute_plan(self, *, parameters: int = 0, trainable_parameters: int | None = None, activation_bytes: int = 0) -> dict[str, Any]:
        """Return a rough compute and memory plan for a workload."""

        dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}.get(self.precision, 4)
        trainable = int(parameters if trainable_parameters is None else trainable_parameters)
        parameter_memory = int(parameters) * dtype_bytes
        gradient_memory = trainable * dtype_bytes
        optimizer_memory = trainable * dtype_bytes * 2
        total = parameter_memory + gradient_memory + optimizer_memory + int(activation_bytes)
        available = self.device.available_memory_bytes or self.device.total_memory_bytes
        fits = None if available is None else total <= int(available)
        return {
            "backend": self.backend,
            "device": self.device_name,
            "precision": self.precision,
            "parameters": int(parameters),
            "trainable_parameters": trainable,
            "estimated_parameter_memory_bytes": parameter_memory,
            "estimated_gradient_memory_bytes": gradient_memory,
            "estimated_optimizer_memory_bytes": optimizer_memory,
            "estimated_activation_memory_bytes": int(activation_bytes),
            "estimated_total_memory_bytes": total,
            "available_memory_bytes": available,
            "fits_selected_device": fits,
        }

    def torch_device(self):
        """Return a `torch.device` for low-level interoperability."""

        import torch

        return torch.device(self.device_name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe runtime report."""

        return {
            "backend": self.backend,
            "device": self.device.to_dict(),
            "precision": self.precision,
            "available_devices": [device.to_dict() for device in self.available_devices],
            "warnings": list(self.warnings),
            "memory": self.memory_report(),
            "backend_capabilities": dict(self.backend_capabilities),
        }
