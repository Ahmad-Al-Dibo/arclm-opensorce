"""First-class model architecture system."""

from .base import (
    Architecture,
    ArchitectureCapabilities,
    ArchitectureKind,
    ArchitectureMetadata,
    CapabilitySupport,
    IdentityWeightMapper,
)
from .native import ArcLMNativeCausalLM
from .registry import ArchitectureRegistry, architectures

__all__ = [
    "Architecture",
    "ArchitectureCapabilities",
    "ArchitectureKind",
    "ArchitectureMetadata",
    "ArchitectureRegistry",
    "ArcLMNativeCausalLM",
    "CapabilitySupport",
    "IdentityWeightMapper",
    "architectures",
]
