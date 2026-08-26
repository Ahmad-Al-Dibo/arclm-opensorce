"""Compatibility re-exports for the first-class architecture system."""

from __future__ import annotations

from ..architectures import Architecture, ArcLMNativeCausalLM

ArcLMNativeArchitecture = ArcLMNativeCausalLM

__all__ = ["Architecture", "ArcLMNativeArchitecture", "ArcLMNativeCausalLM"]
