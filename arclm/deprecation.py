"""Deprecation helpers for ArcLM compatibility APIs."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def warn_deprecated(
    name: str,
    replacement: Optional[str] = None,
    removal_version: str = "0.9.0",
) -> None:
    """Emit a standardized ArcLM deprecation warning."""

    message = f"{name} is deprecated and scheduled for removal in ArcLM {removal_version}."
    if replacement:
        message += f" Use {replacement} instead."
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def deprecated(
    name: Optional[str] = None,
    replacement: Optional[str] = None,
    removal_version: str = "0.9.0",
) -> Callable[[F], F]:
    """Decorator that warns when a deprecated function is called."""

    def decorate(func: F) -> F:
        public_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(public_name, replacement, removal_version)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


__all__ = ["deprecated", "warn_deprecated"]
