"""Optional compatibility backends.

This package intentionally does not import external ecosystems at module import
time. Backend modules load optional dependencies only when compatibility models
are requested.
"""

__all__: list[str] = []
