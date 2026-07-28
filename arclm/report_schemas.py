"""Versioned report envelope helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ._version import __version__
from .reproducibility import fingerprint


REPORT_SCHEMA_VERSION = "1.0"


@dataclass
class ReportEnvelope:
    """Versioned wrapper for serialized reports."""

    report_type: str
    payload: dict[str, Any]
    schema_version: str = REPORT_SCHEMA_VERSION
    arclm_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    configuration_fingerprint: Optional[str] = None
    source_fingerprint: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    serialization: str = "json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)


def envelope_report(report: Any, *, report_type: Optional[str] = None, config: Any = None, source: Any = None) -> ReportEnvelope:
    """Wrap a report object in a stable envelope."""

    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    return ReportEnvelope(
        report_type=report_type or payload.get("report_type", type(report).__name__),
        payload=payload,
        configuration_fingerprint=fingerprint(config).value if config is not None else None,
        source_fingerprint=fingerprint(source).value if source is not None else None,
        warnings=list(payload.get("warnings", [])),
        errors=list(payload.get("errors", [])),
    )


__all__ = ["REPORT_SCHEMA_VERSION", "ReportEnvelope", "envelope_report"]
