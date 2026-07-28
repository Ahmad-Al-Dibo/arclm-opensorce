"""Security checks for local artifacts and controlled model loading."""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ._version import __version__


SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"]?[^'\"\s]+"),
]


@dataclass(frozen=True)
class LoadingPolicy:
    """Explicit checkpoint/model loading trust policy."""

    mode: str = "safe"
    allow_pickle: bool = False
    allow_remote_code: bool = False
    verify_hashes: bool = True

    @classmethod
    def safe(cls) -> "LoadingPolicy":
        return cls("safe", allow_pickle=False, allow_remote_code=False, verify_hashes=True)

    @classmethod
    def trusted_local(cls) -> "LoadingPolicy":
        return cls("trusted_local", allow_pickle=True, allow_remote_code=False, verify_hashes=True)

    @classmethod
    def legacy_unsafe(cls) -> "LoadingPolicy":
        warnings.warn(
            "legacy_unsafe loading may invoke pickle deserialization or untrusted code. Use only for trusted local artifacts.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cls("legacy_unsafe", allow_pickle=True, allow_remote_code=True, verify_hashes=False)

    @classmethod
    def from_value(cls, value: str | "LoadingPolicy") -> "LoadingPolicy":
        if isinstance(value, LoadingPolicy):
            return value
        if value == "safe":
            return cls.safe()
        if value == "trusted_local":
            return cls.trusted_local()
        if value == "legacy_unsafe":
            return cls.legacy_unsafe()
        raise ValueError("loading policy must be 'safe', 'trusted_local', or 'legacy_unsafe'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SecurityIssue:
    category: str
    path: str
    message: str
    severity: str = "warning"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class SecurityReport:
    issues: list[SecurityIssue] = field(default_factory=list)
    scanned_files: int = 0
    report_type: str = "security_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "scanned_files": self.scanned_files,
            "report_type": self.report_type,
            "schema_version": self.schema_version,
            "arclm_version": self.arclm_version,
            "is_valid": self.is_valid,
        }


def scan_for_secrets(paths: Iterable[str | Path]) -> SecurityReport:
    """Scan small text files for common accidental secret patterns."""

    report = SecurityReport()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = [item for item in path.rglob("*") if item.is_file()]
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate.stat().st_size > 2_000_000:
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except Exception:
                continue
            report.scanned_files += 1
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    report.issues.append(SecurityIssue("possible_secret", str(candidate), "Possible secret-like value detected.", "error"))
                    break
    return report


def artifact_digest(path: str | Path) -> str:
    """Return SHA-256 digest for a file artifact."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_safe_model_options(*, trust_remote_code: bool = False, policy: str | LoadingPolicy = "safe") -> None:
    """Fail if unsafe remote-code execution would be enabled implicitly."""

    resolved = LoadingPolicy.from_value(policy)
    if trust_remote_code and not resolved.allow_remote_code:
        raise ValueError("trust_remote_code=True must be an explicit user decision and is not allowed in safe mode.")


__all__ = ["LoadingPolicy", "SecurityIssue", "SecurityReport", "artifact_digest", "scan_for_secrets", "validate_safe_model_options"]
