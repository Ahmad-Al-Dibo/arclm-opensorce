"""Release-candidate artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ._version import __version__


SUSPICIOUS_PACKAGE_PATTERNS = (
    "_outputs",
    ".venv",
    ".test-venv",
    ".test-sdist-venv",
    "site/",
    "build/",
    ".arclm",
    "/runs/",
    ".cache",
    "__pycache__",
    "model.safetensors",
)


@dataclass
class PackageContentReport:
    """Distribution content scan result."""

    path: str
    files: int
    suspicious: list[str] = field(default_factory=list)
    report_type: str = "package_content_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__

    @property
    def is_valid(self) -> bool:
        return not self.suspicious

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_valid"] = self.is_valid
        return data


def artifact_checksums(paths: Iterable[str | Path]) -> dict[str, str]:
    """Return SHA-256 checksums keyed by file name."""

    return {Path(path).name: _sha256(Path(path)) for path in paths}


def scan_distribution(path: str | Path) -> PackageContentReport:
    """Scan a wheel or source distribution for generated/private artifacts."""

    artifact = Path(path)
    names = _distribution_names(artifact)
    normalized = [name.replace("\\", "/") for name in names]
    suspicious = [name for name in normalized if any(pattern in name for pattern in SUSPICIOUS_PACKAGE_PATTERNS)]
    return PackageContentReport(path=str(artifact), files=len(names), suspicious=suspicious)


def generate_sbom(output: str | Path, *, include_extras: bool = True) -> Path:
    """Generate a lightweight CycloneDX-style JSON SBOM for the current environment."""

    components = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].lower()):
        components.append(
            {
                "type": "library",
                "name": distribution.metadata["Name"],
                "version": distribution.version,
                "purl": f"pkg:pypi/{distribution.metadata['Name'].lower()}@{distribution.version}",
                "licenses": [{"license": {"name": value}} for value in distribution.metadata.get_all("License") or []],
            }
        )
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"type": "library", "name": "arclm", "version": __version__}},
        "components": components,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sbom, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _distribution_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError("Expected a .whl or .tar.gz artifact.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PackageContentReport", "artifact_checksums", "generate_sbom", "scan_distribution"]
