"""Public API stability manifest for ArcLM.

The manifest is intentionally central.  It gives tests and documentation one
place to detect accidental public API changes while ArcLM is still pre-1.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional


class Stability(str, Enum):
    """Release-candidate stability labels for public interfaces."""

    STABLE = "stable"
    PROVISIONAL = "provisional"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    INTERNAL = "internal"


@dataclass(frozen=True)
class APIEntry:
    """One public API manifest entry."""

    path: str
    kind: str
    stability: Stability
    since: str
    replacement: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, str | None]:
        data = asdict(self)
        data["stability"] = self.stability.value
        return data


_ENTRIES: list[APIEntry] = [
    APIEntry("arclm.Config", "class", Stability.PROVISIONAL, "0.1.0", notes="Legacy native training configuration."),
    APIEntry("arclm.create_config", "function", Stability.PROVISIONAL, "0.1.0"),
    APIEntry("arclm.data.validate_records", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.data.DataPipeline", "class", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.data.open_dataset", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.data.analyze_dataset", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.data.split_dataset", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.data.shard_dataset", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.data.find_duplicates", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.tokenization.tokenize_dataset", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.tokenization.TokenizationConfig", "class", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.models.inspect_model_support", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.models.load_model", "function", Stability.PROVISIONAL, "0.9.0"),
    APIEntry("arclm.training.train", "function", Stability.PROVISIONAL, "0.9.0"),
    APIEntry("arclm.evaluation.evaluate", "function", Stability.PROVISIONAL, "0.9.0"),
    APIEntry("arclm.inference.generate", "function", Stability.PROVISIONAL, "0.9.0"),
    APIEntry("arclm.workflow.run_workflow", "function", Stability.PROVISIONAL, "0.9.0"),
    APIEntry("arclm.config.ArcLMConfig", "class", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.config.load_arclm_config", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.config.migrate_config", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.checkpoints.inspect_checkpoint", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.checkpoints.verify_checkpoint", "function", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.security.LoadingPolicy", "class", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.resources.DeviceConfig", "class", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.runs.Run", "class", Stability.PROVISIONAL, "0.8.0"),
    APIEntry("arclm.registry.Registry", "class", Stability.STABLE, "0.9.0"),
    APIEntry("arclm.MiniGPT", "class", Stability.DEPRECATED, "0.1.0", replacement="arclm.ArcLM"),
    APIEntry(
        "arclm.checkpoint_is_compatible_for_tuining",
        "function",
        Stability.DEPRECATED,
        "0.1.0",
        replacement="arclm.checkpoint_is_compatible_for_tuning",
    ),
    APIEntry("arclm.logics", "module", Stability.DEPRECATED, "0.1.0", notes="Historical logic helpers outside ArcLM's causal-LM scope."),
]

_CLI_STABILITY: dict[str, Stability] = {
    "arclm version": Stability.STABLE,
    "arclm info": Stability.STABLE,
    "arclm doctor": Stability.STABLE,
    "arclm config validate": Stability.STABLE,
    "arclm config show": Stability.STABLE,
    "arclm config migrate": Stability.STABLE,
    "arclm data validate": Stability.STABLE,
    "arclm data analyze": Stability.STABLE,
    "arclm data split": Stability.STABLE,
    "arclm data shard": Stability.STABLE,
    "arclm model inspect": Stability.STABLE,
    "arclm model certify": Stability.EXPERIMENTAL,
    "arclm checkpoint inspect": Stability.STABLE,
    "arclm checkpoint verify": Stability.STABLE,
    "arclm train": Stability.PROVISIONAL,
    "arclm evaluate": Stability.PROVISIONAL,
    "arclm generate": Stability.PROVISIONAL,
    "arclm run": Stability.PROVISIONAL,
}


def api_manifest() -> list[dict[str, str | None]]:
    """Return the machine-readable API stability manifest."""

    return [entry.to_dict() for entry in sorted(_ENTRIES, key=lambda item: item.path)]


def stable_api_paths() -> list[str]:
    """Return stable public API paths used by snapshot tests."""

    return sorted(entry.path for entry in _ENTRIES if entry.stability is Stability.STABLE)


def classify_api(path: str) -> Optional[APIEntry]:
    """Return manifest metadata for a public API path when known."""

    for entry in _ENTRIES:
        if entry.path == path:
            return entry
    return None


def cli_manifest() -> dict[str, str]:
    """Return command stability labels."""

    return {command: stability.value for command, stability in sorted(_CLI_STABILITY.items())}


def assert_no_unknown_stable_apis(paths: Iterable[str]) -> None:
    """Fail when a stable API snapshot contains entries missing from the manifest."""

    known = set(stable_api_paths())
    unknown = sorted(set(paths) - known)
    if unknown:
        raise AssertionError("Unknown stable API path(s): " + ", ".join(unknown))


__all__ = ["APIEntry", "Stability", "api_manifest", "assert_no_unknown_stable_apis", "classify_api", "cli_manifest", "stable_api_paths"]
