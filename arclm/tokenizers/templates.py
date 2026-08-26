"""ArcLM-owned chat template formatting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Iterable, Protocol


class ChatFormatter(Protocol):
    """Research contract for custom chat formatting logic."""

    def format(self, messages: Iterable[dict[str, Any]], *, add_generation_prompt: bool = False) -> str: ...


@dataclass(frozen=True)
class ChatTemplate:
    """Small public chat-template abstraction for instruct/chat data."""

    name: str = "default"
    system: str = "<|system|>\n{content}</s>"
    user: str = "<|user|>\n{content}</s>"
    assistant: str = "<|assistant|>\n{content}</s>"
    tool: str = "<|tool|>\n{content}</s>"
    separator: str = "\n"
    generation_prompt: str = "<|assistant|>\n"
    default_system: str | None = None
    role_templates: dict[str, str] = field(default_factory=dict)

    _registry: ClassVar[dict[str, "ChatTemplate"]] = {}

    def format(self, messages: Iterable[dict[str, Any]], *, add_generation_prompt: bool = False) -> str:
        """Format chat messages into model training text."""

        normalized = self._normalize_messages(messages)
        if self.default_system and not any(message["role"] == "system" for message in normalized):
            normalized.insert(0, {"role": "system", "content": self.default_system})

        parts = []
        for message in normalized:
            template = self.template_for(message["role"])
            parts.append(template.format(**message))
        if add_generation_prompt:
            parts.append(self.generation_prompt)
        return self.separator.join(part.strip() for part in parts if part is not None)

    def template_for(self, role: str) -> str:
        normalized = str(role or "user").lower().strip()
        if normalized in self.role_templates:
            return self.role_templates[normalized]
        if normalized == "system":
            return self.system
        if normalized == "assistant":
            return self.assistant
        if normalized == "tool":
            return self.tool
        return self.user

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | "ChatTemplate" | None) -> "ChatTemplate":
        if isinstance(data, ChatTemplate):
            return data
        if not data:
            return cls.default()
        allowed = {field_name for field_name in cls.__dataclass_fields__ if not field_name.startswith("_")}
        return cls(**{key: value for key, value in dict(data).items() if key in allowed})

    @classmethod
    def default(cls) -> "ChatTemplate":
        return cls()

    @classmethod
    def register(cls, name: str, template: "ChatTemplate") -> None:
        cls._registry[str(name).lower().strip()] = template

    @classmethod
    def get(cls, name: str) -> "ChatTemplate":
        key = str(name).lower().strip()
        if key == "default":
            return cls.default()
        try:
            return cls._registry[key]
        except KeyError as exc:
            raise ValueError(f"Unknown chat template: {name!r}.") from exc

    @staticmethod
    def _normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Chat messages must be dictionaries with role/content fields.")
            normalized.append(
                {
                    **message,
                    "role": str(message.get("role", "user")).lower().strip(),
                    "content": str(message.get("content", "")),
                }
            )
        return normalized

__all__ = ["ChatFormatter", "ChatTemplate"]
