"""Shared data models for the baseagent harness.

These types form the provider-neutral core of the system: the engine speaks in
these terms, and each provider adapter converts to/from its native API format.
Content blocks mirror Anthropic's model (``text`` / ``tool_use`` /
``tool_result`` / ``image``) since that is the richest common denominator.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union


# ---------------------------------------------------------------------------
# Usage / token accounting
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Token usage for a single request or cumulative across a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0  # Gemini/prompt-caching writes
    total_tokens: int = 0

    @classmethod
    def empty(cls) -> "Usage":
        return cls()

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_write_input_tokens += other.cache_write_input_tokens
        self.total_tokens = self._compute_total()

    def _compute_total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.cache_write_input_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Usage":
        return cls(
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            cache_creation_input_tokens=int(d.get("cache_creation_input_tokens", 0)),
            cache_read_input_tokens=int(d.get("cache_read_input_tokens", 0)),
            cache_write_input_tokens=int(d.get("cache_write_input_tokens", 0)),
        )


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class ToolUseBlock:
    """An assistant request to invoke a tool (mirrors the ref ToolUseBlock)."""

    type: Literal["tool_use"] = "tool_use"
    id: str = field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:24]}")
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class ToolResultBlock:
    """The result of a tool invocation, correlated to a ToolUseBlock by id."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""

    # ``is_error`` marks a failed invocation; ``content`` is a string for an
    # error, or a string / list[TextBlock | ImageBlock] for a success.
    is_error: bool = False
    content: Union[str, list[Any], None] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "is_error": self.is_error,
            "content": self.content,
        }


@dataclass
class ImageBlock:
    type: Literal["image"] = "image"
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "image", "source": self.source}


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock]
BLOCK_CLASSES = {
    "text": TextBlock,
    "tool_use": ToolUseBlock,
    "tool_result": ToolResultBlock,
    "image": ImageBlock,
}


def block_from_dict(d: dict[str, Any]) -> ContentBlock:
    cls = BLOCK_CLASSES.get(d.get("type", ""))
    if cls is None:
        raise ValueError(f"Unknown content block type: {d.get('type')!r}")
    return cls(**{k: v for k, v in d.items() if k != "type"})


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single conversation message with content blocks."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock] = field(default_factory=list)
    # Optional plain-text shorthand (convenience for the engine/server).
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": [b.to_dict() for b in self.content],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        blocks = [block_from_dict(b) for b in d.get("content", [])]
        return cls(role=d["role"], content=blocks)

    @classmethod
    def text_message(cls, role: Literal["user", "assistant"], text: str) -> "Message":
        return cls(role=role, content=[TextBlock(text=text)], text=text)


# ---------------------------------------------------------------------------
# Tool results / errors
# ---------------------------------------------------------------------------

ToolStatus = Literal["success", "error"]


@dataclass
class ToolResult:
    """Outcome of executing a single tool (used by tools and the engine)."""

    status: ToolStatus
    output: str = ""
    is_error: bool = False
    # A short machine-readable tag, e.g. "bad_parameters", "path_outside_sandbox".
    error_type: Optional[str] = None
    error_message: str = ""

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        return cls(status="success", output=output, is_error=False)

    @classmethod
    def fail(cls, message: str, error_type: Optional[str] = None) -> "ToolResult":
        return cls(
            status="error",
            output=message,
            is_error=True,
            error_type=error_type,
            error_message=message,
        )

    @property
    def as_block_content(self) -> list[Any]:
        """The content to place inside a ToolResultBlock for the model."""
        return [{"type": "text", "text": self.output}]


# ---------------------------------------------------------------------------
# Stream events (the server streams these over SSE)
# ---------------------------------------------------------------------------

@dataclass
class RequestStartEvent:
    type: Literal["request_start"] = "request_start"
    iteration: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "iteration": self.iteration}


@dataclass
class AssistantMessageEvent:
    type: Literal["assistant_message"] = "assistant_message"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


@dataclass
class ToolUseEvent:
    type: Literal["tool_use"] = "tool_use"
    name: str = ""
    tool_use_id: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "name": self.name, "id": self.tool_use_id, "input": self.input}


@dataclass
class ToolResultEvent:
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    is_error: bool = False
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "id": self.tool_use_id, "is_error": self.is_error, "output": self.output}


@dataclass
class IterationEvent:
    type: Literal["iteration"] = "iteration"
    iteration: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "iteration": self.iteration}


@dataclass
class UsageEvent:
    type: Literal["usage"] = "usage"
    usage: Usage = field(default_factory=Usage.empty)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "usage": self.usage.to_dict()}


@dataclass
class TerminalEvent:
    type: Literal["terminal"] = "terminal"
    reason: str = "stop"  # stop | max_turns | budget | error | aborted
    turn_count: int = 0
    text: str = ""
    usage: Usage = field(default_factory=Usage.empty)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "reason": self.reason,
            "turn_count": self.turn_count,
            "text": self.text,
            "usage": self.usage.to_dict(),
            "error": self.error,
        }


StreamEvent = Union[
    RequestStartEvent,
    AssistantMessageEvent,
    ToolUseEvent,
    ToolResultEvent,
    IterationEvent,
    UsageEvent,
    TerminalEvent,
]

EVENT_CLASSES = {
    "request_start": RequestStartEvent,
    "assistant_message": AssistantMessageEvent,
    "tool_use": ToolUseEvent,
    "tool_result": ToolResultEvent,
    "iteration": IterationEvent,
    "usage": UsageEvent,
    "terminal": TerminalEvent,
}


def event_from_dict(d: dict[str, Any]) -> StreamEvent:
    cls = EVENT_CLASSES.get(d.get("type", ""))
    if cls is None:
        raise ValueError(f"Unknown stream event type: {d.get('type')!r}")
    kwargs = dict(d)
    kwargs.pop("type", None)
    if "usage" in kwargs and isinstance(kwargs["usage"], dict):
        kwargs["usage"] = Usage.from_dict(kwargs["usage"])
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Terminal result of a run
# ---------------------------------------------------------------------------

@dataclass
class Terminal:
    """The terminal result of a run, mirroring the ref query loop's Terminal."""

    reason: Literal["stop", "max_turns", "budget", "error", "aborted"] = "stop"
    turn_count: int = 0
    text: str = ""
    usage: Usage = field(default_factory=Usage.empty)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def to_event(self) -> TerminalEvent:
        return TerminalEvent(
            reason=self.reason,
            turn_count=self.turn_count,
            text=self.text,
            usage=self.usage,
            error=self.error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "turn_count": self.turn_count,
            "text": self.text,
            "usage": self.usage.to_dict(),
            "error": self.error,
            "duration_seconds": (
                round((self.ended_at or time.time()) - self.started_at, 3)
            ),
        }
