"""Provider abstraction: the transport-agnostic interface every adapter implements.

A provider converts the engine's provider-neutral ``system + messages + tools``
into one vendor's API, streams the model's response back as a sequence of
:class:`~baseagent.types.StreamEvent` (so the server can re-stream them over SSE),
and finally yields a :class:`Turn` carrying the assembled assistant message, stop
reason, and per-request usage.

The engine consumes the generator and does not care which vendor produced it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

from ..config import ProviderConfig
from ..types import Message, Usage, StreamEvent


@dataclass
class Turn:
    """The completed model response for a single request to the provider."""

    assistant_message: Message
    stop_reason: str  # 'tool_use' | 'stop' | 'max_tokens' | 'end_turn' | 'error'
    usage: Usage = field(default_factory=Usage.empty)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_uses(self):
        return [b for b in self.assistant_message.content if b.type == "tool_use"]


class LLMProvider(abc.ABC):
    """Base class for all model providers."""

    #: Canonical name used by config ``provider.type``.
    name: str = "base"

    def __init__(self, config: ProviderConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def stream_completion(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent | Turn]:
        """Stream a completion.

        Yields :class:`StreamEvent` items (request_start, assistant_message
        text deltas, tool_use) and finally a :class:`Turn`.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _api_key(self) -> str:
        return self.config.resolve_api_key()

    def _usage_from(self, usage_map: dict[str, Any], mapping: dict[str, str] | None = None) -> Usage:
        """Build a Usage from a vendor usage dict, optionally remapping keys.

        ``mapping`` maps canonical Usage field names -> vendor key names.
        """
        m = mapping or {}
        def g(canonical: str) -> int:
            key = m.get(canonical, canonical)
            v = usage_map.get(key, 0)
            try:
                return int(v or 0)
            except (TypeError, ValueError):
                return 0

        return Usage(
            input_tokens=g("input_tokens"),
            output_tokens=g("output_tokens"),
            cache_creation_input_tokens=g("cache_creation_input_tokens"),
            cache_read_input_tokens=g("cache_read_input_tokens"),
            cache_write_input_tokens=g("cache_write_input_tokens"),
        )
