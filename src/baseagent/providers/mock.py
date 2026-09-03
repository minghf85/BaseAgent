"""Mock provider for offline testing and demos (no API key required).

Emulates a model that follows a deterministic script of assistant responses,
cycling through tool-use / text turns and finally stopping. This proves the
engine loop (turn counting, tool execution, usage accumulation, terminal
reasons) end-to-end without any external service.

Config: ``provider.type: mock`` and optional ``provider.extra_body`` with a
``script`` list. Each entry is either:

- ``{"tool": {"name": "Write", "input": {...}}}``  -> emit a tool_use
- ``{"text": "..."}``                               -> emit final text and stop

If no script is given, a sensible default (one Write, then stop) is used.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..config import ProviderConfig
from ..types import (
    AssistantMessageEvent,
    Message,
    RequestStartEvent,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
    ToolUseEvent,
    Usage,
    Message as _M,
)
from .base import LLMProvider, Turn

DEFAULT_SCRIPT: list[dict[str, Any]] = [
    {
        "tool": {
            "name": "Write",
            "input": {"file_path": "mock.txt", "content": "hello from mock provider\n"},
        }
    },
    {"text": "Done. I wrote mock.txt."},
]


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        raw = config.extra_body or {}
        self.script = raw.get("script") or DEFAULT_SCRIPT
        self.call_count = 0

    async def stream_completion(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent | Turn]:
        yield RequestStartEvent()
        step = self.script[self.call_count % len(self.script)]
        self.call_count += 1

        blocks: list[Any] = []
        if "tool" in step:
            spec = step["tool"]
            block = ToolUseBlock(name=spec["name"], input=spec.get("input", {}))
            blocks.append(block)
            yield ToolUseEvent(name=block.name, tool_use_id=block.id, input=block.input)
            stop_reason = "tool_use"
        else:
            text = step.get("text", "")
            if text:
                blocks.append(TextBlock(text=text))
                yield AssistantMessageEvent(text=text)
            stop_reason = "stop"

        usage = Usage(
            input_tokens=100 * self.call_count,
            output_tokens=20 * self.call_count,
            total_tokens=120 * self.call_count,
        )
        yield Turn(
            assistant_message=_M(role="assistant", content=blocks),
            stop_reason=stop_reason,
            usage=usage,
        )
