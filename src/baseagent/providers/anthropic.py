"""Anthropic Messages API adapter.

Maps the provider-neutral message model onto ``/v1/messages``. This is the most
direct mapping (the reference code is Anthropic-native): text / tool_use /
tool_result content blocks map 1:1, and streaming reuses the Anthropic event
types (``content_block_start``, ``content_block_delta``, ``message_delta``,
``message_stop``).
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
    Message as _M,
)
from .base import LLMProvider, Turn
from .transport import ProviderError, make_client, normalize_error, parse_json

DEFAULT_BASE_URL = "https://api.anthropic.com"

USAGE_MAPPING = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
    "cache_read_input_tokens": "cache_read_input_tokens",
}


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        base_url = config.base_url or DEFAULT_BASE_URL
        self.client = make_client(
            base_url,
            config.resolve_api_key(),
            config.timeout_seconds,
            {
                "anthropic-version": "2023-06-01",
                "x-api-key": config.resolve_api_key(),
                **config.extra_headers,
            },
        )

    def _to_api_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if not msg.content:
                continue
            blocks = []
            for block in msg.content:
                if block.type == "text":
                    text = block.text
                    if text:
                        blocks.append({"type": "text", "text": text})
                elif block.type == "tool_use":
                    blocks.append(
                        {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                    )
                elif block.type == "tool_result":
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content if isinstance(block.content, list) else block.content or "",
                            "is_error": block.is_error,
                        }
                    )
                elif block.type == "image":
                    blocks.append({"type": "image", "source": block.source})
            if blocks:
                out.append({"role": msg.role, "content": blocks})
        return out

    def _build_body(self, system, messages, tools) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": self._to_api_messages(messages),
            "stream": True,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        if self.config.thinking:
            text_blocks_present = any(
                b.type == "text" and b.text for m in messages for b in m.content
            )
            thinking: dict[str, Any] = {"type": "enabled", "budget_tokens": self.config.thinking_budget_tokens}
            body["thinking"] = thinking
            if text_blocks_present:
                # When thinking is on, a text block is required so the model has
                # a place to put its final answer.
                body["messages"][-1] = dict(body["messages"][-1])
                content = list(body["messages"][-1]["content"])
                content.append({"type": "text", "text": "Continue."})
                body["messages"][-1]["content"] = content
        # Merge any extra body overrides last so users can tune anything.
        body.update(self.config.extra_body)
        return body

    def _map_assistant_content(
        self, blocks: list[dict[str, Any]]
    ) -> list[Any]:
        mapped: list[Any] = []
        for b in blocks:
            t = b.get("type")
            if t == "text":
                mapped.append(TextBlock(text=b.get("text", "")))
            elif t == "tool_use":
                mapped.append(
                    ToolUseBlock(id=b.get("id", ""), name=b.get("name", ""), input=b.get("input", {}))
                )
            elif t == "thinking":
                continue  # extended thinking text is not surfaced as assistant text
            else:
                # Unknown block (e.g. server_tool_use) is skipped.
                continue
        return mapped

    async def stream_completion(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent | Turn]:
        body = self._build_body(system, messages, tools)
        yield RequestStartEvent(iteration=1)
        try:
            async with self.client.stream("POST", "/v1/messages", json=body) as resp:
                if resp.status_code >= 400:
                    err_text = (await resp.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"Anthropic API error {resp.status_code}",
                        status_code=resp.status_code,
                        body=err_text,
                    )
                assistant_blocks: list[dict[str, Any]] = []
                usage: dict[str, Any] = {}
                stop_reason = "stop"
                text_delta_acc = ""

                from .transport import read_sse_lines

                async for event, data in read_sse_lines(resp):
                    payload = parse_json(data)
                    etype = payload.get("type")
                    if etype == "content_block_start":
                        block = payload.get("content_block", {})
                        assistant_blocks.append(block)
                        if block.get("type") == "tool_use":
                            yield ToolUseEvent(
                                name=block.get("name", ""),
                                tool_use_id=block.get("id", ""),
                                input=block.get("input", {}),
                            )
                    elif etype == "content_block_delta":
                        delta = payload.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            text_delta_acc += text
                            yield AssistantMessageEvent(text=text)
                        elif delta.get("type") == "input_json_delta":
                            pass  # tool_use input is assembled server-side
                    elif etype == "message_delta":
                        d = payload.get("delta", {})
                        if d.get("stop_reason"):
                            stop_reason = d["stop_reason"]
                        usage.update(payload.get("usage", {}))
                    elif etype == "message_start":
                        usage.update(payload.get("message", {}).get("usage", {}))
                    elif etype == "message_stop":
                        break

                mapped = self._map_assistant_content(assistant_blocks)
                # If only tool_use blocks were emitted and no text delta was
                # streamed, avoid an empty text block in the assistant message.
                msg = _M(role="assistant", content=mapped)
                yield Turn(
                    assistant_message=msg,
                    stop_reason=stop_reason or ("tool_use" if any(b.type == "tool_use" for b in mapped) else "stop"),
                    usage=self._usage_from(usage, USAGE_MAPPING),
                )
        except Exception as exc:  # noqa: BLE001 - normalize all failures
            raise normalize_error(exc)
