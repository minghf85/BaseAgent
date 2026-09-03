"""OpenAI-compatible chat completions adapter (functions/streaming).

Also the base for the ```openai_compatible`` provider type (any server exposing
``/chat/completions``) and for Ollama (see ``ollama.py``).
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

DEFAULT_BASE_URL = "https://api.openai.com/v1"

USAGE_MAPPING = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "cache_read_input_tokens": "prompt_tokens_details.cached_tokens",
}


class OpenAIProvider(LLMProvider):
    """Chat-completions adapter. ``base_url`` defaults to OpenAI; set it for
    any OpenAI-compatible server (Together, Groq, vLLM, local proxies, etc.)."""

    name = "openai"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        base_url = config.base_url or DEFAULT_BASE_URL
        self.client = make_client(
            base_url,
            config.resolve_api_key(),
            config.timeout_seconds,
            config.extra_headers,
        )

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _to_api_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if not msg.content:
                continue
            if msg.role == "assistant":
                out.append(self._assistant_to_openai(msg.content))
            else:  # user message, which may hold text and/or tool_result blocks
                out.extend(self._user_to_openai(msg.content))
        return out

    def _assistant_to_openai(self, blocks: list[Any]) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in blocks:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": _json_dumps(block.input)},
                    }
                )
            elif block.type == "image":
                pass  # OpenAI images are composed inline in user content; out of scope.
        msg: dict[str, Any] = {"role": "assistant"}
        msg["content"] = "\n".join(text_parts) if text_parts else None
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _user_to_openai(self, blocks: list[Any]) -> list[dict[str, Any]]:
        """User messages may mix plain text and tool_result blocks. In OpenAI
        the results go out as separate ``role: tool`` messages and any plain
        text becomes its own user message."""
        out: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in blocks:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": _tool_result_text(block),
                    }
                )
        if text_parts:
            out.append({"role": "user", "content": "\n".join(text_parts)})
        return out

    def _build_body(self, system, messages, tools) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": ([{"role": "system", "content": system}] if system else [])
            + self._to_api_messages(messages),
            "stream": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(self.config.extra_body)
        return body

    async def stream_completion(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent | Turn]:
        body = self._build_body(system, messages, tools)
        yield RequestStartEvent()
        try:
            async with self.client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    err_text = (await resp.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"OpenAI API error {resp.status_code}",
                        status_code=resp.status_code,
                        body=err_text,
                    )
                content = ""
                tool_calls_acc: dict[int, dict[str, Any]] = {}
                usage: dict[str, Any] = {}
                finish_reason = "stop"

                from .transport import read_sse_lines

                async for event, data in read_sse_lines(resp):
                    if data == "[DONE]":
                        break
                    payload = parse_json(data)
                    choices = payload.get("choices") or []
                    for ch in choices:
                        delta = ch.get("delta") or {}
                        finish = ch.get("finish_reason")
                        if finish:
                            finish_reason = finish
                        text = delta.get("content")
                        if text:
                            content += text
                            yield AssistantMessageEvent(text=text)
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            slot = tool_calls_acc.setdefault(idx, {
                                "id": "", "name": "", "arguments": ""
                            })
                            fn = tc.get("function") or {}
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                    if payload.get("usage"):
                        usage.update(payload["usage"])

                # Build assistant content blocks (may be empty if just tools).
                blocks: list[Any] = []
                if content:
                    blocks.append(TextBlock(text=content))
                for idx in sorted(tool_calls_acc):
                    slot = tool_calls_acc[idx]
                    try:
                        import json
                        args = json.loads(slot["arguments"] or "{}")
                    except Exception:
                        args = {"_parse_error": slot["arguments"]}
                    blocks.append(
                        ToolUseBlock(
                            id=slot.get("id") or f"call_{idx}",
                            name=slot.get("name") or "",
                            input=args,
                        )
                    )

                stop_reason = _map_stop_reason(finish_reason, bool(blocks and blocks[-1].type == "tool_use"))
                usage_obj = self._usage_from(usage, USAGE_MAPPING)
                # Extract cached tokens nested under prompt_tokens_details.
                details = (usage.get("prompt_tokens_details") or {})
                usage_obj.cache_read_input_tokens += int(details.get("cached_tokens", 0) or 0)
                usage_obj.total_tokens = usage_obj._compute_total()

                yield Turn(
                    assistant_message=_M(role="assistant", content=blocks),
                    stop_reason=stop_reason,
                    usage=usage_obj,
                )
        except Exception as exc:  # noqa: BLE001
            raise normalize_error(exc)


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _tool_result_text(block: Any) -> str:
    if isinstance(block.content, str):
        return block.content
    if isinstance(block.content, list):
        parts = []
        for c in block.content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        return "\n".join(parts)
    return str(block.content)


def _map_stop_reason(finish_reason: str, has_tool_use: bool) -> str:
    if finish_reason == "tool_calls" or has_tool_use:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop":
        return "stop"
    if finish_reason == "content_filter":
        return "stop"
    return "stop"
