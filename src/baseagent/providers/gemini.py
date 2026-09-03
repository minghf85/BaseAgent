"""Google Gemini adapter (``v1beta`` ``generateContent``, streaming).

Gemini uses a different framing: system is ``system_instruction``, content
parts are ``text`` / ``functionCall`` / ``functionResponse``, tools are
``FunctionDeclaration`` objects, and results stream as ``data:`` lines from
``:streamGenerateContent?alt=sse``. This adapter translates the provider-neutral
format accordingly and normalizes stop reasons to ``tool_use`` / ``stop``.
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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        base_url = config.base_url or DEFAULT_BASE_URL
        key = config.resolve_api_key()
        self.client = make_client(
            base_url,
            "",
            config.timeout_seconds,
            {"x-goog-api-key": key, **config.extra_headers},
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _to_api_contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    parts.append({"text": block.text})
                elif block.type == "tool_use":
                    parts.append({"functionCall": {"name": block.name, "args": block.input}})
                elif block.type == "tool_result":
                    parts.append(
                        {
                            "functionResponse": {
                                "name": block.tool_use_id[:40] or "tool",
                                "response": {"result": _content_to_text(block.content)},
                            }
                        }
                    )
            if parts:
                contents.append({"role": _gemini_role(msg.role), "parts": parts})
        return contents

    def _build_body(self, system, messages, tools) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": self._to_api_contents(messages)}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
        body["generationConfig"] = {
            "temperature": self.config.temperature,
            "topP": self.config.top_p,
            "maxOutputTokens": self.config.max_tokens,
        }
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
        url = f"/models/{self.config.model}:streamGenerateContent"
        yield RequestStartEvent()
        try:
            async with self.client.stream("POST", url, params={"alt": "sse"}, json=body) as resp:
                if resp.status_code >= 400:
                    err_text = (await resp.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"Gemini API error {resp.status_code}",
                        status_code=resp.status_code,
                        body=err_text,
                    )
                text_parts: list[str] = []
                tool_uses: list[ToolUseBlock] = []
                usage: dict[str, Any] = {}

                from .transport import read_sse_lines

                async for event, data in read_sse_lines(resp):
                    payload = parse_json(data)
                    _merge_usage(usage, payload.get("usageMetadata", {}))
                    for cand in payload.get("candidates", []):
                        for part in (cand.get("content") or {}).get("parts", []):
                            text = part.get("text")
                            if text:
                                text_parts.append(text)
                                yield AssistantMessageEvent(text=text)
                            fc = part.get("functionCall")
                            if fc:
                                block = ToolUseBlock(
                                    name=fc.get("name", ""), input=fc.get("args") or {}
                                )
                                tool_uses.append(block)
                                yield ToolUseEvent(name=block.name, tool_use_id=block.id, input=block.input)

                stop_reason = "tool_use" if tool_uses else "stop"
                usage_obj = self._usage_from(usage, {
                    "input_tokens": "promptTokenCount",
                    "output_tokens": "candidatesTokenCount",
                    "cache_read_input_tokens": "cachedContentTokenCount",
                })
                usage_obj.total_tokens = usage_obj._compute_total()
                yield Turn(
                    assistant_message=_M(
                        role="assistant",
                        content=[*[TextBlock(text=t) for t in text_parts], *tool_uses],
                    ),
                    stop_reason=stop_reason,
                    usage=usage_obj,
                )
        except Exception as exc:  # noqa: BLE001
            raise normalize_error(exc)


def _gemini_role(role: str) -> str:
    return "model" if role == "assistant" else "user"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


def _merge_usage(acc: dict[str, Any], meta: dict[str, Any]) -> None:
    for k, v in (meta or {}).items():
        if isinstance(v, dict):
            acc.setdefault(k, {})
            _merge_usage(acc[k], v)
        else:
            try:
                acc[k] = int(v or 0)
            except (TypeError, ValueError):
                pass
