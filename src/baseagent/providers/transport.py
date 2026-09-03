"""HTTP transport helpers shared by provider adapters.

Provides an httpx client factory (with per-provider base URL / headers / timeout)
and a small streaming SSE reader that turns a ``text/event-stream`` response body
into a sequence of ``(event, data)`` pairs. Keeps all network concerns in one
place so adapters stay thin.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx


class ProviderError(Exception):
    """A normalized error from a provider request (auth, HTTP, network)."""

    def __init__(self, message: str, status_code: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body


def make_client(base_url: str, api_key: str, timeout: float, extra_headers: dict[str, str]) -> httpx.AsyncClient:
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    headers.update(extra_headers)
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    )


async def read_sse_lines(
    resp: httpx.Response,
) -> AsyncIterator[tuple[str, str]]:
    """Yield ``(event, data)`` pairs from an SSE streamed response body.

    Handles CRLF/LF splitting, multi-line ``data:`` fields, and ``event:``
    overrides. The caller is responsible for closing the response.
    """
    event = "message"
    data_parts: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            # Dispatch a complete event.
            if data_parts:
                yield event, "\n".join(data_parts)
            event = "message"
            data_parts = []
            continue
        if line.startswith(":"):
            continue  # comment / keep-alive
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].lstrip())
        # Other SSE fields (id, retry) are ignored.
    if data_parts:
        yield event, "\n".join(data_parts)


def parse_json(data: str) -> Any:
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Invalid JSON in stream payload: {exc}", body=data) from exc


def normalize_error(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text[:2000]
        return ProviderError(f"HTTP {status} from provider", status_code=status, body=body)
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError("Request timed out")
    if isinstance(exc, (httpx.TransportError, httpx.NetworkError, httpx.ConnectError)):
        return ProviderError(f"Network error: {exc}")
    return ProviderError(str(exc))
