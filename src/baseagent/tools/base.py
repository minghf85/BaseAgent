"""Base abstractions for tools and the shared tool execution context."""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Config
from ..types import ToolResult


class ToolError(Exception):
    """Raised by a tool when it cannot fulfill the request.

    ``tool_error_type`` becomes the machine-readable ``error_type`` on the
    resulting :class:`ToolResult`.
    """

    def __init__(self, message: str, error_type: str = "tool_error"):
        super().__init__(message)
        self.message = message
        self.error_type = error_type


@dataclass
class ToolContext:
    """Context handed to every tool invocation.

    Exposes the workspace root (the sandbox) and config-controlled knobs so a
    tool can enforce path containment and honour per-tool options.
    """

    cwd: str
    max_result_chars: int = 30000
    config: Config | None = None
    logger: Any = None  # a logging.Logger or None

    def tool_option(self, name: str, key: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        return self.config.tools.tool_option(name).get(key, default)


class BaseTool(abc.ABC):
    """Interface every tool implements.

    Subclasses declare ``name`` (title case, matching the ref tools), a human
    description, and a JSON Schema ``input_schema`` describing their arguments.
    ``execute()`` receives the resolved kwargs plus a :class:`ToolContext` and
    returns the text that will be sent back to the model as a ``tool_result``.
    """

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    # An execution-side maximum on result length; guards against runaway output.
    max_result_chars: int = 30000

    @abc.abstractmethod
    async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        ...

    # ------------------------------------------------------------------
    # Registry/metadata helpers
    # ------------------------------------------------------------------

    @classmethod
    def to_openai_schema(cls) -> dict[str, Any]:
        """The ``tools`` entry consumed by OpenAI-compatible providers."""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.input_schema,
                # Older OpenAI-compatible endpoints reject strict mode; keep off
                # so the schema (with optionals) is accepted everywhere.
                "strict": False,
            },
        }

    @classmethod
    def to_gemini_declaration(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": cls.input_schema,
        }

    @classmethod
    def to_anthropic_tool(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.input_schema,
        }

    # A tool may register a truncation hook so it can shorten output before
    # sending to the model. The default truncates at ``max_result_chars``.
    def _truncate(self, text: str) -> str:
        limit = self.max_result_chars
        if limit > 0 and len(text) > limit:
            return (
                text[:limit]
                + f"\n\n[... output truncated at {limit} characters ...]"
            )
        return text


# ---------------------------------------------------------------------------
# Path sandbox helpers (spirit of the ref pathValidation / bashSecurity)
# ---------------------------------------------------------------------------

def safe_join(cwd: str, path: str) -> str:
    """Resolve ``path`` against ``cwd`` and enforce containment within it."""
    raw = os.path.expanduser(path)
    if os.path.isabs(raw):
        resolved = os.path.abspath(raw)
    else:
        resolved = os.path.abspath(os.path.join(cwd, raw))
    base = os.path.abspath(cwd)
    try:
        common = os.path.commonpath([base, resolved])
    except ValueError:
        common = ""
    if common != base:
        raise ToolError(
            f"Path is outside the workspace sandbox: {resolved!r}",
            error_type="path_outside_sandbox",
        )
    return resolved


def ensure_relative(cwd: str, path: str) -> str:
    """Return a workspace-relative display path (for tool output readability)."""
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(cwd))
    except ValueError:
        return path
