"""Tool registry: builds and selects tools from configuration.

The registry owns the mapping between tool names and their implementation
classes, honours the ``tools.enabled`` list from config, and exposes per-provider
schema renderings (Anthropic / OpenAI / Gemini) for use by provider adapters.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..tools.base import BaseTool, ToolContext, ToolError
from ..types import ToolResult

from .bash import BashTool
from .edit import EditTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .read import ReadTool
from .write import WriteTool

# The six built-in tools (title-cased names, matching the ref tool directories).
BUILTIN_TOOLS: dict[str, type[BaseTool]] = {
    "Bash": BashTool,
    "Edit": EditTool,
    "Read": ReadTool,
    "Write": WriteTool,
    "Glob": GlobTool,
    "Grep": GrepTool,
}


class ToolRegistry:
    """Holds the enabled tool classes and executes tools by name."""

    def __init__(self, config: Config):
        self.config = config
        self.max_result_chars = config.tools.max_result_chars
        self._tools: dict[str, type[BaseTool]] = {}
        for name in config.tools.enabled:
            cls = BUILTIN_TOOLS.get(name) or BUILTIN_TOOLS.get(name.lower())
            if cls is None:
                # Unknown names are ignored so configs remain forward-compatible;
                # a warning is emitted via the logger that the engine installs.
                continue
            self._tools[cls.name] = cls

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._tools)

    def tool(self, name: str) -> type[BaseTool] | None:
        return self._tools.get(name) or self._tools.get(name.lower())

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [cls.to_anthropic_tool() for cls in self._tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [cls.to_openai_schema() for cls in self._tools.values()]

    def gemini_declarations(self) -> list[dict[str, Any]]:
        return [cls.to_gemini_declaration() for cls in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(self, ctx: ToolContext, name: str, tool_input: dict[str, Any]) -> ToolResult:
        cls = self.tool(name)
        if cls is None:
            return ToolResult.fail(
                f"Unknown tool: {name!r}. Available tools: {', '.join(self.names())}",
                error_type="unknown_tool",
            )
        try:
            result = await cls().execute(ctx, **tool_input)
        except ToolError as exc:
            return ToolResult.fail(exc.message, error_type=exc.error_type)
        except TypeError as exc:
            return ToolResult.fail(
                f"Bad arguments for {name}: {exc}", error_type="bad_parameters"
            )
        except Exception as exc:  # unknown tool-side error -> surfaced, not fatal
            return ToolResult.fail(
                f"{name} raised {type(exc).__name__}: {exc}", error_type="tool_crash"
            )
        # Enforce the configured result cap as a hard safety net.
        if result.output and len(result.output) > self.max_result_chars:
            result.output = (
                result.output[: self.max_result_chars]
                + f"\n\n[... result truncated at {self.max_result_chars} chars ...]"
            )
        return result
