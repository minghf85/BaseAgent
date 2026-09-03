"""The Glob tool: find files by a glob pattern within the workspace."""

from __future__ import annotations

import os
from typing import Any

from .base import BaseTool, ToolContext, ToolError, safe_join, ensure_relative
from ..types import ToolResult


class GlobTool(BaseTool):
    name = "Glob"
    description = (
        "Find files by a glob pattern, resolved against the workspace directory. "
        "Returns matching file paths (workspace-relative). Use patterns like "
        "'**/*.py' or 'src/**/*.ts'. Optionally scope with 'path'."
    )
    max_result_chars: int = 10_000
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match, e.g. '**/*.py'.",
            },
            "path": {
                "type": "string",
                "description": "Optional directory to search within (default workspace root).",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ToolError("Glob requires a non-empty 'pattern'.", "bad_parameters")

        import glob as globlib

        sub = kwargs.get("path")
        base = safe_join(ctx.cwd, sub) if sub else ctx.cwd
        if not os.path.isdir(base):
            raise ToolError(f"Glob search path is not a directory: {base!r}", "path_not_found")

        search = pattern if os.path.isabs(pattern) else os.path.join(base, pattern)
        matches = [m for m in globlib.glob(search, recursive=True) if os.path.isfile(m)]
        # Directory matches are ignored; only report files (like the ref tool).
        matches = [m for m in matches if os.path.isfile(m)]

        if not matches:
            return ToolResult.ok(f"No files found matching pattern {pattern!r}.")

        rels = sorted(ensure_relative(ctx.cwd, m) for m in matches)
        # Cap the printed list; report a count otherwise.
        shown = rels[:200]
        body = "\n".join(shown)
        extra = f"\n... and {len(rels) - 200} more." if len(rels) > 200 else ""
        return ToolResult.ok(f"{len(rels)} match(es):\n{body}{extra}")
