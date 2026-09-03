"""The Read tool: read a file with numbered lines and offset/limit."""

from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolContext, ToolError, safe_join
from ..types import ToolResult


class ReadTool(BaseTool):
    name = "Read"
    description = (
        "Read a file's contents as numbered lines. Provide an optional starting "
        "line 'offset' and a 'limit' on the number of lines to read. Use offset to "
        "page through large files."
    )
    max_result_chars: int = 30_000
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path or workspace-relative path of the file to read."},
            "offset": {"type": "integer", "description": "1-based line number to start from (default 1)."},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default all)."},
        },
        "required": ["file_path"],
    }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path")
        if not isinstance(file_path, str):
            raise ToolError("Read requires a 'file_path' string.", "bad_parameters")
        path = safe_join(ctx.cwd, file_path)

        offset = kwargs.get("offset") or 0
        limit = kwargs.get("limit") or 0
        try:
            offset = int(offset)
            limit = int(limit)
        except (TypeError, ValueError):
            raise ToolError("offset and limit must be integers.", "bad_parameters")

        if not _is_readable_file(path, ctx, must_exist=True):
            raise ToolError(f"File does not exist: {file_path!r}", "file_not_found")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        start = max(offset, 1)
        lines = all_lines[start - 1:] if limit <= 0 else all_lines[start - 1: start - 1 + limit]

        numbered = "".join(f"{i + start:>6}\t{ln}" for i, ln in enumerate(lines))
        output = f"{path} ({len(all_lines)} lines total)"
        if limit > 0:
            output += f", showing lines {start}-{start + len(lines) - 1}"
        output += "\n" + numbered.rstrip("\n")
        return ToolResult.ok(self._truncate(output))


def _is_readable_file(path: str, ctx: ToolContext, must_exist: bool) -> bool:
    import os

    if not os.path.exists(path):
        return False if must_exist else True
    if os.path.isdir(path):
        raise ToolError(f"Path is a directory, not a file: {path!r}", "path_is_directory")
    return True
