"""The Write tool: create or overwrite a file with arbitrary content."""

from __future__ import annotations

import os
from typing import Any

from .base import BaseTool, ToolContext, ToolError, safe_join
from ..types import ToolResult


class WriteTool(BaseTool):
    name = "Write"
    description = (
        "Write text to a file, creating it (and parent directories) or fully "
        "overwriting it. Use this for creating a new file or replacing a file in "
        "one shot; use Edit for targeted changes to an existing file."
    )
    max_result_chars: int = 1000
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or workspace-relative path of the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The complete file content to write.",
            },
        },
        "required": ["file_path", "content"],
    }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path")
        content = kwargs.get("content", "")
        if not isinstance(file_path, str):
            raise ToolError("Write requires a 'file_path' string.", "bad_parameters")
        if content is None:
            content = ""
        content = str(content)

        path = safe_join(ctx.cwd, file_path)
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

        n = content.count("\n") + (1 if content else 0)
        return ToolResult.ok(f"Wrote {n} line(s) to {file_path!r} ({len(content)} chars).")
