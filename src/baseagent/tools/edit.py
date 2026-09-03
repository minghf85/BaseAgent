"""The Edit tool: perform an exact-string replacement in an existing file."""

from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolContext, ToolError, safe_join
from ..types import ToolResult


class EditTool(BaseTool):
    name = "Edit"
    description = (
        "Replace an exact substring in an existing file with new text. The "
        "old_string must match exactly (including whitespace) and must appear "
        "exactly once in the file unless 'replace_all' is set. For partial edits "
        "that span many lines, include enough surrounding context to make "
        "old_string unique."
    )
    max_result_chars: int = 2000
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or workspace-relative path of the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace.",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace every occurrence; otherwise require exactly one match.",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path")
        old_string = kwargs.get("old_string")
        new_string = kwargs.get("new_string", "")
        replace_all = bool(kwargs.get("replace_all", False))

        if not isinstance(file_path, str) or not isinstance(old_string, str):
            raise ToolError("Edit requires 'file_path' and 'old_string'.", "bad_parameters")
        new_string = str(new_string)

        path = safe_join(ctx.cwd, file_path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            raise ToolError(f"File does not exist: {file_path!r}", "file_not_found")
        except IsADirectoryError:
            raise ToolError(f"Path is a directory: {file_path!r}", "path_is_directory")

        if old_string not in content:
            raise ToolError(
                f"old_string not found in {file_path!r}. It must match exactly. "
                "Check whitespace/indentation, or read the file first.",
                "old_string_not_found",
            )

        count = content.count(old_string)
        if count > 1 and not replace_all:
            raise ToolError(
                f"Found {count} occurrences of old_string in {file_path!r}. "
                "Include more surrounding context to make it unique, or set replace_all=true.",
                "multiple_occurrences",
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
            occurrences = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            occurrences = 1

        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)

        prefix = f"Replaced {occurrences} occurrence(s) in {file_path!r}."
        # Show a small before/after excerpt for clarity.
        return ToolResult.ok(self._truncate(f"{prefix}\n{_excerpt(old_string, new_string)}"))


def _excerpt(old_string: str, new_string: str) -> str:
    def one(s: str) -> str:
        lines = s.splitlines()
        if not lines:
            return ""
        if len(lines) > 4:
            return "\n".join(lines[:2]) + "\n...\n" + "\n".join(lines[-2:])
        return "\n".join(lines)

    return f"BEFORE:\n{one(old_string)}\n---\nAFTER:\n{one(new_string)}"
