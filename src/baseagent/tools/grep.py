"""The Grep tool: regular-expression search over file contents."""

from __future__ import annotations

import os
import re
from typing import Any

from .base import BaseTool, ToolContext, ToolError, safe_join, ensure_relative
from ..types import ToolResult


class GrepTool(BaseTool):
    name = "Grep"
    description = (
        "Search file contents with a regular expression (ripgrep-style). Returns "
        "matching lines with line numbers. Use '-i' for case-insensitive matching, "
        "'glob' to filter by filename, and 'output_mode' to control output."
    )
    max_result_chars: int = 20_000
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression to search for.",
            },
            "path": {
                "type": "string",
                "description": "Optional directory or file to search (default workspace root, recursive).",
            },
            "glob": {
                "type": "string",
                "description": "Optional filename glob to filter files, e.g. '*.py'.",
            },
            "-i": {
                "type": "boolean",
                "description": "Case-insensitive match.",
            },
            "-n": {
                "type": "boolean",
                "description": "Always include line numbers.",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "content (default), files_with_matches, or count.",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ToolError("Grep requires a non-empty 'pattern'.", "bad_parameters")

        ignore_case = bool(kwargs.get("-i", kwargs.get("i", False)))
        line_numbers = bool(kwargs.get("-n", kwargs.get("n", False)))
        output_mode = kwargs.get("output_mode", "content") or "content"
        glob_filter = kwargs.get("glob")
        try:
            flags = re.IGNORECASE if ignore_case else 0
            rx = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolError(f"Invalid regular expression: {exc}", "bad_parameters")

        sub = kwargs.get("path")
        base = safe_join(ctx.cwd, sub) if sub else ctx.cwd
        if not os.path.exists(base):
            raise ToolError(f"Grep search path not found: {base!r}", "path_not_found")

        file_matches: dict[str, list[tuple[int, str]]] = {}
        for filep in _iter_files(base, glob_filter):
            try:
                with open(filep, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            hits = []
            for i, line in enumerate(content.splitlines(), start=1):
                if rx.search(line):
                    hits.append((i, line))
            if hits:
                file_matches[filep] = hits

        return ToolResult.ok(self._truncate(_render(ctx, file_matches, output_mode, line_numbers)))


def _iter_files(base: str, glob_filter: str | None) -> list[str]:
    import fnmatch

    out = []
    if os.path.isfile(base):
        return [base]
    for root, dirs, files in os.walk(base):
        for name in files:
            if glob_filter and not fnmatch.fnmatch(name, glob_filter):
                continue
            out.append(os.path.join(root, name))
    return out


def _render(ctx, file_matches, output_mode, line_numbers) -> str:
    if output_mode == "count":
        lines = [
            f"{ensure_relative(ctx.cwd, f)}: {len(hits)}"
            for f, hits in file_matches.items()
        ]
        return f"{len(file_matches)} file(s) matched:\n" + "\n".join(lines)
    if output_mode == "files_with_matches":
        lines = [ensure_relative(ctx.cwd, f) for f in file_matches]
        return f"{len(lines)} file(s):\n" + "\n".join(lines)

    body = []
    for f, hits in file_matches.items():
        rel = ensure_relative(ctx.cwd, f)
        for i, line in hits[:50]:
            prefix = f"{i}: " if line_numbers or len(hits) > 1 else ""
            body.append(f"{rel}:{i}:{prefix}{line}")
        if len(hits) > 50:
            body.append(f"  ... {len(hits) - 50} more match(es) in {rel}")
    total = sum(len(h) for h in file_matches.values())
    return f"{total} match(es) in {len(file_matches)} file(s):\n" + "\n".join(body)
