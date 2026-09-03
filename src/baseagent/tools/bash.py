"""The Bash tool: run a shell command in the workspace directory."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseTool, ToolContext, ToolError
from ..types import ToolResult

_STDOUT_CUTOFF = 300_000


class BashTool(BaseTool):
    name = "Bash"
    description = (
        "Run a shell command in the workspace directory. Use this for build steps, "
        "tests, git, or any command-line operation. The working directory is the "
        "workspace root. Results are the merged stdout/stderr."
    )
    max_result_chars: int = 30_000
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command(s) to execute.",
            },
            "description": {
                "type": "string",
                "description": "A clear, concise description of what the command does.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (default from config, ~120s).",
            },
        },
        "required": ["command"],
    }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("Bash requires a non-empty 'command' argument.", "bad_parameters")

        timeout_ms = kwargs.get("timeout") or ctx.tool_option(
            "Bash", "timeout_ms", ctx.config.tools.bash_timeout_ms if ctx.config else 120_000
        )
        timeout_s = float(timeout_ms) / 1000.0 if timeout_ms else None
        shell = ctx.tool_option("Bash", "shell", ctx.config.tools.bash_shell if ctx.config else "")
        env = ctx.tool_option("Bash", "env", {}) or {}

        timed_out = False
        try:
            result, raw_output = await _run_command(ctx.cwd, command, shell, timeout_s, env)
        except ToolError:
            raise
        except Exception as exc:  # e.g. FileNotFoundError for a bad shell
            raise ToolError(f"Failed to start command: {exc}", "spawn_error") from exc

        if result.returncode == 124:
            timed_out = True
            output = f"[command timed out after {timeout_s:.1f}s and was killed]"
        else:
            output = raw_output

        rendered = _render(command, output, result.returncode, timed_out)
        if result.returncode != 0:
            return ToolResult.fail(rendered, error_type="command_failed")
        return ToolResult.ok(rendered)


async def _run_command(
    cwd: str, command: str, shell: str, timeout_s: float | None, env: dict[str, str]
) -> tuple[Any, str]:
    import os

    full_env = {**os.environ, **env}
    exec_shell = shell or os.environ.get("COMSPEC", "cmd.exe") if os.name == "nt" else (shell or "/bin/sh")

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        shell=True,
        executable=exec_shell or None,
        env=full_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        raw_out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        class _TimedOut:
            returncode = 124
        return _TimedOut(), ""

    text = raw_out.decode("utf-8", errors="replace")
    if len(text) > _STDOUT_CUTOFF:
        text = text[:_STDOUT_CUTOFF] + "\n\n[... stdout cut off at 300k chars ...]"
    return proc, text


def _render(command: str, output: str, returncode: int, timed_out: bool) -> str:
    head = f"$ {command}"
    if timed_out:
        return head + "\n" + output
    tail = f"\n[exit code: {returncode}]" if returncode != 0 else ""
    return f"{head}\n{output.strip()}{tail}"
