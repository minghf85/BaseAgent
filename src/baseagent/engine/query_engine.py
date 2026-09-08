"""The query engine: the core agent loop.

Mirrors the reference ``ref/query.ts`` / ``ref/QueryEngine.ts`` harness loop:

    build messages -> call the model (streaming) -> collect tool_use blocks
    -> execute tools -> append tool_results to history -> repeat until stop
    or maxTurns, with usage accounting and budget guards.

One :class:`QueryEngine` owns one conversation: history and cumulative usage
persist across ``run()`` calls on the same engine. Each ``run()`` yields a
sequence of :class:`~baseagent.types.StreamEvent` (re-streamed over SSE by the
server) and finally a :class:`~baseagent.types.Terminal`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, AsyncGenerator

from ..config import Config
from ..providers.factory import build_provider
from ..providers.base import LLMProvider, Turn
from ..tools.registry import ToolRegistry
from ..tools.base import ToolContext
from ..types import (
    IterationEvent,
    Message,
    RequestStartEvent,
    StreamEvent,
    Terminal,
    ToolResult,
    ToolResultBlock,
    ToolResultEvent,
    UsageEvent,
)
from .context import estimate_context_tokens
from .usage import UsageTracker
from ..extension.extension import ExtensionContext

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, config: Config):
        self.config = config
        self.registry = ToolRegistry(config)
        self.provider: LLMProvider = build_provider(config)
        self.tracker = UsageTracker(config.limits, config.provider.model)
        self.history: list[Message] = []
        self.turn_count = 0
        self.user_prompt_seen = 0
        self._abort = asyncio.Event()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def abort(self) -> None:
        """Request a clean stop at the next iteration/tool boundary."""
        self._abort.set()

    def reset(self) -> None:
        """Clear conversation history and cumulative usage (fresh conversation)."""
        self.history.clear()
        self.turn_count = 0
        self.tracker = UsageTracker(self.config.limits, self.config.provider.model)
        self._abort.clear()

    def provider_tools(self) -> list[dict[str, Any]]:
        """The tool schema set appropriate for the configured provider."""
        ptype = self.config.provider.type
        if ptype == "anthropic":
            return self.registry.anthropic_tools()
        if ptype == "gemini":
            return self.registry.gemini_declarations()
        # openai / openai_compatible / ollama / mock
        return self.registry.openai_tools()

    def _new_run_state(self):
        # A per-run abort latch that clears at the start of each run(), so
        # abort() only affects the in-flight run (a fresh run starts clean).
        return asyncio.Event()

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        max_iterations: int | None = None,
        system_override: str | None = None,
    ) -> AsyncIterator[StreamEvent | Terminal]:
        """Run the agent on ``prompt``. Yields stream events then the Terminal."""
        run_abort = self._new_run_state()
        max_iterations = max_iterations or self.config.limits.max_iterations
        system = system_override if system_override is not None else self.config.prompt.build()

        # Pre-flight confirmation callback hook for permission prompts (unused by
        # default tools; extensible).
        confirm = getattr(self, "confirm_tool", None)

        # Extension lifecycle context (extensions are runtime plugins; see
        # baseagent.extension). meta is empty here — an extension that needs stage
        # descriptors is built with them by its host.
        ext_ctx = ExtensionContext(
            workspace=self.config.workspace,
            turn_count=self.turn_count,
            config=self.config,
        )
        for ext in self.config.extensions:
            _call_hook(ext, "on_run_start", ext_ctx)

        self_history = self.history
        # Append the user prompt as a fresh user message.
        self_history.append(Message.text_message("user", prompt))
        self.user_prompt_seen += 1

        ctx = ToolContext(
            cwd=self.config.workspace,
            max_result_chars=self.config.tools.max_result_chars,
            config=self.config,
            logger=logger,
        )

        try:
            while True:
                # ---- context pre-call guard (optional) --------------------
                if self.config.limits.max_context_tokens > 0:
                    est = estimate_context_tokens(self_history, system)
                    if est > self.config.limits.max_context_tokens:
                        yield Terminal(
                            reason="error",
                            turn_count=self.turn_count,
                            text="",
                            error=(
                                f"Estimated context ({est} tokens) exceeds "
                                f"limits.max_context_tokens ({self.config.limits.max_context_tokens})."
                            ),
                            usage=self.tracker.cumulative,
                        )
                        self._end_extensions(ext_ctx)
                        return

                # ---- budget pre-check (before spending more) -------------
                reason = self.tracker.check_stop()
                if reason:
                    yield Terminal(
                        reason="budget",
                        turn_count=self.turn_count,
                        text="",
                        error=f"Budget limit reached ({reason}).",
                        usage=self.tracker.cumulative,
                    )
                    self._end_extensions(ext_ctx)
                    return

                yield RequestStartEvent(iteration=self.turn_count + 1)

                # ---- call the model --------------------------------------
                turn: Turn
                async for ev in self.provider.stream_completion(
                    system=system,
                    messages=self_history,
                    tools=self.provider_tools(),
                ):
                    if isinstance(ev, Turn):
                        turn = ev
                    else:
                        yield ev

                self.tracker.record(turn.usage)
                yield UsageEvent(usage=self.tracker.cumulative)

                # Append the assistant message to the conversation history.
                self_history.append(turn.assistant_message)
                self.turn_count += 1

                tool_uses = turn.tool_uses
                if not tool_uses:
                    # Model finished without more tool calls.
                    final_text = _assistant_text(turn.assistant_message)
                    yield Terminal(
                        reason="stop",
                        turn_count=self.turn_count,
                        text=final_text,
                        usage=self.tracker.cumulative,
                    )
                    self._end_extensions(ext_ctx)
                    return

                # ---- execute the requested tools -------------------------
                results: list[ToolResult] = []
                result_blocks: list[Any] = []
                for tu in tool_uses:
                    if confirm is not None:
                        try:
                            ok = await confirm(tu.name, tu.input)
                            if ok is False:
                                results.append(ToolResult.fail("Permission denied by user.", "permission_denied"))
                                result_blocks.append(
                                    ToolResultBlock(tool_use_id=tu.id, content="Permission denied by user.", is_error=True)
                                )
                                yield ToolResultEvent(tool_use_id=tu.id, is_error=True, output="Permission denied by user.")
                                continue
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            pass
                    r = await self.registry.run(ctx, tu.name, tu.input)
                    results.append(r)
                    result_blocks.append(
                        ToolResultBlock(tool_use_id=tu.id, content=r.as_block_content, is_error=r.is_error)
                    )
                    yield ToolResultEvent(tool_use_id=tu.id, is_error=r.is_error, output=r.output)

                # ---- iteration boundary -----------------------------------
                yield IterationEvent(iteration=self.turn_count)
                if run_abort.is_set() or self._abort.is_set():
                    yield Terminal(
                        reason="aborted",
                        turn_count=self.turn_count,
                        text=_assistant_text(turn.assistant_message),
                        usage=self.tracker.cumulative,
                    )
                    self._end_extensions(ext_ctx)
                    return
                if self.turn_count >= max_iterations:
                    yield Terminal(
                        reason="max_turns",
                        turn_count=self.turn_count,
                        text=_assistant_text(turn.assistant_message),
                        usage=self.tracker.cumulative,
                    )
                    self._end_extensions(ext_ctx)
                    return

                # Feed tool results back to the model as a user message.
                self_history.append(Message(role="user", content=result_blocks))

                # Extension hook: observers may validate/commit the workspace
                # and, on failure, inject a redo message for the next turn. The
                # injected message is an ordinary user turn, so the redo still
                # counts against max_iterations (no unbounded loop).
                ext_ctx.turn_count = self.turn_count
                for ext in self.config.extensions:
                    note = _call_hook(ext, "on_turn_end", ext_ctx)
                    if note:
                        self_history.append(Message.text_message("user", note))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # provider/tool failure -> terminal error
            logger.exception("Query engine run failed")
            yield Terminal(
                reason="error",
                turn_count=self.turn_count,
                text="",
                error=str(exc),
                usage=self.tracker.cumulative,
            )
            self._end_extensions(ext_ctx)

    def _end_extensions(self, ctx: ExtensionContext) -> None:
        ctx.turn_count = self.turn_count
        for ext in self.config.extensions:
            _call_hook(ext, "on_run_end", ctx)


def _call_hook(ext, name: str, ctx: ExtensionContext) -> str | None:
    """Invoke a lifecycle hook, swallowing extension errors (never break the run)."""
    fn = getattr(ext, name, None)
    if fn is None:
        return None
    try:
        result = fn(ctx)
    except Exception:  # pragma: no cover - defensive
        logger.exception("extension %r hook %s failed", getattr(ext, "name", ext), name)
        return None
    if name == "on_turn_end":
        return result if isinstance(result, str) and result else None
    return None


def _assistant_text(message: Message) -> str:
    return "\n".join(
        b.text for b in message.content if b.type == "text" and b.text
    )
