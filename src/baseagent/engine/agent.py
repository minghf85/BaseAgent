"""The Agent facade: a clean public API over the QueryEngine loop.

``Agent.submit()`` is an async generator of :class:`StreamEvent | Terminal`,
exactly what the HTTP/SSE server consumes. Convenience method ``run_sync()``
collects the events and returns only the final :class:`Terminal` for scripts.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from ..config import Config
from ..types import StreamEvent, Terminal
from .query_engine import QueryEngine


class Agent:
    """A conversation-level agent. One Agent == one engine conversation."""

    def __init__(self, config: Config):
        self._config = config
        self._engine = QueryEngine(config)

    # -- low-level: replaces config/engine (useful for mocking in tests) ---
    @classmethod
    def from_engine(cls, config: Config, engine: QueryEngine) -> "Agent":
        inst = cls(config)
        inst._engine = engine
        return inst

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: str,
        *,
        max_iterations: Optional[int] = None,
        system_override: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent | Terminal]:
        return self._engine.run(
            prompt,
            max_iterations=max_iterations,
            system_override=system_override,
        )

    # ------------------------------------------------------------------
    # Blocking convenience API
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        max_iterations: Optional[int] = None,
        system_override: Optional[str] = None,
    ) -> Terminal:
        """Await the stream and return only the final Terminal."""
        terminal: Terminal | None = None
        async for item in self.submit(
            prompt, max_iterations=max_iterations, system_override=system_override
        ):
            if isinstance(item, Terminal):
                terminal = item
        assert terminal is not None, "engine terminated without a Terminal"
        return terminal

    def run_sync(
        self,
        prompt: str,
        *,
        max_iterations: Optional[int] = None,
        system_override: Optional[str] = None,
    ) -> Terminal:
        return asyncio.run(
            self.run(prompt, max_iterations=max_iterations, system_override=system_override)
        )

    # ------------------------------------------------------------------
    # Control / introspection
    # ------------------------------------------------------------------

    def abort(self) -> None:
        self._engine.abort()

    def reset(self) -> None:
        self._engine.reset()

    @property
    def history(self):
        return self._engine.history

    @property
    def usage(self):
        return self._engine.tracker.to_dict()

    @property
    def engine(self) -> QueryEngine:
        return self._engine
