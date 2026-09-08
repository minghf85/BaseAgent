"""Extension host for the agent harness.

An :class:`AgentExtension` is a declarative plugin that the engine calls at
lifecycle boundaries of a run. The harness itself stays provider/task-agnostic:
extensions observe the workspace and may influence the conversation by
injecting a message after a turn (e.g. "your edit failed validation, redo it").

This is the general seam the memory layer (gitmem) plugs into — nothing here
knows about memory or skills; those live in the extension implementations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - avoided at runtime to prevent cycles
    from ..config import Config


@dataclass
class ExtensionContext:
    """Snapshot handed to an extension at each lifecycle boundary.

    ``meta`` is a free-form bag the runner populates with stage descriptors
    (user_id / session_id / query / stage). The harness only guarantees
    ``workspace``, ``turn_count``, and ``config``.
    """

    workspace: str
    turn_count: int
    config: Optional["Config"] = None
    meta: dict[str, Any] = field(default_factory=dict)


class AgentExtension(abc.ABC):
    """Lifecycle hook interface implemented by harness plugins."""

    #: Canonical plugin name (used for logging / config selection).
    name: str = ""

    # ------------------------------------------------------------------
    # Hooks (all optional; default = no-op)
    # ------------------------------------------------------------------

    def on_run_start(self, ctx: ExtensionContext) -> None:
        """Called once before the first model request of a run."""

    def on_turn_end(self, ctx: ExtensionContext) -> Optional[str]:
        """Called after each iteration boundary (tools executed, next request
        pending). Return a non-None string to append it as a user message and
        continue the loop (letting the agent react / redo); return None to let
        the loop carry on unchanged."""

    def on_run_end(self, ctx: ExtensionContext) -> None:
        """Called once when the run finishes (any terminal reason)."""


__all__ = ["AgentExtension", "ExtensionContext"]
