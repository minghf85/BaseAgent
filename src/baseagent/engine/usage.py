"""Usage tracking and budget enforcement for a run.

Accumulates per-request usage into a cumulative total and enforces the two
budget guards from config: a total output-token cap (``max_tokens_total``) and a
monetary cap (``max_budget_usd``, driven by the per-model pricing table).

The checks are advisory-pre-flight: the engine asks ``would_exceed`` before
starting the *next* request so it can stop cleanly at a turn boundary rather
than mid-stream.
"""

from __future__ import annotations

from ..config import LimitsConfig
from ..types import Usage


class BudgetExceeded(Exception):
    """Raised (converted to a terminal) when a budget guard trips."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


class UsageTracker:
    def __init__(self, limits: LimitsConfig, model: str):
        self.limits = limits
        self.model = model
        self.cumulative = Usage.empty()
        self.request_count = 0
        self.cost_usd = 0.0
        self._monitor = None  # optional callback (usage) -> None

    def set_monitor(self, cb) -> None:
        self._monitor = cb

    def record(self, usage: Usage) -> None:
        self.cumulative.add(usage)
        self.request_count += 1
        self.cost_usd += self.limits.cost_usd(self.model, usage)
        if self._monitor is not None:
            try:
                self._monitor(self)
            except Exception:
                pass

    def would_exceed_tokens(self, upcoming_output: int = 0) -> bool:
        if self.limits.max_tokens_total <= 0:
            return False
        return (self.cumulative.output_tokens + upcoming_output) > self.limits.max_tokens_total

    def would_exceed_budget(self, upcoming_cost: float = 0.0) -> bool:
        if self.limits.max_budget_usd <= 0:
            return False
        return (self.cost_usd + upcoming_cost) > self.limits.max_budget_usd

    def check_stop(self, upcoming_output: int = 0) -> str | None:
        """Return a terminal reason ('budget_tokens' | 'budget_usd') if a guard
        trips, else None."""
        if self.limits.max_tokens_total > 0 and self.would_exceed_tokens(upcoming_output):
            return "budget_tokens"
        if self.limits.max_budget_usd > 0 and self.would_exceed_budget():
            return "budget_usd"
        return None

    def to_dict(self) -> dict:
        return {
            "requests": self.request_count,
            "usage": self.cumulative.to_dict(),
            "cost_usd": round(self.cost_usd, 6),
            "tokens_limit": self.limits.max_tokens_total,
            "budget_limit_usd": self.limits.max_budget_usd,
        }
