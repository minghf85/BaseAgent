"""BaseAgent — a modular, configurable Claude-Code-style agent harness.

Exposes the core public API: config loading and the Agent facade.
"""

from __future__ import annotations

from .config import Config
from .types import (
    Message,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolResult,
    Usage,
    Terminal,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Message",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ToolResult",
    "Usage",
    "Terminal",
    "__version__",
]
